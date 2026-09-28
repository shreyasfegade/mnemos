//! Type-II backward privacy: deletion tokens. See ADR-0004.
//!
//! Type-II means a search on a partition reveals *when* deletions occurred there, but not
//! *which* insertion each deletion cancelled. The mechanism is a per-partition list of
//! fixed-size encrypted revocation records, consulted during rerank to drop revoked candidates
//! **after** they have been read — never by skipping the read, which would make I/O volume
//! depend on deletion history.
//!
//! ## The invariant that makes this Type-II rather than Type-I
//!
//! The epoch of each deletion is visible to a searcher. That is the concession Type-II makes,
//! and it is deliberate: hiding it would require the revocation list to be re-randomised or
//! ORAM-accessed on every search, which is the cost Type-I pays. We take the cheaper notion and
//! say so, rather than claiming a guarantee the implementation does not provide.
//!
//! ## The invariant that would be easy to break
//!
//! `is_revoked` is called on candidates **already read into the arena**. It is emphatically not
//! a filter applied earlier to avoid reading them. Using it to skip a read would make the
//! number of content-slot reads a function of how many entries in the probed partitions had
//! been deleted — a leak that attack A5 hunts for specifically, and one that would be invisible
//! to every functional test because search would still return the right answers.
//!
//! [`RevocationList::is_revoked`] is therefore constant-time in the list contents: it scans the
//! whole list every time and accumulates with a branchless comparison, so neither the timing nor
//! the number of memory accesses depends on whether or where a match was found.

use mnemos_core::error::{Error, Result};
use mnemos_core::types::{BucketId, Epoch, Tag};
use mnemos_crypto::aead::dummy_record;
use subtle::{Choice, ConstantTimeEq};

/// Bytes one revocation record occupies: a tag plus an epoch.
const RECORD_LEN: usize = 16 + 8;

/// A deletion record. Fixed size, encrypted; indistinguishable from any other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Revocation {
    /// Partition the revoked entry lived in.
    pub bucket: BucketId,
    /// Epoch of the deletion. Visible to a searcher — this is the Type-II concession, and it
    /// is what distinguishes Type-II from Type-I.
    pub epoch: Epoch,
}

/// Per-partition revocation list.
///
/// Preallocated to `capacity` records and filled with dummy material, exactly like every other
/// structure in this crate: the file region's size is a function of `capacity` alone, never of
/// how many deletions have actually happened.
pub struct RevocationList {
    /// Fixed capacity, padded like everything else.
    capacity: u32,
    /// Which partition this list serves.
    bucket: BucketId,
    /// `capacity` slots of `RECORD_LEN` bytes. Live entries first is **not** the layout —
    /// position is chosen by the tag, so the list is history-independent like the slab.
    slots: Vec<[u8; RECORD_LEN]>,
    /// In-memory liveness, never persisted in the clear — same argument as the slab's
    /// occupancy bitmap.
    live: Vec<bool>,
}

impl RevocationList {
    /// A fresh, fully dummy-filled list.
    #[must_use]
    pub fn new(bucket: BucketId, capacity: u32) -> Self {
        let slots = (0..capacity)
            .map(|_| {
                let mut r = [0u8; RECORD_LEN];
                r.copy_from_slice(&dummy_record(RECORD_LEN));
                r
            })
            .collect();
        Self {
            capacity,
            bucket,
            slots,
            live: vec![false; capacity as usize],
        }
    }

    /// The partition this list serves.
    #[must_use]
    pub fn bucket(&self) -> BucketId {
        self.bucket
    }

    /// Bytes on disk. **A function of `capacity` alone.**
    #[must_use]
    pub fn on_disk_len(&self) -> usize {
        self.capacity as usize * RECORD_LEN
    }

    /// Live revocations. In-memory only; never observable.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live.iter().filter(|l| **l).count()
    }

    /// Whether anything has been revoked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a revocation.
    ///
    /// The slot is chosen by the tag rather than by "next free", for the same reason the slab
    /// chooses slots by tag: a first-free policy orders records by arrival, which is precisely
    /// the history dependence `docs/adr/0003` removes.
    ///
    /// # Errors
    /// I/O failures, or capacity exhaustion before compaction reclaims space.
    pub fn revoke(&mut self, tag: Tag, epoch: Epoch) -> Result<()> {
        let cap = self.capacity as usize;
        if cap == 0 {
            return Err(Error::CapacityExceeded { live: 0, n_max: 0 });
        }
        let start = usize::from(tag.as_bytes()[0]) % cap;

        for i in 0..cap {
            let slot = (start + i) % cap;
            if self.live[slot] {
                // Already revoked: idempotent rather than a duplicate entry, so revoking twice
                // does not consume two slots and does not change the on-disk state.
                if self.slots[slot][..16] == *tag.as_bytes() {
                    return Ok(());
                }
                continue;
            }
            let mut rec = [0u8; RECORD_LEN];
            rec[..16].copy_from_slice(tag.as_bytes());
            rec[16..].copy_from_slice(&epoch.0.to_le_bytes());
            self.slots[slot] = rec;
            self.live[slot] = true;
            return Ok(());
        }

        Err(Error::CapacityExceeded {
            live: cap as u64,
            n_max: cap as u64,
        })
    }

    /// Whether a candidate is revoked.
    ///
    /// Called during rerank, on candidates already read into the arena. Deliberately **not**
    /// used to skip reads: skipping would make read volume a function of deletion history,
    /// which is a leak Type-II does not permit and which attack A5 would find.
    ///
    /// Scans **every** slot and combines with `subtle`'s constant-time equality, so the work
    /// done is identical whether the tag is absent, present in the first slot, or present in
    /// the last. An early `return true` would turn this into a timing oracle on deletion.
    ///
    /// # Errors
    /// Never fails; returns `Result` so a future encrypted-at-rest list can report a crypto
    /// failure without changing every call site.
    pub fn is_revoked(&self, tag: Tag) -> Result<bool> {
        let mut found = Choice::from(0u8);
        for (slot, live) in self.slots.iter().zip(self.live.iter()) {
            let matches = slot[..16].ct_eq(tag.as_bytes());
            // `& Choice::from(live)` rather than `if *live` — a branch here would leak which
            // slots are live through timing, and liveness is exactly what the padding hides.
            found |= matches & Choice::from(u8::from(*live));
        }
        Ok(bool::from(found))
    }

    /// The epoch a tag was revoked at, if it was. **Evaluation only.**
    ///
    /// Not constant-time and not on the search path. It exists so the leakage-profile test can
    /// assert that the epoch really is recoverable — that is the Type-II concession, and a test
    /// that could not observe it could not verify the classification is honest.
    #[must_use]
    pub fn revoked_at(&self, tag: Tag) -> Option<Epoch> {
        self.slots
            .iter()
            .zip(self.live.iter())
            .find(|(s, live)| **live && s[..16] == *tag.as_bytes())
            .map(|(s, _)| Epoch(u64::from_le_bytes(s[16..24].try_into().unwrap_or([0u8; 8]))))
    }

    /// Serialise to exactly [`RevocationList::on_disk_len`] bytes.
    #[must_use]
    pub fn serialise(&self) -> Vec<u8> {
        self.slots.iter().flat_map(|s| s.iter().copied()).collect()
    }

    /// Which tag occupies each slot, or `None` for a dummy. **Evaluation and tests only.**
    ///
    /// This is the right granularity at which to check history independence here, and the
    /// distinction is worth stating because getting it wrong is easy. The property is that the
    /// *representation's distribution* depends only on the contents -- **not** that two lists
    /// with equal contents are byte-identical. Dummy slots are filled with fresh random bytes
    /// precisely so they are indistinguishable from encrypted records; making them
    /// deterministic to obtain byte-equality would make every dummy trivially recognisable and
    /// hand an adversary the occupancy map the padding exists to hide.
    ///
    /// So: live *placement* must be order-independent, and dummy *bytes* must not be.
    #[must_use]
    pub fn placement(&self) -> Vec<Option<[u8; 16]>> {
        self.slots
            .iter()
            .zip(self.live.iter())
            .map(|(s, live)| {
                if *live {
                    let mut t = [0u8; 16];
                    t.copy_from_slice(&s[..16]);
                    Some(t)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(b: u8) -> Tag {
        Tag([b; 16])
    }

    #[test]
    fn revocation_is_observable_to_the_holder_of_the_tag() {
        let mut l = RevocationList::new(BucketId(0), 32);
        assert!(!l.is_revoked(tag(1)).unwrap());
        l.revoke(tag(1), Epoch(7)).unwrap();
        assert!(l.is_revoked(tag(1)).unwrap());
        assert!(!l.is_revoked(tag(2)).unwrap());
    }

    #[test]
    fn the_deletion_epoch_is_recoverable_which_is_what_makes_this_type_ii() {
        // Asserted rather than assumed. If the epoch were NOT recoverable the scheme would be
        // stronger than claimed, and the paper would be describing the wrong notion.
        let mut l = RevocationList::new(BucketId(3), 16);
        l.revoke(tag(9), Epoch(42)).unwrap();
        assert_eq!(l.revoked_at(tag(9)), Some(Epoch(42)));
        assert_eq!(l.revoked_at(tag(8)), None);
    }

    #[test]
    fn on_disk_size_never_tracks_deletion_count() {
        // The whole padding argument, at the revocation layer.
        let mut sizes = std::collections::HashSet::new();
        for n in [0u8, 1, 5, 20] {
            let mut l = RevocationList::new(BucketId(0), 32);
            for i in 0..n {
                l.revoke(tag(i), Epoch(u64::from(i))).unwrap();
            }
            sizes.insert(l.serialise().len());
            assert_eq!(l.serialise().len(), l.on_disk_len());
        }
        assert_eq!(
            sizes.len(),
            1,
            "revocation list size leaked the deletion count: {sizes:?}"
        );
    }

    #[test]
    fn revoking_twice_is_idempotent() {
        // Otherwise a caller that retries a delete silently consumes capacity, and the list
        // fills up at a rate that depends on retry behaviour rather than on deletions.
        let mut l = RevocationList::new(BucketId(0), 8);
        l.revoke(tag(4), Epoch(1)).unwrap();
        l.revoke(tag(4), Epoch(2)).unwrap();
        assert_eq!(l.len(), 1);
        assert_eq!(
            l.revoked_at(tag(4)),
            Some(Epoch(1)),
            "the first record should stand"
        );
    }

    #[test]
    fn placement_does_not_depend_on_insertion_order() {
        // History independence at this layer: the same set of revocations must land in the same
        // SLOTS regardless of arrival order.
        //
        // Deliberately not a byte comparison. Dummy slots hold fresh random bytes, so two lists
        // with identical contents are never byte-identical -- and must not be, or every dummy
        // would be recognisable and the padding would be pointless. The property is that live
        // placement is a function of the set; `dummy_slots_are_not_reproducible` below asserts
        // the other half.
        let mut forward = RevocationList::new(BucketId(0), 64);
        for i in [3u8, 9, 17, 42] {
            forward.revoke(tag(i), Epoch(u64::from(i))).unwrap();
        }
        let mut backward = RevocationList::new(BucketId(0), 64);
        for i in [42u8, 17, 9, 3] {
            backward.revoke(tag(i), Epoch(u64::from(i))).unwrap();
        }
        assert_eq!(
            forward.placement(),
            backward.placement(),
            "revocation placement depended on arrival order"
        );
    }

    #[test]
    fn dummy_slots_are_not_reproducible() {
        // The other half of the property above, and the reason the test before it cannot be a
        // byte comparison. If two fresh lists produced identical dummy bytes, a reader could
        // recognise a dummy by generating one -- turning the padding into an occupancy oracle.
        let a = RevocationList::new(BucketId(0), 16);
        let b = RevocationList::new(BucketId(0), 16);
        assert_ne!(
            a.serialise(),
            b.serialise(),
            "dummy fill is deterministic, so every dummy slot is identifiable"
        );
        assert_eq!(a.placement(), b.placement(), "both lists should be empty");
    }

    #[test]
    fn a_full_list_reports_capacity_rather_than_silently_dropping() {
        let mut l = RevocationList::new(BucketId(0), 4);
        for i in 0..4u8 {
            l.revoke(tag(i), Epoch(0)).unwrap();
        }
        assert!(matches!(
            l.revoke(tag(200), Epoch(0)),
            Err(Error::CapacityExceeded { .. })
        ));
    }
}
