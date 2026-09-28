//! The history-independent slab. **This is where ADR-0003 becomes code.**
//!
//! The whole security argument for the historical cohort reduces to one property of this file:
//!
//! > Two slabs containing the same *set* of entries are byte-identical, modulo nonces and
//! > dummy fill, **regardless of the order those entries were inserted**.
//!
//! Three mechanisms together produce it, and all three are required:
//!
//! 1. **Address by identity.** `slot = PRF(K_idx, tag) mod slot_count`. Position is a function
//!    of the entry's tag, which derives from a *random* `MemoryId` — never a counter, never a
//!    content hash.
//! 2. **Robin Hood collision resolution with a total order on tags.** Plain linear probing is
//!    *not* history-independent: of two entries colliding on a home slot, the one that arrived
//!    first keeps it, so the layout encodes arrival order. Robin Hood — the entry with the
//!    greater probe distance displaces the lesser, ties broken by tag order — is *uniquely
//!    represented*: the final arrangement depends only on the set.
//! 3. **Full preallocation and indistinguishable dummies.** The file is `n_max`-sized from
//!    `create`, and every slot — live or dummy — is the same size and drawn from the same
//!    distribution.
//!
//! The property is machine-checked by [`tests::layout_is_independent_of_insertion_order`],
//! a direct instance of Definition 1(iii) in `THREAT-MODEL.md`.
//!
//! # On-disk layout
//!
//! ```text
//! [ header: 4096 B, plaintext ]     magic, version, slot_count, slot_size, block_size
//! [ slot 0 ][ slot 1 ] ... [ slot slot_count-1 ]
//!
//! slot = [ tag: 16 B ][ sealed: nonce 24 B | ciphertext block_size | MAC 16 B ]
//! ```
//!
//! **The header is plaintext, deliberately.** It holds only geometry, every field of which the
//! leakage profile already declares observable (file sizes are `n_max`-derived constants, and
//! `n_max` is a public parameter). Encrypting public values would be theatre.
//!
//! **Tags are stored in the clear, also deliberately.** A tag is `PRF(K_tag, random_id)`: 16
//! uniformly random bytes to anyone without the key, unlinkable to content. Storing them in the
//! clear lets Robin Hood probing run without decrypting every slot it walks — and since the
//! adversary cannot compute `PRF(K_idx, tag)`, it cannot turn a tag into a home slot, so it
//! cannot tell a displaced entry from an undisplaced one.
//!
//! **Occupancy is not on disk here.** Which slots are live is a secret (corpus size at slot
//! granularity), so it lives encrypted in the manifest and in memory while unlocked. That is
//! why [`Slab::open`] takes the occupancy set as an argument rather than scanning for it.
//!
//! # Known residual
//!
//! Robin Hood displacement cascades are **data-dependent in length**: inserting into a crowded
//! neighbourhood writes more slots than inserting into an empty one. The batch is padded to a
//! fixed *record* count, but not to a fixed *slot-write* count. This is a real, currently
//! unmitigated channel and is tracked as a `TBD` row in `docs/leakage-profile.md`. The fix is
//! to pad each batch to a fixed slot-write budget; it belongs with the engine, which owns the
//! budget parameter.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use mnemos_core::error::{Error, Result};
use mnemos_core::traits::{BlobStore, Record};
use mnemos_core::types::{Epoch, SlotAddr, Tag};
use mnemos_core::Params;
use mnemos_crypto::aead::{dummy_record, MAC_LEN, NONCE_LEN};
use mnemos_crypto::kdf::DerivedKey;
use mnemos_crypto::prf;

/// Bytes of tag stored in the clear at the head of each slot.
pub const TAG_LEN: usize = 16;
/// Fixed plaintext header size.
pub const HEADER_LEN: usize = 4096;

const MAGIC: &[u8; 8] = b"MNEMOS01";
const FORMAT_VERSION: u32 = 1;

/// Bytes one slot occupies, for a single-block record.
#[must_use]
pub fn slot_size(block_size: usize) -> usize {
    TAG_LEN + NONCE_LEN + block_size + MAC_LEN
}

/// Fixed-layout, preallocated blob store.
pub struct Slab {
    file: File,
    slot_count: u64,
    slot_size: usize,
    /// Live-slot set. **In memory only** — persisted encrypted in the manifest, never in the
    /// slab, because occupancy is a secret and the slab is what the adversary reads.
    occupied: Vec<bool>,
    /// Cached clear tags, one per slot, mirroring the head of each slot so probing stays in
    /// memory.
    tags: Vec<Tag>,
    /// Where compaction resumes. Advances by a fixed budget each epoch, independently of how
    /// much is actually reclaimable.
    compaction_cursor: u64,
}

impl Slab {
    /// Create and fully preallocate a slab, filling every slot with dummy material.
    ///
    /// Preallocation is not a performance choice: it makes corpus growth within `n_max`
    /// invisible. Crossing `n_max` triggers a doubling, a loud declared event
    /// (`docs/leakage-profile.md`, row `Grow`).
    ///
    /// # Errors
    /// I/O failures, or insufficient disk for the full preallocation.
    pub fn create(path: &Path, params: &Params) -> Result<Self> {
        let slot_count = params.index.n_max;
        let ssz = slot_size(params.crypto.block_size_bytes as usize);

        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)?;

        let mut header = vec![0u8; HEADER_LEN];
        header[..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        header[12..20].copy_from_slice(&slot_count.to_le_bytes());
        header[20..24].copy_from_slice(&u32::try_from(ssz).unwrap_or(0).to_le_bytes());
        header[24..28].copy_from_slice(&params.crypto.block_size_bytes.to_le_bytes());
        file.write_all(&header)?;

        // Every slot starts as dummy material. Live and dummy slots must be drawn from the same
        // distribution, so this is uniform random bytes — not zeros, and not an encryption of
        // zeros, either of which would be distinguishable to a key holder and would make "how
        // many dummies are there" a question with an answer.
        let mut tags = Vec::with_capacity(slot_count as usize);
        for _ in 0..slot_count {
            let d = dummy_record(ssz);
            file.write_all(&d)?;
            let mut t = [0u8; TAG_LEN];
            t.copy_from_slice(&d[..TAG_LEN]);
            tags.push(Tag(t));
        }
        file.flush()?;

        Ok(Self {
            file,
            slot_count,
            slot_size: ssz,
            occupied: vec![false; slot_count as usize],
            tags,
            compaction_cursor: 0,
        })
    }

    /// Open an existing slab. `occupied` comes from the encrypted manifest.
    ///
    /// # Errors
    /// I/O, or [`Error::Format`] on magic, version, or geometry mismatch.
    ///
    /// # Panics
    /// If the header is shorter than `HEADER_LEN`, which `read_exact` already rejects.
    pub fn open(path: &Path, params: &Params, occupied: Vec<bool>) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;

        let mut header = vec![0u8; HEADER_LEN];
        file.read_exact(&mut header)?;
        if &header[..8] != MAGIC {
            return Err(Error::Format("bad magic — not a mnemos slab".into()));
        }
        let version = u32::from_le_bytes(header[8..12].try_into().expect("4 bytes"));
        if version != FORMAT_VERSION {
            return Err(Error::Format(format!(
                "slab format version {version}; this build expects {FORMAT_VERSION}"
            )));
        }
        let slot_count = u64::from_le_bytes(header[12..20].try_into().expect("8 bytes"));
        let ssz = u32::from_le_bytes(header[20..24].try_into().expect("4 bytes")) as usize;

        if ssz != slot_size(params.crypto.block_size_bytes as usize) {
            return Err(Error::Format(
                "slab slot size disagrees with params.crypto.block_size_bytes".into(),
            ));
        }
        if occupied.len() as u64 != slot_count {
            return Err(Error::Format(
                "manifest occupancy bitmap length does not match slab slot count".into(),
            ));
        }

        let mut tags = Vec::with_capacity(slot_count as usize);
        for i in 0..slot_count {
            file.seek(SeekFrom::Start(HEADER_LEN as u64 + i * ssz as u64))?;
            let mut t = [0u8; TAG_LEN];
            file.read_exact(&mut t)?;
            tags.push(Tag(t));
        }

        Ok(Self {
            file,
            slot_count,
            slot_size: ssz,
            occupied,
            tags,
            compaction_cursor: 0,
        })
    }

    /// The live-slot bitmap, for persisting into the manifest.
    #[must_use]
    pub fn occupancy(&self) -> &[bool] {
        &self.occupied
    }

    /// Live entry count.
    #[must_use]
    pub fn live_count(&self) -> u64 {
        self.occupied.iter().filter(|b| **b).count() as u64
    }

    /// Payload bytes per slot (everything after the clear tag).
    #[must_use]
    pub fn payload_len(&self) -> usize {
        self.slot_size - TAG_LEN
    }

    /// Locate a live tag, or `None`. Follows the same probe walk as insertion.
    #[must_use]
    pub fn lookup(&self, k_idx: &[u8; 32], tag: Tag) -> Option<SlotAddr> {
        let mut slot = self.home_of(k_idx, tag);
        for _ in 0..self.slot_count {
            if self.occupied[slot as usize] && prf::tags_equal(self.tags[slot as usize], tag) {
                return Some(SlotAddr(slot));
            }
            if !self.occupied[slot as usize] {
                return None;
            }
            slot = (slot + 1) % self.slot_count;
        }
        None
    }

    /// Mark a slot free and overwrite it with dummy material.
    ///
    /// # Errors
    /// I/O failures.
    pub fn free(&mut self, addr: SlotAddr) -> Result<()> {
        let d = dummy_record(self.slot_size);
        let mut t = [0u8; TAG_LEN];
        t.copy_from_slice(&d[..TAG_LEN]);
        self.write_slot_raw(addr.0, Tag(t), &d[TAG_LEN..])?;
        self.occupied[addr.0 as usize] = false;
        Ok(())
    }

    fn probe_distance(&self, k_idx: &[u8; 32], slot: u64) -> u64 {
        let home = self.home_of(k_idx, self.tags[slot as usize]);
        (slot + self.slot_count - home) % self.slot_count
    }

    fn home_of(&self, k_idx: &[u8; 32], tag: Tag) -> u64 {
        prf::home_slot(&DerivedKey(*k_idx), tag, self.slot_count).0
    }

    fn write_slot_raw(&mut self, slot: u64, tag: Tag, payload: &[u8]) -> Result<()> {
        debug_assert_eq!(payload.len(), self.slot_size - TAG_LEN);
        self.file.seek(SeekFrom::Start(
            HEADER_LEN as u64 + slot * self.slot_size as u64,
        ))?;
        self.file.write_all(tag.as_bytes())?;
        self.file.write_all(payload)?;
        self.tags[slot as usize] = tag;
        Ok(())
    }

    fn read_slot_payload(&mut self, slot: u64) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; self.slot_size - TAG_LEN];
        self.file.seek(SeekFrom::Start(
            HEADER_LEN as u64 + slot * self.slot_size as u64 + TAG_LEN as u64,
        ))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Insert one record under Robin Hood probing, returning every slot written.
    ///
    /// # The tie-break is the whole point
    ///
    /// When incumbent and candidate have equal probe distance, the **lexicographically smaller
    /// tag takes the earlier slot**. That single rule makes the arrangement uniquely
    /// represented: without it, "whoever got here first stays" puts arrival order back into the
    /// layout and the historical cohort becomes distinguishable again.
    ///
    /// # Errors
    /// [`Error::CapacityExceeded`] when the table is full; I/O failures.
    pub fn insert_robin_hood(
        &mut self,
        k_idx: &[u8; 32],
        tag: Tag,
        payload: &[u8],
    ) -> Result<Vec<SlotAddr>> {
        let mut cur_tag = tag;
        let mut cur_payload = payload.to_vec();
        let mut slot = self.home_of(k_idx, cur_tag);
        let mut dist: u64 = 0;
        let mut touched = Vec::new();

        for _ in 0..self.slot_count {
            if !self.occupied[slot as usize] {
                self.write_slot_raw(slot, cur_tag, &cur_payload)?;
                self.occupied[slot as usize] = true;
                touched.push(SlotAddr(slot));
                return Ok(touched);
            }

            let incumbent_tag = self.tags[slot as usize];
            let incumbent_dist = self.probe_distance(k_idx, slot);

            // Displace when we are "poorer" (further from home), or on an exact tie when our
            // tag sorts first. Both branches are deterministic functions of the tag set alone.
            let displace = dist > incumbent_dist
                || (dist == incumbent_dist && cur_tag.as_bytes() < incumbent_tag.as_bytes());

            if displace {
                let evicted_payload = self.read_slot_payload(slot)?;
                self.write_slot_raw(slot, cur_tag, &cur_payload)?;
                touched.push(SlotAddr(slot));
                cur_tag = incumbent_tag;
                cur_payload = evicted_payload;
                dist = incumbent_dist;
            }

            slot = (slot + 1) % self.slot_count;
            dist += 1;
        }

        Err(Error::CapacityExceeded {
            live: self.live_count(),
            n_max: self.slot_count,
        })
    }

    /// Overwrite one slot with dummy material **without** changing its occupancy.
    ///
    /// Used to pad a flush up to a fixed slot-write budget. Targets a free slot so no live
    /// entry is destroyed; if the chosen slot is live it is left alone and the write lands on
    /// the next free one, so the operation is always a real disk write of the same shape as a
    /// genuine one.
    ///
    /// # Errors
    /// I/O failures.
    pub fn write_padding_slot(&mut self, hint: u64) -> Result<SlotAddr> {
        for i in 0..self.slot_count {
            let slot = (hint + i) % self.slot_count;
            if !self.occupied[slot as usize] {
                let d = dummy_record(self.slot_size);
                let mut t = [0u8; TAG_LEN];
                t.copy_from_slice(&d[..TAG_LEN]);
                self.write_slot_raw(slot, Tag(t), &d[TAG_LEN..])?;
                return Ok(SlotAddr(slot));
            }
        }
        Err(Error::CapacityExceeded {
            live: self.live_count(),
            n_max: self.slot_count,
        })
    }

    /// Rewrite exactly `budget` slots from the compaction cursor, whatever is reclaimable.
    ///
    /// # Errors
    /// I/O failures.
    pub fn compact_with_budget(&mut self, budget: u64) -> Result<Vec<SlotAddr>> {
        let ssz = self.slot_size;
        // The slots actually rewritten, returned so the caller can announce them.
        //
        // Reporting only the *budget* used to be enough, and it was not: compaction writes to
        // specific file offsets, an adversary watching the slab sees exactly which ones, and an
        // event saying "four slots" describes strictly less than the disk does. The cursor is
        // data-independent, so this discloses nothing new -- but "it happens to be harmless" is
        // an argument that has to be made about a modelled channel, not used to justify leaving
        // one unmodelled. See `tests/channel_enumeration.rs`.
        let mut touched = Vec::new();
        for _ in 0..budget {
            let slot = self.compaction_cursor % self.slot_count;
            if !self.occupied[slot as usize] {
                let d = dummy_record(ssz);
                let mut t = [0u8; TAG_LEN];
                t.copy_from_slice(&d[..TAG_LEN]);
                self.write_slot_raw(slot, Tag(t), &d[TAG_LEN..])?;
                touched.push(SlotAddr(slot));
            }
            self.compaction_cursor = (self.compaction_cursor + 1) % self.slot_count;
        }
        self.file.flush()?;
        Ok(touched)
    }
}

impl BlobStore for Slab {
    fn home_slot(&self, k_idx: &[u8; 32], tag: Tag) -> SlotAddr {
        SlotAddr(self.home_of(k_idx, tag))
    }

    fn write_batch(
        &mut self,
        k_idx: &[u8; 32],
        _epoch: Epoch,
        records: &[Record],
    ) -> Result<Vec<SlotAddr>> {
        // Records arrive already dummy-padded to `epoch_writes` and PRF-shuffled, so neither
        // the count nor the order carries information. See the module-level "Known residual"
        // note for what displacement cascades still leak.
        let mut touched = Vec::new();
        for r in records {
            touched.extend(self.insert_robin_hood(k_idx, r.tag, &r.ciphertext)?);
        }
        self.file.flush()?;
        Ok(touched)
    }

    fn read_slot(&self, addr: SlotAddr) -> Result<Vec<u8>> {
        if addr.0 >= self.slot_count {
            return Err(Error::Format("slot address out of range".into()));
        }
        // Dummy slots read successfully and are indistinguishable from live ones — the point,
        // not a bug.
        let mut f = self.file.try_clone()?;
        let mut buf = vec![0u8; self.slot_size - TAG_LEN];
        f.seek(SeekFrom::Start(
            HEADER_LEN as u64 + addr.0 * self.slot_size as u64 + TAG_LEN as u64,
        ))?;
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn compact(&mut self, _epoch: Epoch) -> Result<()> {
        // Default budget; the engine calls `compact_with_budget` with the configured value.
        self.compact_with_budget(64).map(|_| ())
    }

    fn capacity(&self) -> u64 {
        self.slot_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn test_params(n_max: u64) -> Params {
        let mut p = Params::discover().expect("params.toml");
        p.index.n_max = n_max;
        p.crypto.block_size_bytes = 64; // keep test slabs small
        p
    }

    /// Build a slab by inserting `tags` in order; return the canonical layout.
    ///
    /// Canonicalisation drops payloads (they differ by nonce) and keeps `(occupied, tag)` per
    /// slot — exactly the "modulo nonces and dummy fill" in the property statement.
    fn build_layout(tags: &[[u8; 16]], n_max: u64, seed: u8) -> Vec<Option<[u8; 16]>> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("slab{seed}.bin"));
        let params = test_params(n_max);
        let mut slab = Slab::create(&path, &params).unwrap();
        let k_idx = [7u8; 32];
        let plen = slab.payload_len();

        for t in tags {
            slab.insert_robin_hood(&k_idx, Tag(*t), &dummy_record(plen))
                .unwrap();
        }

        (0..slab.slot_count as usize)
            .map(|i| {
                if slab.occupied[i] {
                    Some(slab.tags[i].0)
                } else {
                    None
                }
            })
            .collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        /// **The paper's core hypothesis, machine-checked.**
        ///
        /// For any two insertion orders of the same entry set, the resulting slabs must agree
        /// slot-for-slot on `(occupied, tag)`. This is Definition 1(iii) of `THREAT-MODEL.md`
        /// as an executable statement — the test that catches a regression quietly
        /// reintroducing arrival-order dependence.
        #[test]
        fn layout_is_independent_of_insertion_order(
            tags in prop::collection::hash_set(any::<[u8; 16]>(), 1..30),
            rotate in 0usize..30,
        ) {
            let forward: Vec<[u8; 16]> = tags.into_iter().collect();
            let mut shuffled = forward.clone();
            shuffled.rotate_left(rotate % forward.len());
            shuffled.reverse();

            let a = build_layout(&forward, 128, 1);
            let b = build_layout(&shuffled, 128, 2);
            prop_assert_eq!(a, b, "layout must depend only on the SET of tags");
        }
    }

    #[test]
    fn every_inserted_tag_is_findable() {
        let dir = tempfile::tempdir().unwrap();
        let params = test_params(64);
        let mut slab = Slab::create(&dir.path().join("s.bin"), &params).unwrap();
        let k_idx = [3u8; 32];
        let plen = slab.payload_len();

        let tags: Vec<Tag> = (0..40u8).map(|i| Tag([i; 16])).collect();
        for t in &tags {
            slab.insert_robin_hood(&k_idx, *t, &dummy_record(plen))
                .unwrap();
        }
        assert_eq!(slab.live_count(), 40);
        for t in &tags {
            assert!(
                slab.lookup(&k_idx, *t).is_some(),
                "inserted tag must be findable"
            );
        }
    }

    #[test]
    fn preallocated_size_is_independent_of_content() {
        // The leakage-profile claim: "file sizes are n_max-derived constants".
        let dir = tempfile::tempdir().unwrap();
        let params = test_params(64);
        let empty = dir.path().join("empty.bin");
        let full = dir.path().join("full.bin");

        let a = Slab::create(&empty, &params).unwrap();
        let mut b = Slab::create(&full, &params).unwrap();
        let plen = b.payload_len();
        for i in 0..30u8 {
            b.insert_robin_hood(&[1u8; 32], Tag([i; 16]), &dummy_record(plen))
                .unwrap();
        }
        drop(a);
        drop(b);

        assert_eq!(
            std::fs::metadata(&empty).unwrap().len(),
            std::fs::metadata(&full).unwrap().len(),
            "an empty and a populated slab must be the same size on disk"
        );
    }

    #[test]
    fn capacity_exceeded_is_an_error_not_an_infinite_loop() {
        let dir = tempfile::tempdir().unwrap();
        let params = test_params(8);
        let mut slab = Slab::create(&dir.path().join("s.bin"), &params).unwrap();
        let plen = slab.payload_len();
        let mut last = Ok(vec![]);
        for i in 0..20u8 {
            last = slab.insert_robin_hood(&[1u8; 32], Tag([i; 16]), &dummy_record(plen));
            if last.is_err() {
                break;
            }
        }
        assert!(matches!(last, Err(Error::CapacityExceeded { .. })));
    }

    #[test]
    fn reopen_recovers_the_same_layout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.bin");
        let params = test_params(64);
        let k_idx = [5u8; 32];

        let occupancy;
        let before;
        {
            let mut slab = Slab::create(&path, &params).unwrap();
            let plen = slab.payload_len();
            for i in 0..20u8 {
                slab.insert_robin_hood(&k_idx, Tag([i; 16]), &dummy_record(plen))
                    .unwrap();
            }
            occupancy = slab.occupancy().to_vec();
            before = slab.tags.clone();
        }

        let reopened = Slab::open(&path, &params, occupancy).unwrap();
        assert_eq!(reopened.tags, before, "tags must survive a reopen");
        for i in 0..20u8 {
            assert!(reopened.lookup(&k_idx, Tag([i; 16])).is_some());
        }
    }
}
