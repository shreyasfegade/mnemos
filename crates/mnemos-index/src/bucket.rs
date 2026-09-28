//! Fixed-capacity, fully-padded, **encrypted** partitions.
//!
//! A bucket occupies `bucket_capacity × slot_len` bytes on disk whether it holds zero entries
//! or all of them. That is the direct answer to the leakage-profile row reading **"per-bucket
//! occupancy: none"** — and it is precisely the row where an unpadded partitioned index reads
//! "entries per partition", which is what the structure-transfer attack (finding F2) consumes
//! to map the store's semantics and then reach backwards into the pre-compromise corpus.
//!
//! Measured: padding takes that attack from `Adv_hist = 0.641` to `0.000` at every injection
//! budget tested. This is the cheapest strong defence in the system, and it is a file-layout
//! decision rather than a cryptographic one.
//!
//! # The slots are encrypted, and for a long time they were not
//!
//! Padding hides *how many* entries a partition holds. It does nothing about *what they are*.
//! An earlier version of this module wrote each slot as `tag ‖ QuantVec::encode()` — the raw
//! 128-bit tag and the raw int8 embedding — straight to disk, unencrypted.
//!
//! That is a total break of the scheme's central claim, and it needs no attack at all: an
//! adversary who reads `index.slab` obtains every stored entry's embedding vector in the clear,
//! and an embedding is a description of meaning. Topic recovery would be immediate and
//! essentially perfect, against both cohorts, whatever the padding did.
//!
//! It survived because every test was consistent with it. Retrieval worked. The padding tests
//! passed — the file really is a constant size. The attack simulations reported
//! `Adv_hist = 0.000` because they model *the design*, in which slots are opaque, rather than
//! the bytes this module was actually writing. Nothing in the suite compared the design to the
//! disk, which is the same failure mode recorded in `the project notes` for the event stream, and the
//! reason `leakage_profile.rs` now derives its table from emitted events instead of from prose.
//!
//! Slots are now sealed with AES-256-GCM under `K_bucket`, with a per-slot random nonce stored
//! beside the ciphertext. Consequences, all of them improvements:
//!
//! - A dummy is simply `slot_len` random bytes. It no longer has to imitate the *structure* of a
//!   real slot, because a real slot no longer has visible structure — which removes an entire
//!   class of defect (see the `dummy_scores_stay_in_the_same_range_as_real_ones` history).
//! - Live and dummy slots are distinguished by whether the AEAD authenticates, so the
//!   distinction requires the key. Without it they are both uniform random strings.
//! - Reads are unchanged in volume: the whole partition is still read, every slot is still
//!   processed, and a failed authentication costs the same as a successful one.

use mnemos_core::error::Result;
use mnemos_core::types::{QuantVec, Tag};
use mnemos_crypto::aead::{dummy_record, open_metadata, seal_metadata};
use mnemos_crypto::kdf::DerivedKey;

/// Nonce bytes stored in front of each slot's ciphertext.
const NONCE_LEN: usize = 12;
/// AES-GCM authentication tag length.
const MAC_LEN: usize = 16;

/// Bytes one bucket slot occupies: nonce, then the sealed `tag ‖ vector`, then its MAC.
#[must_use]
pub fn slot_len(dim: usize) -> usize {
    NONCE_LEN + 16 + QuantVec::encoded_len(dim) + MAC_LEN
}

/// Bytes a whole bucket occupies. A function of `capacity` alone — never of occupancy.
#[must_use]
pub fn bucket_len(capacity: u32, dim: usize) -> usize {
    capacity as usize * slot_len(dim)
}

/// One slot in a partition, in the clear.
///
/// Only ever exists in memory, and only while the store is unlocked. On disk this is a sealed
/// byte string indistinguishable from the random bytes used for a dummy.
#[derive(Clone, Debug)]
pub struct BucketSlot {
    /// Entry tag.
    pub tag: Tag,
    /// Quantised vector.
    pub vector: QuantVec,
}

impl BucketSlot {
    /// Seal to exactly `slot_len(dim)` bytes under `key`.
    ///
    /// The nonce is fresh random per write rather than derived from the slot index. A derived
    /// nonce would repeat whenever a slot is rewritten — compaction, revocation, reuse after a
    /// delete — and nonce reuse under GCM is catastrophic, not merely untidy. Storing 12 bytes
    /// of nonce is the cost of never having to reason about that.
    ///
    /// # Errors
    /// [`mnemos_core::Error::Crypto`] on AEAD failure.
    pub fn seal(&self, key: &DerivedKey, dim: usize) -> Result<Vec<u8>> {
        use rand::RngCore;
        let mut nonce = [0u8; NONCE_LEN];
        rand::rngs::OsRng.fill_bytes(&mut nonce);

        let want = QuantVec::encoded_len(dim);
        let mut plain = Vec::with_capacity(16 + want);
        plain.extend_from_slice(self.tag.as_bytes());
        let mut enc = self.vector.encode();
        enc.resize(want, 0);
        plain.extend_from_slice(&enc[..want]);

        let sealed = seal_metadata(key, &plain, &nonce)?;
        let mut out = Vec::with_capacity(slot_len(dim));
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&sealed);
        debug_assert_eq!(out.len(), slot_len(dim));
        Ok(out)
    }

    /// Open a slot, or `None` if it is a dummy, a freed slot, or otherwise not ours.
    ///
    /// `None` is the *expected* outcome for the majority of slots in a sparse partition; it is
    /// not an error path. Authentication failure is exactly how a live slot is told from a
    /// dummy, and it is the reason that distinction requires the key.
    #[must_use]
    pub fn open(key: &DerivedKey, bytes: &[u8], dim: usize) -> Option<Self> {
        if bytes.len() < slot_len(dim) {
            return None;
        }
        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(&bytes[..NONCE_LEN]);
        let plain = open_metadata(key, &bytes[NONCE_LEN..slot_len(dim)], &nonce).ok()?;

        if plain.len() < 16 {
            return None;
        }
        let mut t = [0u8; 16];
        t.copy_from_slice(&plain[..16]);
        let vector = QuantVec::decode(&plain[16..])?;
        Some(Self {
            tag: Tag(t),
            vector,
        })
    }

    /// A dummy slot: `slot_len` uniformly random bytes.
    ///
    /// Random bytes suffice now that slots are sealed, because a sealed slot *is* a
    /// pseudorandom string. The earlier unencrypted layout forced dummies to imitate the
    /// structure of a real slot — plausible int8 codes, a plausible scale — and getting that
    /// imitation subtly wrong caused two separate defects: dummies that outranked real entries,
    /// and dummies identifiable by an implausible scale. Encryption removes the requirement
    /// rather than making it easier to satisfy.
    #[must_use]
    pub fn dummy_bytes(dim: usize) -> Vec<u8> {
        dummy_record(slot_len(dim))
    }
}

/// A vector for a dummy *record*.
///
/// Distinct from [`BucketSlot::dummy_bytes`], which fills an unused slot. This is for a dummy
/// record that goes through the **normal insert path**: every record in a flush batch is
/// indexed, dummies included, because indexing only the real ones would make the index-write
/// count reveal how many of the batch were real and undo the WAL's padding one layer down.
///
/// It is drawn as a random unit vector and quantised through the production path. Since the
/// slot is sealed, the value is not observable — but it still has to be a *valid* quantised
/// vector so the scoring path treats it uniformly with real entries.
#[must_use]
pub fn dummy_vector(dim: usize) -> QuantVec {
    use mnemos_core::types::Embedding;
    use rand::RngCore;
    let mut rng = rand::rngs::OsRng;

    let raw: Vec<f32> = (0..dim)
        .map(|_| (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
        .collect();
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    QuantVec::quantise(&Embedding(raw.into_iter().map(|x| x / norm).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnemos_core::types::Embedding;

    fn key() -> DerivedKey {
        DerivedKey([0x2Bu8; 32])
    }

    fn slot(seed: f32) -> BucketSlot {
        BucketSlot {
            tag: Tag([7u8; 16]),
            vector: QuantVec::quantise(&Embedding(vec![seed, 0.4, -0.2, 0.6])),
        }
    }

    #[test]
    fn sealed_slot_width_is_fixed() {
        let a = slot(1.0).seal(&key(), 4).unwrap();
        let b = slot(0.1).seal(&key(), 4).unwrap();
        assert_eq!(a.len(), slot_len(4));
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn roundtrip() {
        let s = slot(0.9);
        let bytes = s.seal(&key(), 4).unwrap();
        let back = BucketSlot::open(&key(), &bytes, 4).expect("open");
        assert_eq!(back.tag.as_bytes(), s.tag.as_bytes());
        assert_eq!(back.vector.codes, s.vector.codes);
    }

    #[test]
    fn the_plaintext_vector_is_not_present_in_the_sealed_slot() {
        // THE regression test for the break this module's header documents. The raw encoding of
        // the vector must not appear anywhere in the bytes written to disk.
        let s = slot(0.77);
        let bytes = s.seal(&key(), 4).unwrap();
        let raw = s.vector.encode();
        assert!(
            !bytes.windows(raw.len()).any(|w| w == raw.as_slice()),
            "the plaintext embedding is present in the sealed slot"
        );
        let tag = s.tag.as_bytes();
        assert!(
            !bytes.windows(tag.len()).any(|w| w == tag.as_slice()),
            "the plaintext tag is present in the sealed slot"
        );
    }

    #[test]
    fn a_dummy_does_not_open() {
        // How a live slot is told from a dummy -- and the reason it takes a key to do so.
        assert!(BucketSlot::open(&key(), &BucketSlot::dummy_bytes(8), 8).is_none());
    }

    #[test]
    fn the_wrong_key_does_not_open_a_real_slot() {
        let bytes = slot(0.5).seal(&key(), 4).unwrap();
        assert!(BucketSlot::open(&DerivedKey([9u8; 32]), &bytes, 4).is_none());
    }

    #[test]
    fn tampering_is_detected() {
        let mut bytes = slot(0.5).seal(&key(), 4).unwrap();
        let n = bytes.len();
        bytes[n - 1] ^= 0xFF;
        assert!(
            BucketSlot::open(&key(), &bytes, 4).is_none(),
            "AEAD accepted a modified slot"
        );
    }

    #[test]
    fn two_seals_of_the_same_slot_differ() {
        // Fresh nonce per write. A deterministic slot encoding would let an adversary detect
        // that two partitions hold the same entry, and would make a rewritten slot recognisable
        // as unchanged.
        let s = slot(0.5);
        assert_ne!(s.seal(&key(), 4).unwrap(), s.seal(&key(), 4).unwrap());
    }

    #[test]
    fn bucket_length_depends_only_on_capacity() {
        assert_eq!(bucket_len(64, 768), 64 * slot_len(768));
    }
}
