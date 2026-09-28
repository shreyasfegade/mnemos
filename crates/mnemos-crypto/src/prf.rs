//! HMAC-SHA-256, truncated to 128 bits.
//!
//! Two uses, and the separation between them is load-bearing:
//!
//! - `tag = PRF(K_tag, memory_id ‖ chunk_idx)` — an entry's identity in the store.
//! - `slot = PRF(K_idx, tag) mod slot_count` — **where** it lives.
//!
//! The second is the mechanism of history independence (ADR-0003): physical position is a
//! function of identity, never of arrival time. Since `memory_id` is random rather than a
//! counter or a content hash, position carries no information at all — which is exactly what
//! attack A3 needs to find, and does not.

use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;

use mnemos_core::types::{MemoryId, SlotAddr, Tag};

use crate::kdf::DerivedKey;

type HmacSha256 = Hmac<Sha256>;

/// `HMAC-SHA-256(key, data)`, truncated to 16 bytes.
///
/// Truncation to 128 bits is standard and ample here: a tag collision costs a slot collision,
/// which Robin Hood probing already handles correctly, so the birthday bound at 2^64 entries
/// is far beyond `n_max`.
///
/// Public because `mnemos-baselines` builds Sophos's and Diana's keyword tokens from the same
/// primitive. Sharing it is deliberate: a baseline that used a different PRF would invite the
/// objection that the comparison is measuring the PRF rather than the scheme.
///
/// # Panics
/// Never. `Hmac::new_from_slice` accepts any key length, and the key here is a fixed 32 bytes.
#[must_use]
pub fn mac16(key: &[u8; 32], data: &[u8]) -> [u8; 16] {
    let mut m = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    m.update(data);
    let full = m.finalize().into_bytes();
    let mut out = [0u8; 16];
    out.copy_from_slice(&full[..16]);
    out
}

/// `tag = truncate_128(HMAC-SHA-256(K_tag, memory_id ‖ chunk_idx))`.
#[must_use]
pub fn tag_for(k_tag: &DerivedKey, id: MemoryId, chunk: u32) -> Tag {
    let mut data = [0u8; 20];
    data[..16].copy_from_slice(&id.0);
    data[16..].copy_from_slice(&chunk.to_be_bytes());
    Tag(mac16(&k_tag.0, &data))
}

/// `slot = PRF(K_idx, tag) mod slot_count`.
///
/// The caller resolves collisions with Robin Hood hashing and a deterministic tie-break on the
/// tag, which is what makes the final layout *uniquely represented* — plain linear probing
/// would let the first of two colliding entries keep the home slot, reintroducing exactly the
/// arrival-order dependence this function exists to remove.
///
/// # Panics
/// If `slot_count` is zero. Callers get this from `Params`, which rejects a zero capacity.
#[must_use]
pub fn home_slot(k_idx: &DerivedKey, tag: Tag, slot_count: u64) -> SlotAddr {
    assert!(slot_count > 0, "slot_count must be non-zero");
    let m = mac16(&k_idx.0, tag.as_bytes());
    let v = u64::from_be_bytes(m[..8].try_into().expect("8 bytes"));
    SlotAddr(v % slot_count)
}

/// Deterministic per-epoch shuffle for the order slots are written within a flush batch.
///
/// Without this the write order inside a batch would follow the order the agent produced the
/// entries, which hands a live adversary a within-epoch ordering signal for free — one that
/// epoch batching was supposed to have removed.
///
/// # Panics
/// If `n` exceeds `u64::MAX`, which is unreachable for a batch size.
#[must_use]
pub fn batch_permutation(k_epoch: &DerivedKey, n: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    // Fisher-Yates driven by a PRF stream rather than a system RNG: the permutation must be
    // reproducible from the epoch key so a crash mid-flush can be replayed to the same layout.
    // A fresh random shuffle would make recovery produce a different arrangement, which would
    // break history independence -- two stores with identical content would differ.
    for i in (1..n).rev() {
        let iu = u64::try_from(i).expect("batch index fits in u64");
        let m = mac16(&k_epoch.0, &iu.to_be_bytes());
        let r = u64::from_be_bytes(m[..8].try_into().expect("8 bytes"));
        let j = usize::try_from(r % (iu + 1)).expect("remainder < i fits in usize");
        perm.swap(i, j);
    }
    perm
}

/// Constant-time tag equality.
///
/// **Every** tag comparison in the codebase goes through here. A variable-time `==` on tags is
/// a timing oracle for "does this entry exist", which is a confidentiality break on AS-3 that
/// no amount of encryption elsewhere repairs.
#[must_use]
pub fn tags_equal(a: Tag, b: Tag) -> bool {
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::kdf::DerivedKey;

    fn key(b: u8) -> DerivedKey {
        DerivedKey([b; 32])
    }

    #[test]
    fn tags_are_deterministic_and_key_dependent() {
        let id = MemoryId([3u8; 16]);
        assert_eq!(tag_for(&key(1), id, 0).0, tag_for(&key(1), id, 0).0);
        assert_ne!(tag_for(&key(1), id, 0).0, tag_for(&key(2), id, 0).0);
        assert_ne!(tag_for(&key(1), id, 0).0, tag_for(&key(1), id, 1).0);
    }

    #[test]
    fn slot_address_depends_only_on_identity() {
        // The mechanism of history independence: position is a function of the tag, so it
        // cannot encode when the entry arrived.
        let t = Tag([9u8; 16]);
        assert_eq!(home_slot(&key(1), t, 1024), home_slot(&key(1), t, 1024));
        assert!(home_slot(&key(1), t, 1024).0 < 1024);
    }

    #[test]
    fn batch_permutation_is_a_permutation_and_reproducible() {
        let p1 = batch_permutation(&key(5), 16);
        let p2 = batch_permutation(&key(5), 16);
        assert_eq!(p1, p2, "must replay identically after a crash");
        let mut sorted = p1.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..16).collect::<Vec<_>>());
        assert_ne!(
            p1,
            batch_permutation(&key(6), 16),
            "must depend on the epoch key"
        );
    }

    #[test]
    fn constant_time_equality_agrees_with_the_obvious_answer() {
        let a = Tag([1u8; 16]);
        let b = Tag([1u8; 16]);
        let c = Tag([2u8; 16]);
        assert!(tags_equal(a, b));
        assert!(!tags_equal(a, c));
    }
}
