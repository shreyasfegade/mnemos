//! The key hierarchy. See `ARCHITECTURE.md` §4 for the diagram.
//!
//! ```text
//!   RK (hardware, non-exportable)
//!    └─ unwraps ─▶ MK (locked arena, zeroised on lock)
//!                   └─ HKDF-SHA-256 ─┬─▶ K_content   entry AEAD
//!                                    ├─▶ K_tag       tag = PRF(K_tag, id ‖ chunk)
//!                                    ├─▶ K_idx       slot addr = PRF(K_idx, tag)
//!                                    ├─▶ K_bucket    centroid table AEAD
//!                                    └─▶ K_e         per-epoch: search tokens, batch keys
//! ```
//!
//! **Why per-epoch keys.** Two reasons, one practical and one theoretical. Practically, an
//! adversary who scrapes one epoch's key from process memory post-τ learns nothing about
//! earlier epochs. Theoretically, per-epoch keys make the per-epoch leakage functions
//! independent, which is what lets Theorem 1's proof decompose the transcript epoch by epoch
//! instead of reasoning about the whole history at once.

use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::{Zeroize, ZeroizeOnDrop};

use mnemos_core::error::{Error, Result};
use mnemos_core::types::Epoch;

/// HKDF `info` strings. Distinct per purpose so no two derived keys can ever collide.
///
/// Do not reuse or reorder these. A collision between, say, `K_tag` and `K_idx` would make
/// slot addresses derivable from tags by anyone holding either — quietly destroying the
/// separation the design depends on.
mod info {
    pub const CONTENT: &[u8] = b"mnemos/v1/content";
    pub const TAG: &[u8] = b"mnemos/v1/tag";
    pub const IDX: &[u8] = b"mnemos/v1/idx";
    pub const BUCKET: &[u8] = b"mnemos/v1/bucket";
    pub const EPOCH_PREFIX: &[u8] = b"mnemos/v1/epoch/";
}

/// A 256-bit derived key. Zeroised on drop; never `Debug`.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct DerivedKey(pub [u8; 32]);

impl std::fmt::Debug for DerivedKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DerivedKey(<redacted>)")
    }
}

/// All keys derived from the master key, resident only while unlocked.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct KeyHierarchy {
    master: [u8; 32],
    /// Entry-content AEAD key.
    pub content: DerivedKey,
    /// Tag PRF key.
    pub tag: DerivedKey,
    /// Slot-address PRF key.
    pub idx: DerivedKey,
    /// Centroid-table AEAD key.
    pub bucket: DerivedKey,
}

impl KeyHierarchy {
    /// Derive the hierarchy from an unwrapped master key.
    ///
    /// The caller must have obtained `master` from [`mnemos_core::traits::KeyStore`] and must
    /// place this value in the locked arena.
    ///
    /// # Errors
    /// [`mnemos_core::Error::Crypto`] on HKDF failure (unreachable for fixed-length output,
    /// but propagated rather than unwrapped so a future change cannot introduce a panic here).
    pub fn derive(master: &[u8; 32]) -> Result<Self> {
        Ok(Self {
            master: *master,
            content: hkdf(master, info::CONTENT)?,
            tag: hkdf(master, info::TAG)?,
            idx: hkdf(master, info::IDX)?,
            bucket: hkdf(master, info::BUCKET)?,
        })
    }

    /// Derive the key for one epoch: `HKDF(MK, "mnemos/v1/epoch/" ‖ e)`.
    ///
    /// # Errors
    /// [`mnemos_core::Error::Crypto`] on HKDF failure.
    pub fn epoch_key(&self, epoch: Epoch) -> Result<DerivedKey> {
        let mut info = Vec::with_capacity(info::EPOCH_PREFIX.len() + 8);
        info.extend_from_slice(info::EPOCH_PREFIX);
        info.extend_from_slice(&epoch.0.to_be_bytes());
        hkdf(&self.master, &info)
    }
}

/// `HKDF-SHA-256(ikm, info)` to 32 bytes, for key hierarchies defined outside this module.
///
/// # Errors
/// [`mnemos_core::Error::Crypto`] on HKDF failure.
pub fn derive_key(ikm: &[u8; 32], info: &[u8]) -> Result<DerivedKey> {
    hkdf(ikm, info)
}

fn hkdf(ikm: &[u8; 32], info: &[u8]) -> Result<DerivedKey> {
    // No salt: the IKM is already a uniformly-random 256-bit master key unwrapped from
    // hardware, so the extract step has nothing to condition. Domain separation comes
    // entirely from `info`, which is why the strings in `mod info` must stay distinct.
    let hk = Hkdf::<Sha256>::new(None, ikm);
    let mut out = [0u8; 32];
    hk.expand(info, &mut out).map_err(|_| Error::Crypto)?;
    Ok(DerivedKey(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_keys_are_pairwise_distinct() {
        // Guards against an info-string collision, which would be silent and catastrophic.
        let h = KeyHierarchy::derive(&[7u8; 32]).unwrap();
        let keys = [&h.content.0, &h.tag.0, &h.idx.0, &h.bucket.0];
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j], "info strings must not collide");
            }
        }
    }

    #[test]
    fn epoch_keys_differ_between_epochs() {
        let h = KeyHierarchy::derive(&[7u8; 32]).unwrap();
        assert_ne!(
            h.epoch_key(Epoch(1)).unwrap().0,
            h.epoch_key(Epoch(2)).unwrap().0
        );
    }
}
