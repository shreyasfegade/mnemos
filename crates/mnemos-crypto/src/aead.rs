//! Authenticated encryption.
//!
//! | Use | Cipher | Why |
//! |---|---|---|
//! | Entry contents | XChaCha20-Poly1305 | 192-bit nonce is birthday-safe under random generation, so **no counter is needed**. A counter is state, and state on disk is snapshot leakage — precisely what we are trying to eliminate. |
//! | Manifest, centroid table | AES-256-GCM | AES-NI dominates, and nonce uniqueness is structural (one write per epoch). |
//!
//! ## Padding
//!
//! Plaintext is length-prefixed and padded to a multiple of `crypto.block_size_bytes` (4 KiB)
//! before encryption, so ciphertext length reveals only `⌈len / 4096⌉`. Entries above
//! `max_entry_bytes` spill to a chunk chain, each chunk padded identically.
//!
//! ## AAD
//!
//! The entry's [`Tag`] is the associated data. This binds a ciphertext to its slot identity, so
//! an adversary who relocates a ciphertext to a different slot gets an authentication failure
//! rather than a successful decryption in the wrong place.

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce as GcmNonce};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use rand::RngCore;

use mnemos_core::error::{Error, Result};
use mnemos_core::types::Tag;

use crate::kdf::DerivedKey;

/// Bytes of XChaCha20-Poly1305 nonce prepended to every sealed record.
pub const NONCE_LEN: usize = 24;
/// Poly1305 tag length.
pub const MAC_LEN: usize = 16;
/// Length prefix inside the padded plaintext.
const LEN_PREFIX: usize = 4;

/// Encrypt entry content: length-prefix, pad, then seal with a fresh random nonce.
///
/// Output layout is `nonce ‖ ciphertext ‖ mac`, where the ciphertext covers a whole number of
/// blocks.
///
/// # Errors
/// [`Error::Crypto`] on failure — deliberately opaque, since distinguishing failure modes is a
/// decryption oracle.
pub fn seal_content(
    key: &DerivedKey,
    plaintext: &[u8],
    aad: Tag,
    block_size: usize,
) -> Result<Vec<u8>> {
    if block_size == 0 || !block_size.is_power_of_two() {
        return Err(Error::Crypto);
    }
    let padded = pad(plaintext, block_size)?;

    let cipher = XChaCha20Poly1305::new(key.0.as_ref().into());
    let mut nonce_bytes = [0u8; NONCE_LEN];
    rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = XNonce::from_slice(&nonce_bytes);

    let ct = cipher
        .encrypt(
            nonce,
            Payload {
                msg: &padded,
                aad: aad.as_bytes(),
            },
        )
        .map_err(|_| Error::Crypto)?;

    let mut out = Vec::with_capacity(NONCE_LEN + ct.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ct);
    Ok(out)
}

/// Decrypt entry content **into caller-provided arena memory** and strip padding.
///
/// Takes `out` rather than returning a `Vec` on purpose: a returned `Vec` would be a heap
/// allocation outside the locked arena, and TA-4 says plaintext lives in the arena or nowhere.
///
/// Returns the plaintext length written to `out`.
///
/// # Errors
/// [`Error::Crypto`] on authentication failure or if `out` is too small.
pub fn open_content(key: &DerivedKey, sealed: &[u8], aad: Tag, out: &mut [u8]) -> Result<usize> {
    if sealed.len() < NONCE_LEN + MAC_LEN {
        return Err(Error::Crypto);
    }
    let (nonce_bytes, ct) = sealed.split_at(NONCE_LEN);

    let cipher = XChaCha20Poly1305::new(key.0.as_ref().into());
    let padded = cipher
        .decrypt(
            XNonce::from_slice(nonce_bytes),
            Payload {
                msg: ct,
                aad: aad.as_bytes(),
            },
        )
        .map_err(|_| Error::Crypto)?;

    let plain = unpad(&padded)?;
    if plain.len() > out.len() {
        return Err(Error::Crypto);
    }
    out[..plain.len()].copy_from_slice(plain);
    Ok(plain.len())
}

/// Encrypt a metadata structure (manifest, centroid table) with AES-256-GCM.
///
/// # Errors
/// [`Error::Crypto`] on failure.
pub fn seal_metadata(key: &DerivedKey, plaintext: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new(key.0.as_ref().into());
    cipher
        .encrypt(GcmNonce::from_slice(nonce), plaintext)
        .map_err(|_| Error::Crypto)
}

/// Decrypt a metadata structure.
///
/// # Errors
/// [`Error::Crypto`] on authentication failure.
pub fn open_metadata(key: &DerivedKey, sealed: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new(key.0.as_ref().into());
    cipher
        .decrypt(GcmNonce::from_slice(nonce), sealed)
        .map_err(|_| Error::Crypto)
}

/// Fill a slot with material indistinguishable from a live record.
///
/// Dummy slots are **not** encryptions of zeros — that would be distinguishable to anyone
/// holding the key, and more importantly it would make "how many dummies are there" a question
/// with an answer. Uniform random bytes of exactly the live-record length are.
#[must_use]
pub fn dummy_record(len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    rand::rngs::OsRng.fill_bytes(&mut v);
    v
}

/// Sealed length for a plaintext of `len` bytes at the given block size.
///
/// Exposed so the store can size slots without performing a trial encryption — and so tests can
/// assert the padding property directly.
#[must_use]
pub fn sealed_len(plaintext_len: usize, block_size: usize) -> usize {
    let blocks = (plaintext_len + LEN_PREFIX).div_ceil(block_size);
    NONCE_LEN + blocks * block_size + MAC_LEN
}

fn pad(plaintext: &[u8], block_size: usize) -> Result<Vec<u8>> {
    let total = plaintext.len() + LEN_PREFIX;
    let padded_len = total.div_ceil(block_size) * block_size;
    let len_u32: u32 = plaintext.len().try_into().map_err(|_| Error::Crypto)?;

    let mut buf = vec![0u8; padded_len];
    buf[..LEN_PREFIX].copy_from_slice(&len_u32.to_be_bytes());
    buf[LEN_PREFIX..LEN_PREFIX + plaintext.len()].copy_from_slice(plaintext);
    // Trailing bytes stay zero. They are encrypted, so they are not observable; using random
    // fill here would buy nothing and cost entropy on every write.
    Ok(buf)
}

fn unpad(padded: &[u8]) -> Result<&[u8]> {
    if padded.len() < LEN_PREFIX {
        return Err(Error::Crypto);
    }
    let len =
        u32::from_be_bytes(padded[..LEN_PREFIX].try_into().map_err(|_| Error::Crypto)?) as usize;
    padded
        .get(LEN_PREFIX..LEN_PREFIX + len)
        .ok_or(Error::Crypto)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = 4096;

    fn key() -> DerivedKey {
        DerivedKey([42u8; 32])
    }

    #[test]
    fn roundtrip_recovers_the_plaintext() {
        let tag = Tag([7u8; 16]);
        let msg = b"the agent learned that ANALYZE fixes the stale-statistics seq scan";
        let sealed = seal_content(&key(), msg, tag, BLOCK).unwrap();
        let mut out = vec![0u8; BLOCK];
        let n = open_content(&key(), &sealed, tag, &mut out).unwrap();
        assert_eq!(&out[..n], msg);
    }

    #[test]
    fn ciphertext_length_reveals_only_block_count() {
        // The padding property the leakage profile's "entry sizes" row depends on: two
        // plaintexts within the same 4 KiB block must produce identical-length output.
        let tag = Tag([1u8; 16]);
        let a = seal_content(&key(), &[0u8; 10], tag, BLOCK).unwrap();
        let b = seal_content(&key(), &[0u8; 3000], tag, BLOCK).unwrap();
        assert_eq!(a.len(), b.len(), "same block => same sealed length");

        let c = seal_content(&key(), &[0u8; 5000], tag, BLOCK).unwrap();
        assert!(
            c.len() > a.len(),
            "crossing a block boundary is observable, by design"
        );
        assert_eq!(c.len(), sealed_len(5000, BLOCK));
        assert_eq!(a.len(), sealed_len(10, BLOCK));
    }

    #[test]
    fn wrong_tag_fails_authentication() {
        // The AAD binding: a ciphertext moved to a different slot must not decrypt, or an
        // adversary could relocate records to confuse the index.
        let sealed = seal_content(&key(), b"secret", Tag([1u8; 16]), BLOCK).unwrap();
        let mut out = vec![0u8; BLOCK];
        assert!(open_content(&key(), &sealed, Tag([2u8; 16]), &mut out).is_err());
    }

    #[test]
    fn wrong_key_fails_authentication() {
        let sealed = seal_content(&key(), b"secret", Tag([1u8; 16]), BLOCK).unwrap();
        let mut out = vec![0u8; BLOCK];
        assert!(open_content(&DerivedKey([9u8; 32]), &sealed, Tag([1u8; 16]), &mut out).is_err());
    }

    #[test]
    fn nonces_differ_across_seals() {
        // Random 192-bit nonces, so no counter state is persisted. Reusing a nonce under the
        // same key would be catastrophic; the birthday bound at 24 bytes makes it a non-issue.
        let tag = Tag([1u8; 16]);
        let a = seal_content(&key(), b"x", tag, BLOCK).unwrap();
        let b = seal_content(&key(), b"x", tag, BLOCK).unwrap();
        assert_ne!(a[..NONCE_LEN], b[..NONCE_LEN]);
        assert_ne!(
            a, b,
            "identical plaintexts must not produce identical ciphertexts"
        );
    }

    #[test]
    fn metadata_roundtrip() {
        let nonce = [3u8; 12];
        let sealed = seal_metadata(&key(), b"manifest", &nonce).unwrap();
        assert_eq!(open_metadata(&key(), &sealed, &nonce).unwrap(), b"manifest");
    }

    #[test]
    fn dummy_records_are_the_right_length_and_not_constant() {
        let a = dummy_record(64);
        let b = dummy_record(64);
        assert_eq!(a.len(), 64);
        assert_ne!(a, b, "dummies must not be a fixed pattern");
    }
}
