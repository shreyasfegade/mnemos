//! XChaCha20-Poly1305, in place, so plaintext lives only in the caller's locked scratch buffer.
//!
//! One AEAD for every structure in v2. v1 used AES-256-GCM for metadata beside `XChaCha20` for
//! content; the second primitive bought nothing and its 96-bit random nonces had a birthday
//! bound that full-region re-encryption consumes quickly. A 192-bit random nonce does not.

use chacha20poly1305::aead::generic_array::GenericArray;
use chacha20poly1305::aead::{AeadInPlace, KeyInit};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use rand::RngCore;
use zeroize::Zeroize;

use mnemos_core::error::{Error, Result};

use crate::geometry::{LEN_PREFIX, MAC_LEN, NONCE_LEN};

/// Encrypt `scratch` in place and append `nonce ‖ ciphertext ‖ mac` to `out`.
///
/// On return `scratch` holds ciphertext, not plaintext.
///
/// # Errors
/// [`Error::Crypto`] on failure.
pub fn seal_in_place(
    key: &[u8; 32],
    scratch: &mut [u8],
    aad: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    let cipher = XChaCha20Poly1305::new(key.as_slice().into());
    let mut nonce = [0u8; NONCE_LEN];
    rand::rngs::OsRng.fill_bytes(&mut nonce);
    let tag = cipher
        .encrypt_in_place_detached(XNonce::from_slice(&nonce), aad, scratch)
        .map_err(|_| Error::Crypto)?;
    out.extend_from_slice(&nonce);
    out.extend_from_slice(scratch);
    out.extend_from_slice(tag.as_slice());
    Ok(())
}

/// Decrypt `sealed` into `scratch`, whose length must equal the plaintext length.
///
/// On failure `scratch` is zeroised.
///
/// # Errors
/// [`Error::Crypto`] on a length mismatch or authentication failure.
pub fn open_in_place(key: &[u8; 32], sealed: &[u8], aad: &[u8], scratch: &mut [u8]) -> Result<()> {
    let n = scratch.len();
    if sealed.len() != NONCE_LEN + n + MAC_LEN {
        return Err(Error::Crypto);
    }
    let cipher = XChaCha20Poly1305::new(key.as_slice().into());
    scratch.copy_from_slice(&sealed[NONCE_LEN..NONCE_LEN + n]);
    let tag = GenericArray::from_slice(&sealed[NONCE_LEN + n..]);
    cipher
        .decrypt_in_place_detached(XNonce::from_slice(&sealed[..NONCE_LEN]), aad, scratch, tag)
        .map_err(|_| {
            scratch.zeroize();
            Error::Crypto
        })
}

/// Seal a heap buffer that is not plaintext content (manifest, table). Returns
/// `nonce ‖ ciphertext ‖ mac`.
///
/// # Errors
/// [`Error::Crypto`] on failure.
pub fn seal_vec(key: &[u8; 32], mut plain: Vec<u8>, aad: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(NONCE_LEN + plain.len() + MAC_LEN);
    let r = seal_in_place(key, &mut plain, aad, &mut out);
    plain.zeroize();
    r.map(|()| out)
}

/// Inverse of [`seal_vec`].
///
/// # Errors
/// [`Error::Crypto`] on authentication failure.
pub fn open_vec(key: &[u8; 32], sealed: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
    if sealed.len() < NONCE_LEN + MAC_LEN {
        return Err(Error::Crypto);
    }
    let mut buf = vec![0u8; sealed.len() - NONCE_LEN - MAC_LEN];
    open_in_place(key, sealed, aad, &mut buf)?;
    Ok(buf)
}

/// Append `n` uniformly random bytes.
pub fn random_into(out: &mut Vec<u8>, n: usize) {
    let start = out.len();
    out.resize(start + n, 0);
    rand::rngs::OsRng.fill_bytes(&mut out[start..]);
}

/// Associated data for a header at a physical position. Binding position means a header moved
/// to another slot fails to authenticate.
#[must_use]
pub fn header_aad(region: u32, slot: u32) -> [u8; 19] {
    let mut a = [0u8; 19];
    a[..11].copy_from_slice(b"mnemos/v2/h");
    a[11..15].copy_from_slice(&region.to_le_bytes());
    a[15..].copy_from_slice(&slot.to_le_bytes());
    a
}

/// Associated data for a header sealed at `put` time, before its position exists.
pub const STAGED_AAD: &[u8] = b"mnemos/v2/staged";

/// Text and metadata out of a decrypted payload: a big-endian length, then `text 0x00 meta`.
///
/// Shared by Mnemos-P and Mnemos-M, which use the same payload framing.
///
/// # Errors
/// [`Error::Crypto`] on a frame whose length prefix does not fit the payload.
pub fn decode_frame(ps: &[u8]) -> Result<(String, String)> {
    let n = u32::from_be_bytes(
        ps.get(..LEN_PREFIX)
            .ok_or(Error::Crypto)?
            .try_into()
            .map_err(|_| Error::Crypto)?,
    ) as usize;
    let frame = ps.get(LEN_PREFIX..LEN_PREFIX + n).ok_or(Error::Crypto)?;
    let sep = frame.iter().position(|b| *b == 0).unwrap_or(frame.len());
    let text = String::from_utf8_lossy(&frame[..sep]).into_owned();
    let meta = String::from_utf8_lossy(frame.get(sep + 1..).unwrap_or(&[])).into_owned();
    Ok((text, meta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_tamper() {
        let key = [7u8; 32];
        let mut scratch = b"secret header bytes".to_vec();
        let mut out = Vec::new();
        seal_in_place(&key, &mut scratch, b"aad", &mut out).unwrap();
        assert_ne!(
            &scratch[..],
            b"secret header bytes",
            "scratch must hold ciphertext"
        );
        let mut back = vec![0u8; 19];
        open_in_place(&key, &out, b"aad", &mut back).unwrap();
        assert_eq!(&back, b"secret header bytes");
        assert!(open_in_place(&key, &out, b"other", &mut back).is_err());
        assert!(back.iter().all(|b| *b == 0), "failed open must zeroise");
        let mut bad = out.clone();
        bad[30] ^= 1;
        assert!(open_in_place(&key, &bad, b"aad", &mut back).is_err());
    }

    #[test]
    fn position_binding() {
        assert_ne!(header_aad(1, 2), header_aad(2, 1));
    }
}
