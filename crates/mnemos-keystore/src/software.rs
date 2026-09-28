//! **INSECURE.** A file-backed key, for CI and tests only.
//!
//! This module provides **no protection whatsoever** against the threat model's adversary: a
//! full host compromise reads the key file. It exists because CI runners have no TPM, and for
//! no other reason.
//!
//! Two gates, both required (ADR-0005):
//! 1. `--features insecure-software-keystore` at compile time.
//! 2. `MNEMOS_I_ACCEPT_INSECURE=1` at runtime.
//!
//! Any experiment result produced with this provider is **not a security result**.
//! [`KeyStore::provider_name`] returns `"software-INSECURE"` and that string is recorded in
//! every experiment's metadata, so a benchmark run without a TPM can never be mistaken for one
//! with it — including by us, six weeks later, while assembling figures.

use std::path::{Path, PathBuf};

use mnemos_core::error::{Error, Result};
use mnemos_core::traits::KeyStore;
use mnemos_crypto::aead::{open_metadata, seal_metadata};
use mnemos_crypto::kdf::DerivedKey;
use rand::RngCore;
use zeroize::Zeroize;

/// File-backed key. Not secure. Not for any purpose other than CI.
pub struct SoftwareKeystore {
    key: DerivedKey,
    path: PathBuf,
}

impl SoftwareKeystore {
    /// Load or create the key file at the default location.
    ///
    /// # Errors
    /// [`Error::KeyCustody`] if the runtime gate is not set, or on I/O failure.
    pub fn open() -> Result<Self> {
        Self::open_at(Path::new("mnemos-insecure.key"), gate_satisfied())
    }

    /// Load or create the key file at a specific path.
    ///
    /// `insecure_accepted` is the runtime gate, passed explicitly rather than read from the
    /// environment here. Making it a parameter means every caller states *in its own source*
    /// that it is opting into an insecure keystore — a test that passes `true` is visibly
    /// doing so, and cannot do it by accident through ambient process state.
    ///
    /// # Errors
    /// [`Error::KeyCustody`] if the gate is not satisfied, or on I/O failure.
    pub fn open_at(path: &Path, insecure_accepted: bool) -> Result<Self> {
        if !insecure_accepted {
            return Err(Error::KeyCustody(
                "software keystore is gated on MNEMOS_I_ACCEPT_INSECURE=1; it provides NO \
                 protection against the threat model's adversary and exists only for CI"
                    .into(),
            ));
        }

        let key = if path.exists() {
            let bytes = std::fs::read(path)
                .map_err(|e| Error::KeyCustody(format!("reading key file: {e}")))?;
            let arr: [u8; 32] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| Error::KeyCustody("key file is not 32 bytes".into()))?;
            DerivedKey(arr)
        } else {
            eprintln!(
                "WARNING: mnemos is using the SOFTWARE keystore. The root key is on disk in \
                 plaintext. This provides no protection against the threat model's adversary. \
                 Results produced in this configuration are not security results."
            );
            let mut arr = [0u8; 32];
            rand::rngs::OsRng.fill_bytes(&mut arr);
            std::fs::write(path, arr)
                .map_err(|e| Error::KeyCustody(format!("writing key file: {e}")))?;
            let k = DerivedKey(arr);
            arr.zeroize();
            k
        };

        Ok(Self {
            key,
            path: path.to_path_buf(),
        })
    }

    /// Where the key file lives. Surfaced so callers can report it in provenance metadata.
    #[must_use]
    pub fn key_path(&self) -> &Path {
        &self.path
    }
}

impl KeyStore for SoftwareKeystore {
    fn provider_name(&self) -> &'static str {
        // Deliberately shouted. This string ends up in experiment metadata and in the paper's
        // reproducibility appendix.
        "software-INSECURE"
    }

    fn is_hardware(&self) -> bool {
        false
    }

    fn unwrap_master_key(&self, wrapped: &[u8]) -> Result<Vec<u8>> {
        if wrapped.len() < 12 {
            return Err(Error::KeyCustody("wrapped key too short".into()));
        }
        let (nonce_bytes, ct) = wrapped.split_at(12);
        let nonce: [u8; 12] = nonce_bytes.try_into().expect("12 bytes");
        open_metadata(&self.key, ct, &nonce).map_err(|_| Error::KeyCustody("unwrap failed".into()))
    }

    fn wrap_master_key(&self, master: &[u8]) -> Result<Vec<u8>> {
        let mut nonce = [0u8; 12];
        rand::rngs::OsRng.fill_bytes(&mut nonce);
        let ct = seal_metadata(&self.key, master, &nonce)
            .map_err(|_| Error::KeyCustody("wrap failed".into()))?;
        let mut out = Vec::with_capacity(12 + ct.len());
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ct);
        Ok(out)
    }
}

/// Whether the runtime gate is set. Read from the environment in exactly one place.
#[must_use]
pub fn gate_satisfied() -> bool {
    std::env::var("MNEMOS_I_ACCEPT_INSECURE").as_deref() == Ok("1")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests opt into the insecure keystore explicitly, at each call site.
    const ACCEPTED: bool = true;

    #[test]
    fn wrap_unwrap_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let ks = SoftwareKeystore::open_at(&dir.path().join("k"), ACCEPTED).unwrap();

        let master = [42u8; 32];
        let wrapped = ks.wrap_master_key(&master).unwrap();
        assert_ne!(
            wrapped,
            master.to_vec(),
            "wrapped key must not be the plaintext"
        );
        assert_eq!(ks.unwrap_master_key(&wrapped).unwrap(), master.to_vec());
    }

    #[test]
    fn key_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k");

        let wrapped = {
            let ks = SoftwareKeystore::open_at(&path, ACCEPTED).unwrap();
            ks.wrap_master_key(&[7u8; 32]).unwrap()
        };
        let ks2 = SoftwareKeystore::open_at(&path, ACCEPTED).unwrap();
        assert_eq!(ks2.unwrap_master_key(&wrapped).unwrap(), vec![7u8; 32]);
    }

    #[test]
    fn provider_name_is_unmistakable() {
        let dir = tempfile::tempdir().unwrap();
        let ks = SoftwareKeystore::open_at(&dir.path().join("k"), ACCEPTED).unwrap();
        assert!(ks.provider_name().contains("INSECURE"));
        assert!(!ks.is_hardware());
    }

    #[test]
    fn refuses_to_open_without_the_gate() {
        let dir = tempfile::tempdir().unwrap();
        let err = SoftwareKeystore::open_at(&dir.path().join("k"), false);
        assert!(err.is_err(), "the gate must be enforced, not advisory");
    }

    #[test]
    fn tampered_wrapped_key_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let ks = SoftwareKeystore::open_at(&dir.path().join("k"), ACCEPTED).unwrap();
        let mut wrapped = ks.wrap_master_key(&[1u8; 32]).unwrap();
        let n = wrapped.len();
        wrapped[n - 1] ^= 0xFF;
        assert!(ks.unwrap_master_key(&wrapped).is_err());
    }
}
