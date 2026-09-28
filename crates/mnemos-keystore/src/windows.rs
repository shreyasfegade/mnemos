//! Windows TPM 2.0 via the CNG Platform Crypto Provider. **The primary path** — it matches the
//! development machine, so it is exercised daily rather than being an untested branch.
//!
//! ## What the TPM actually does here
//!
//! It holds a persisted, **non-exportable** RSA-2048 key pair called the *root key*. Mnemos
//! never sees the private half: to wrap the master key it calls `NCryptEncrypt`, and to unwrap
//! it calls `NCryptDecrypt`, and in the second case the arithmetic happens inside the TPM. The
//! root key cannot be copied off the machine even by an attacker with administrator rights,
//! which is the whole point — an imaged disk yields a wrapped master key that is useless
//! anywhere else.
//!
//! That is the difference between this provider and `software`: the software provider stores
//! the root key in a file, so imaging the disk yields both halves.
//!
//! ## Why RSA-OAEP rather than a symmetric wrap
//!
//! The Platform Crypto Provider exposes RSA and ECDSA; it does not expose a persisted symmetric
//! key usable for AES-KW. RSA-OAEP with SHA-256 over a 32-byte payload is the shape the API
//! actually supports, and the cost — a few milliseconds — is paid once per unlock and never on
//! the hot path.
//!
//! ## Unsafe
//!
//! `NCrypt` is a C API, so every call in this file is `unsafe`. The crate is otherwise
//! `#![forbid(unsafe_code)]`; the exception is scoped by the `ncrypt-unsafe` feature exactly as
//! `mnemos-crypto` scopes the exception it needs for `VirtualLock`. Every block carries a
//! `SAFETY` note.

use std::ffi::c_void;

use mnemos_core::error::{Error, Result};
use mnemos_core::traits::KeyStore;
use windows_sys::Win32::Security::Cryptography::{
    NCryptCreatePersistedKey, NCryptDecrypt, NCryptEncrypt, NCryptFinalizeKey, NCryptFreeObject,
    NCryptOpenKey, NCryptOpenStorageProvider, NCryptSetProperty, BCRYPT_OAEP_PADDING_INFO,
    BCRYPT_RSA_ALGORITHM, BCRYPT_SHA256_ALGORITHM, MS_PLATFORM_CRYPTO_PROVIDER, NCRYPT_KEY_HANDLE,
    NCRYPT_LENGTH_PROPERTY, NCRYPT_PAD_OAEP_FLAG, NCRYPT_PROV_HANDLE,
};

/// Stable name for the persisted root key. Changing it orphans every existing store, because
/// the wrapped master keys on disk can only be unwrapped by the key that wrapped them.
const KEY_NAME: &[u16] = &[
    b'm' as u16,
    b'n' as u16,
    b'e' as u16,
    b'm' as u16,
    b'o' as u16,
    b's' as u16,
    b'-' as u16,
    b'r' as u16,
    b'o' as u16,
    b'o' as u16,
    b't' as u16,
    b'-' as u16,
    b'v' as u16,
    b'1' as u16,
    0,
];

/// `NTE_BAD_KEYSET` — the key does not exist yet, so create it.
///
/// Written as a `u32 as i32` cast rather than a hand-computed negative literal. The first
/// version of this constant was hand-computed and wrong, which sent every first run down the
/// "provider refused" branch instead of the "create the key" branch — and because the caller
/// falls back to the software keystore on any error, that failure was completely silent: the
/// TPM was present and working, and Mnemos quietly never used it.
const NTE_BAD_KEYSET: i32 = 0x8009_0016u32 as i32;

/// RSA modulus size. 2048 is the floor every TPM 2.0 supports; 4096 is not universal.
const KEY_BITS: u32 = 2048;

/// TPM-backed key custody through `NCrypt` with the Microsoft Platform Crypto Provider.
pub struct WindowsCngPcp {
    /// `NCRYPT_KEY_HANDLE` for the persisted, non-exportable root key.
    key_handle: NCRYPT_KEY_HANDLE,
    /// Provider handle, freed alongside the key.
    provider: NCRYPT_PROV_HANDLE,
}

// SAFETY: NCrypt handles are process-wide opaque integers; the CNG API is documented as
// thread-safe for concurrent use of a key handle. Nothing in this struct has thread affinity.
unsafe impl Send for WindowsCngPcp {}
unsafe impl Sync for WindowsCngPcp {}

impl WindowsCngPcp {
    /// Open or create the Mnemos root key in the TPM.
    ///
    /// Key properties, all non-negotiable:
    /// - `NCRYPT_ALLOW_EXPORT_FLAG` **not** set — the key cannot leave the TPM.
    /// - Persisted under a stable name so a store survives restarts.
    /// - `NCRYPT_MACHINE_KEY_FLAG` **not** set — per-user, not per-machine, so another local
    ///   account cannot ask the TPM to unwrap on our behalf.
    ///
    /// # Errors
    /// [`Error::KeyCustody`] when no TPM is present or the provider refuses. Common on VMs and
    /// in CI; the caller falls back per `params.keystore.providers` if policy allows. Returning
    /// an error rather than panicking is load-bearing: [`crate::select`] walks the provider list
    /// and falls through on `Err`, so a provider that panicked would take down every caller
    /// before a later provider could be tried.
    pub fn open() -> Result<Self> {
        let mut provider: NCRYPT_PROV_HANDLE = 0;

        // SAFETY: `provider` is a valid out-pointer; the provider name is a static
        // NUL-terminated UTF-16 string supplied by windows-sys.
        let status =
            unsafe { NCryptOpenStorageProvider(&raw mut provider, MS_PLATFORM_CRYPTO_PROVIDER, 0) };
        if status != 0 {
            return Err(Error::KeyCustody(format!(
                "the CNG Platform Crypto Provider is unavailable (0x{status:08x}); this machine \
                 has no usable TPM 2.0"
            )));
        }

        let mut key: NCRYPT_KEY_HANDLE = 0;
        // SAFETY: valid provider handle, NUL-terminated key name, valid out-pointer.
        let status = unsafe { NCryptOpenKey(provider, &raw mut key, KEY_NAME.as_ptr(), 0, 0) };

        let key = if status == 0 {
            key
        } else if status == NTE_BAD_KEYSET {
            Self::create_key(provider)?
        } else {
            // SAFETY: `provider` was successfully opened above and is not used again.
            unsafe { NCryptFreeObject(provider) };
            return Err(Error::KeyCustody(format!(
                "opening the persisted root key failed (0x{status:08x})"
            )));
        };

        Ok(Self {
            key_handle: key,
            provider,
        })
    }

    /// Create the persisted root key. Called once, on first run.
    fn create_key(provider: NCRYPT_PROV_HANDLE) -> Result<NCRYPT_KEY_HANDLE> {
        let mut key: NCRYPT_KEY_HANDLE = 0;

        // SAFETY: valid provider, static algorithm and key-name strings, valid out-pointer.
        // `dwFlags = 0` means: not exportable, not a machine key. Both matter -- see the doc
        // comment on `open`.
        let status = unsafe {
            NCryptCreatePersistedKey(
                provider,
                &raw mut key,
                BCRYPT_RSA_ALGORITHM,
                KEY_NAME.as_ptr(),
                0,
                0,
            )
        };
        if status != 0 {
            return Err(Error::KeyCustody(format!(
                "the TPM refused to create a persisted key (0x{status:08x})"
            )));
        }

        let bits = KEY_BITS;
        // SAFETY: `bits` outlives the call; the length is exactly 4 bytes as the property wants.
        let status = unsafe {
            NCryptSetProperty(
                key,
                NCRYPT_LENGTH_PROPERTY,
                (&raw const bits).cast::<u8>(),
                4,
                0,
            )
        };
        if status != 0 {
            // SAFETY: `key` is a live handle from NCryptCreatePersistedKey.
            unsafe { NCryptFreeObject(key) };
            return Err(Error::KeyCustody(format!(
                "the TPM rejected a {KEY_BITS}-bit RSA key (0x{status:08x})"
            )));
        }

        // SAFETY: `key` is live and fully configured.
        let status = unsafe { NCryptFinalizeKey(key, 0) };
        if status != 0 {
            // SAFETY: `key` is a live handle.
            unsafe { NCryptFreeObject(key) };
            return Err(Error::KeyCustody(format!(
                "finalising the root key failed (0x{status:08x})"
            )));
        }
        Ok(key)
    }

    /// OAEP-SHA-256 padding descriptor, with no label.
    fn padding() -> BCRYPT_OAEP_PADDING_INFO {
        BCRYPT_OAEP_PADDING_INFO {
            pszAlgId: BCRYPT_SHA256_ALGORITHM,
            pbLabel: std::ptr::null_mut(),
            cbLabel: 0,
        }
    }
}

impl Drop for WindowsCngPcp {
    fn drop(&mut self) {
        // SAFETY: both handles were produced by NCrypt and are freed exactly once. The root key
        // itself is untouched -- freeing a handle does not delete a persisted key, which is what
        // makes a store survive a restart.
        unsafe {
            NCryptFreeObject(self.key_handle);
            NCryptFreeObject(self.provider);
        }
    }
}

impl KeyStore for WindowsCngPcp {
    fn provider_name(&self) -> &'static str {
        "windows-cng-pcp"
    }

    fn is_hardware(&self) -> bool {
        true
    }

    fn unwrap_master_key(&self, wrapped: &[u8]) -> Result<Vec<u8>> {
        // NCryptDecrypt. The TPM performs the operation; the root key never enters our
        // address space. Tens of milliseconds, which is fine -- this happens once per unlock,
        // never on the hot path.
        let pad = Self::padding();
        let mut needed: u32 = 0;

        // SAFETY: two-call idiom. The first call passes a null output buffer to learn the
        // required size; `wrapped` is a valid readable slice for `cbInput` bytes.
        let status = unsafe {
            NCryptDecrypt(
                self.key_handle,
                wrapped.as_ptr(),
                u32::try_from(wrapped.len()).unwrap_or(u32::MAX),
                (&raw const pad).cast::<c_void>(),
                std::ptr::null_mut(),
                0,
                &raw mut needed,
                NCRYPT_PAD_OAEP_FLAG,
            )
        };
        if status != 0 {
            // Deliberately opaque: distinguishing "wrong key" from "corrupt ciphertext" is a
            // decryption oracle, and the same reasoning applies here as in `mnemos-crypto`.
            return Err(Error::KeyCustody(
                "the TPM refused to unwrap the master key".into(),
            ));
        }

        let mut out = vec![0u8; needed as usize];
        // SAFETY: `out` is now exactly `needed` bytes, which is what the first call asked for.
        let status = unsafe {
            NCryptDecrypt(
                self.key_handle,
                wrapped.as_ptr(),
                u32::try_from(wrapped.len()).unwrap_or(u32::MAX),
                (&raw const pad).cast::<c_void>(),
                out.as_mut_ptr(),
                needed,
                &raw mut needed,
                NCRYPT_PAD_OAEP_FLAG,
            )
        };
        if status != 0 {
            return Err(Error::KeyCustody(
                "the TPM refused to unwrap the master key".into(),
            ));
        }
        out.truncate(needed as usize);
        Ok(out)
    }

    fn wrap_master_key(&self, master: &[u8]) -> Result<Vec<u8>> {
        let pad = Self::padding();
        let mut needed: u32 = 0;

        // SAFETY: two-call idiom, as above.
        let status = unsafe {
            NCryptEncrypt(
                self.key_handle,
                master.as_ptr(),
                u32::try_from(master.len()).unwrap_or(u32::MAX),
                (&raw const pad).cast::<c_void>(),
                std::ptr::null_mut(),
                0,
                &raw mut needed,
                NCRYPT_PAD_OAEP_FLAG,
            )
        };
        if status != 0 {
            return Err(Error::KeyCustody(format!(
                "the TPM refused to wrap the master key (0x{status:08x})"
            )));
        }

        let mut out = vec![0u8; needed as usize];
        // SAFETY: `out` is exactly the size the first call reported.
        let status = unsafe {
            NCryptEncrypt(
                self.key_handle,
                master.as_ptr(),
                u32::try_from(master.len()).unwrap_or(u32::MAX),
                (&raw const pad).cast::<c_void>(),
                out.as_mut_ptr(),
                needed,
                &raw mut needed,
                NCRYPT_PAD_OAEP_FLAG,
            )
        };
        if status != 0 {
            return Err(Error::KeyCustody(format!(
                "the TPM refused to wrap the master key (0x{status:08x})"
            )));
        }
        out.truncate(needed as usize);
        Ok(out)
    }

    fn attest(&self) -> Result<Option<Vec<u8>>> {
        // Future work per ADR-0005: seal the master key to a PCR set so a modified binary
        // cannot unwrap it. The seam exists; v1 does not use it.
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The round trip, on real hardware.
    ///
    /// Skips rather than fails when no TPM is present: a laptop without one is a legitimate
    /// environment and this is the branch that reports that correctly. The skip is printed so
    /// it cannot be mistaken for a pass.
    #[test]
    fn wrap_unwrap_roundtrips_through_the_tpm() {
        let Ok(ks) = WindowsCngPcp::open() else {
            eprintln!("SKIP: no usable TPM 2.0 on this machine (this is the fallback path)");
            return;
        };

        let master = [0x5Au8; 32];
        let wrapped = ks.wrap_master_key(&master).expect("wrap");
        assert_ne!(
            wrapped.as_slice(),
            master.as_slice(),
            "wrap returned the plaintext"
        );
        assert!(
            wrapped.len() >= 256,
            "an RSA-2048 wrap should be 256 bytes, got {}",
            wrapped.len()
        );

        let back = ks.unwrap_master_key(&wrapped).expect("unwrap");
        assert_eq!(back.as_slice(), master.as_slice());
        assert!(ks.is_hardware());
    }

    /// Two wraps of the same key must differ -- OAEP is randomised, and a deterministic wrap
    /// would let an adversary tell whether two stores share a master key.
    #[test]
    fn wrapping_is_randomised() {
        let Ok(ks) = WindowsCngPcp::open() else {
            eprintln!("SKIP: no usable TPM 2.0 on this machine");
            return;
        };
        let master = [0x11u8; 32];
        assert_ne!(
            ks.wrap_master_key(&master).expect("wrap"),
            ks.wrap_master_key(&master).expect("wrap"),
            "OAEP padding is not randomised"
        );
    }

    /// Garbage must be refused rather than returning junk that the caller then treats as a key.
    #[test]
    fn a_corrupt_wrap_is_rejected() {
        let Ok(ks) = WindowsCngPcp::open() else {
            eprintln!("SKIP: no usable TPM 2.0 on this machine");
            return;
        };
        let master = [0x22u8; 32];
        let mut wrapped = ks.wrap_master_key(&master).expect("wrap");
        wrapped[0] ^= 0xFF;
        assert!(
            ks.unwrap_master_key(&wrapped).is_err(),
            "a tampered wrap was accepted"
        );
    }
}
