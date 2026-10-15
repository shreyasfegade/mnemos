//! # mnemos-keystore
//!
//! Custody of the root key. Small, load-bearing, and separately audited. See the paper, Section 6.4.
//!
//! The root key **never crosses this crate's boundary**. Callers get the *master* key, which
//! the hardware unwraps; the root key stays inside the TPM or enclave for its whole life.
//!
//! | Provider | Platform | Hardware |
//! |---|---|---|
//! | [`windows::WindowsCngPcp`] | Windows 11 (primary — matches the dev machine) | TPM 2.0 via CNG Platform Crypto Provider |
//! | `linux::LinuxTpm2` | Linux | `tpm2-tss` |
//! | `macos::MacosEnclave` | macOS | Secure Enclave via `Security.framework` |
//! | [`software::SoftwareKeystore`] | CI only | **none** |
//!
//! ## The software fallback is double-gated, deliberately
//!
//! CI has no TPM, so a fallback must exist. The danger is that it quietly becomes the default
//! and the entire security story turns into theatre — a system that appears to work while TA-1
//! is false. Two independent gates prevent that:
//!
//! 1. **Compile-time:** `--features insecure-software-keystore`.
//! 2. **Runtime:** `MNEMOS_I_ACCEPT_INSECURE=1`.
//!
//! The runtime error names this design, so anyone who trips it learns why rather than reaching
//! for the flag.

// `NCrypt` is a C API, so the Windows TPM provider needs FFI. The exception is scoped by a
// feature rather than removed globally, exactly as `mnemos-crypto` scopes the exception it needs
// for VirtualLock: every other module in this crate remains unsafe-free and is verified so by
// the compiler.
#![cfg_attr(not(feature = "ncrypt-unsafe"), forbid(unsafe_code))]
#![warn(missing_docs)]

use mnemos_core::error::{Error, Result};
use mnemos_core::traits::KeyStore;
use mnemos_core::Params;

#[cfg(feature = "insecure-software-keystore")]
pub mod software;
#[cfg(windows)]
pub mod windows;

/// Select the first available provider in `params.keystore.providers` order.
///
/// # Errors
/// [`Error::KeyCustody`] if no provider is available and `require_hardware` is set — which is
/// the correct outcome, not a problem to work around. A store that silently downgrades its key
/// custody is worse than one that refuses to open.
pub fn select(params: &Params) -> Result<Box<dyn KeyStore>> {
    for name in &params.keystore.providers {
        match name.as_str() {
            #[cfg(windows)]
            "windows-cng-pcp" => {
                if let Ok(ks) = windows::WindowsCngPcp::open() {
                    return Ok(Box::new(ks));
                }
            }
            #[cfg(feature = "insecure-software-keystore")]
            "software" => {
                if params.keystore.require_hardware {
                    continue;
                }
                if std::env::var("MNEMOS_I_ACCEPT_INSECURE").as_deref() != Ok("1") {
                    return Err(Error::KeyCustody(
                        "software keystore requires MNEMOS_I_ACCEPT_INSECURE=1; it provides \
                         NO protection against the threat model's adversary and exists only \
                         for CI"
                            .into(),
                    ));
                }
                return Ok(Box::new(software::SoftwareKeystore::open()?));
            }
            _ => {}
        }
    }
    Err(Error::KeyCustody(format!(
        "no usable provider among {:?} (require_hardware = {})",
        params.keystore.providers, params.keystore.require_hardware
    )))
}
