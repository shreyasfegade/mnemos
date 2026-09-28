//! Error taxonomy.
//!
//! One rule: **an error message must never contain plaintext, key material, query text, or a
//! `MemoryId`.** Errors get logged, logs get read, and the threat model says nothing about
//! logs being protected. Where a message would be more helpful with a secret in it, it stays
//! less helpful.

use thiserror::Error;

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that can go wrong.
#[derive(Debug, Error)]
pub enum Error {
    /// `params.toml` is missing, malformed, or violates a cross-field invariant.
    #[error("parameter error: {0}")]
    Params(String),

    /// A cryptographic operation failed. Deliberately opaque: distinguishing "bad key" from
    /// "bad ciphertext" from "bad AAD" is a decryption oracle.
    #[error("cryptographic operation failed")]
    Crypto,

    /// No key-custody provider was usable.
    ///
    /// The common case is a machine without a TPM while `keystore.require_hardware = true`.
    /// The message names ADR-0005 so whoever hits it learns *why* the software fallback is
    /// double-gated instead of just working.
    #[error("key custody unavailable: {0} (see docs/adr/0005-hardware-key-custody.md)")]
    KeyCustody(String),

    /// The engine is locked; the master key is not resident.
    #[error("store is locked — call unlock() first")]
    Locked,

    /// Store capacity exceeded and growth was not permitted.
    #[error("capacity exceeded: {live} live entries against n_max {n_max}")]
    CapacityExceeded {
        /// Live entry count.
        live: u64,
        /// Declared capacity.
        n_max: u64,
    },

    /// On-disk format is unrecognised or from an incompatible version.
    #[error("format error: {0}")]
    Format(String),

    /// Underlying I/O failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// The requested entry does not exist, or has been deleted.
    ///
    /// Callers **must not** branch on this in a way that produces a timing difference against
    /// the found case — that would reintroduce an existence oracle on the API surface.
    #[error("entry not found")]
    NotFound,
}
