//! # mnemos-crypto
//!
//! Primitives, composed. **No new cryptography is invented here** — that is an explicit
//! non-goal (`the specification` §5.3). Everything is `RustCrypto`, wired together to satisfy the
//! threat model.
//!
//! ## Layout
//!
//! | Module | Role |
//! |---|---|
//! | [`arena`] | The locked plaintext region. **TA-4's mechanism**; see ADR-0001. |
//! | [`kdf`] | HKDF-SHA-256 key hierarchy: `RK → MK → {K_content, K_tag, K_idx, K_bucket, K_e}` |
//! | [`aead`] | XChaCha20-Poly1305 for contents, AES-256-GCM for metadata |
//! | [`prf`] | HMAC-SHA-256, truncated to 128 bits: tags and slot addresses |
//! | [`pprf`] | GGM tree over AES-128, Diana-style: forward-private search tokens |
//!
//! ## Rules that hold everywhere in this crate
//!
//! 1. **All tag and MAC comparisons go through `subtle::ConstantTimeEq`.** A `==` on secret
//!    bytes is a timing oracle on entry existence.
//! 2. **No plaintext-derived addressing.** Slot addresses come from `PRF(K_idx, tag)` where the
//!    tag comes from a random `MemoryId`. Deriving from content would reintroduce
//!    deterministic-encryption leakage through the back door.
//! 3. **Every buffer holding key material or plaintext is `Zeroizing`.**
//! 4. **`unsafe` is confined to [`arena`]**, and each use carries a `SAFETY` block.

#![warn(missing_docs)]
// unsafe is permitted only in `arena`, which needs VirtualLock/mlock.
#![cfg_attr(not(feature = "arena-unsafe"), forbid(unsafe_code))]

pub mod aead;
pub mod arena;
pub mod kdf;
pub mod pprf;
pub mod prf;

pub use arena::Arena;
pub use kdf::KeyHierarchy;
