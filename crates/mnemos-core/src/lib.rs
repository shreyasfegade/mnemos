//! # mnemos-core
//!
//! Shared vocabulary for every other crate: types, traits, parameters, errors, and the
//! event bus that the attack harness and the leakage-profile test all consume.
//!
//! This crate holds **no logic and no secrets**. If something here needs a key, it belongs
//! in `mnemos-crypto`.
//!
//! ## The event bus is not telemetry
//!
//! [`event::LeakEvent`] is the adversary's-eye view, and it is load-bearing:
//!
//! - `python/eval/attacks/` consume it **as the adversary's transcript**, so an attack can
//!   never accidentally see more than the paper's threat model says the adversary sees.
//! - `crates/mnemos-engine/tests/leakage_profile.rs` asserts the emitted event set matches
//!   `docs/leakage-profile.md` row by row. Leak something undocumented and CI fails.
//!
//! 
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod event;
pub mod params;
pub mod traits;
pub mod types;

pub use error::{Error, Result};
pub use params::Params;
pub use types::{BucketId, Embedding, Epoch, MemoryId, QuantVec, SlotAddr, Tag};
