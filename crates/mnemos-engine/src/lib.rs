//! # mnemos-engine
//!
//! Orchestration: the public API, the epoch state machine, and the event bus.
//!
//! This is where the pieces meet, so it is where the invariants that span pieces are enforced:
//!
//! 1. **Constant-volume search** — exactly `(k_probe + d) × C` index reads and exactly `top_k`
//!    content reads, every query.
//! 2. **Fixed-size epoch flush** — exactly `epoch_writes` slot writes, dummy-padded, in
//!    PRF-shuffled order.
//! 3. **Fixed compaction budget** — `compaction_slots` per epoch, regardless of reclaimability.
//! 4. **Unconditional manifest rewrite** — every epoch, changed or not.
//! 5. **Plaintext confined to the arena** — decrypt in, zeroise out, before returning.
//!
//! Each is asserted by `tests/leakage_profile.rs` against `docs/leakage-profile.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod engine;
pub mod epoch;

pub use engine::Engine;
