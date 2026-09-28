//! # mnemos-store
//!
//! Persistence: the history-independent slab, the fixed-length write-ahead log, and the
//! manifest.
//!
//! Everything in this crate is written on the assumption that **the adversary reads every byte
//! it produces** — as a live transcript for `t ≥ τ`, and as a final snapshot for `t < τ`. The
//! design consequences show up as invariants that look wasteful until you remember why they
//! are there:
//!
//! | Invariant | Without it |
//! |---|---|
//! | The slab is preallocated to `n_max` | File size tracks corpus size |
//! | Every slot is the same size and shape, live or dummy | Occupancy is countable |
//! | The WAL is always exactly `epoch_writes` records long | Its fill level counts pending writes |
//! | The manifest is rewritten every epoch, changed or not | Its mtime marks real activity |
//! | Compaction touches a fixed slot budget | Its volume tracks deletion rate |
//!
//! Each of these costs something. Each buys a row in `docs/leakage-profile.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod manifest;
pub mod slab;
pub mod wal;

pub use manifest::Manifest;
pub use slab::Slab;
pub use wal::Wal;
