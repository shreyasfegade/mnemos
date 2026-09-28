//! # mnemos-baselines
//!
//! The things we attack and the things we compare against — **in-tree, on purpose**.
//!
//! The paper's central attack (A3) is an attack on a *baseline*. If the baseline is not
//! reproducible, Figure 1 is not reproducible, and a reviewer who cannot rerun Figure 1 will
//! not believe it. So Sophos and Diana live here, implemented faithfully enough to exhibit the
//! leak, alongside the plaintext and naive baselines that anchor the performance curves.
//!
//! | Baseline | Role | Expected under A3 |
//! |---|---|---|
//! | [`plain::PlaintextIvf`] | Utility ceiling, latency floor | trivially broken |
//! | [`naive::NaiveEts`] | Encrypt-then-search, decrypt-and-scan | broken by A1/A2 |
//! | [`sophos::SophosFp`] | Forward-private DSSE, counter-based | **`Adv_hist` ≥ 0.50 — gate G2** |
//! | [`diana::DianaFp`] | Forward-private DSSE, GGM-tree | **`Adv_hist` ≥ 0.50** |
//!
//! ## Fairness
//!
//! These implementations must be **faithful, not strawmen**. A weakened Sophos that leaks more
//! than the real thing would invalidate the finding, and it is the first thing a hostile
//! reviewer would check. Where the original is ambiguous about what persists to disk, we
//! implement the *most conservative* reading — the one that leaks least — and say so in the
//! module docs. If A3 still works against that, the result is solid.
//!
//! Compass, Opal and ONYX are **not** reimplemented. Their numbers enter the comparison as
//! literature-reported values, labelled as such (`the evaluation plan` E8). A bad reimplementation of
//! someone else's system is worse than no reimplementation.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod diana;
pub mod keyword;
pub mod naive;
pub mod plain;
pub mod sophos;

#[cfg(test)]
mod testutil;

pub use keyword::Keywords;
pub use sophos::SnapshotView;
