//! # mnemos-index
//!
//! Encrypted IVF over int8-quantised embeddings — forward-private, Type-II backward-private,
//! history-independent.
//!
//! ## The one invariant that must never be relaxed
//!
//! > Every search reads **exactly** `(probe_width + decoy_buckets) × bucket_capacity` index
//! > slots and returns candidates from all of them, regardless of the query and regardless of
//! > how many entries actually match.
//!
//! Every plausible optimisation violates it. Skipping empty buckets, stopping early once
//! `top_k` good candidates are found, reading a partial bucket — each makes I/O volume
//! query-dependent, which is the volumetric leakage Xu et al. (CCS 2023) showed survives
//! forward *and* backward privacy. Attack **A5** is aimed at this crate specifically, and it
//! is the attack we would most deserve to lose to.
//!
//! `Params::slots_read_per_search()` is the single expression of this invariant; use it rather
//! than recomputing the product locally.
//!
//! ## Module map
//!
//! | Module | Role |
//! |---|---|
//! | [`ivf`] | Centroid table, partition assignment, the constant-volume probe |
//! | [`kmeans`] | The shared spherical partitioner — same code path as the plaintext baseline |
//! | [`bucket`] | Fixed-capacity padded partitions; live and dummy slots are indistinguishable |
//! | [`token`] | Forward-private search tokens (wraps `mnemos-crypto::pprf`) |
//! | [`revoke`] | Type-II deletion tokens |

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bucket;
pub mod ivf;
pub mod kmeans;
pub mod revoke;
pub mod token;

pub use bucket::BucketSlot;
pub use ivf::EncryptedIvf;
pub use kmeans::spherical_kmeans;
