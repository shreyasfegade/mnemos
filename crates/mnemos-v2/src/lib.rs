//! # mnemos-v2
//!
//! The second Mnemos design. One sealed partition file replaces v1's separate record store and
//! index, which removes three things at once: the result-dependent record read, the dummy
//! records that consumed capacity, and the Robin Hood write budget.
//!
//! ## On-disk layout
//!
//! ```text
//! partitions.bin   [ 64-byte header ][ region 0 ][ region 1 ] ... [ region B-1 ]
//!                  region = C slots; slot = sealed header ‖ sealed payload
//!                  header plaintext  = id (16) ‖ int8 codes (dim) ‖ f32 scale (4)
//!                  payload plaintext = u32 length ‖ text ‖ 0 ‖ meta ‖ zero padding
//! manifest.enc     sealed, fixed size: epoch, cursor, occupancy bitmap, tombstones
//! table.enc        sealed, written once: centroids, probe groups
//! wrapped.key      master key wrapped by the key store
//! ```
//!
//! Every ciphertext is XChaCha20-Poly1305 under a fresh random 192-bit nonce. A free slot is
//! uniformly random bytes of the slot length.
//!
//! ## The four rules the security argument rests on
//!
//! 1. **Every write rewrites whole regions.** Inserting re-encrypts every live slot of the
//!    target region and refreshes every free slot, so a write reveals only which region.
//! 2. **Every flush rewrites exactly `w + 1` regions.** One per staged record, uniformly random
//!    regions for the rest, and one maintenance region chosen by a round-robin cursor.
//! 3. **A search reads whole regions and nothing else.** Headers are ranked in memory and the
//!    top payloads are decrypted from buffers already read.
//! 4. **A deletion writes nothing when it happens.** It records a tombstone in memory, which
//!    reaches disk inside the next unconditional manifest rewrite; the slot is freed the next
//!    time its region is rewritten, at the latest when the cursor reaches it.
//!
//! Logical partitions map to physical regions through a keyed permutation, so a public
//! partition table does not tell an observer which region holds which topic.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod engine;
pub mod geometry;
pub mod manifest;
pub mod regions;
pub mod seal;
pub mod table;

pub use engine::{EngineV2, V2Stats};
pub use geometry::Geometry;

use mnemos_core::error::{Error, Result};

/// Evaluation-only switches that remove one mechanism each.
///
/// The shipped engine is `Ablation::default()`. Each switch reproduces a specific failure so
/// experiment E4 can measure what the mechanism is worth instead of asserting it. None of them
/// is reachable from `params.toml`.
// Each switch removes one mechanism and is toggled independently by the experiments; any
// combination `validate` accepts is a configuration to measure, not a state to transition.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ablation {
    /// Condition S1 removed: each region is its own file holding only live slots, so file sizes
    /// reveal per-partition occupancy.
    pub growing_regions: bool,
    /// Condition S2 removed: headers are written in the clear and free slots are zero bytes.
    pub unsealed_slots: bool,
    /// Full-region rewrites removed: an insert writes only its slot, a purge only the freed one.
    pub slot_writes_only: bool,
    /// Deterministic intra-region placement: a record takes the lowest-indexed free slot of its
    /// region instead of a uniformly random one.
    ///
    /// This is the negative control for `slot_writes_only`. Under a per-slot write the adversary
    /// sees the index of the slot written, and whether that index says anything about the
    /// region's pre-compromise occupancy depends entirely on how the slot was chosen. With a
    /// random free slot it says nothing while the history is admissible; with the first free
    /// slot the index *is* the occupancy. E5 measures both.
    pub first_free_slot: bool,
    /// Padding removed: a flush rewrites only the regions of its real records.
    pub no_padding: bool,
    /// Admissibility removed: a full partition spills into its two next-nearest partitions.
    pub spill: bool,
    /// Keyed permutation removed: physical region equals logical partition.
    pub identity_permutation: bool,
    /// Tombstoning removed: a deletion rewrites the record's region immediately.
    pub immediate_delete: bool,
    /// Adds v1's result-dependent read: the top-`k` slots are re-read after ranking.
    pub result_reads: bool,
}

impl Ablation {
    /// Whether this is the shipped configuration.
    #[must_use]
    pub fn is_shipped(&self) -> bool {
        *self == Self::default()
    }

    /// Reject combinations the storage backends do not implement.
    ///
    /// # Errors
    /// [`Error::Params`] for an unsupported combination.
    pub fn validate(&self) -> Result<()> {
        if self.growing_regions && self.slot_writes_only {
            return Err(Error::Params(
                "ablation: growing_regions and slot_writes_only cannot be combined".into(),
            ));
        }
        Ok(())
    }
}
