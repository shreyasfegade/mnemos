//! # mnemos-mem
//!
//! **Mnemos-M**: the memory-resident design.
//!
//! Mnemos-P (`crates/mnemos-v2`) searches an encrypted index on disk, because that is what the
//! searchable-encryption literature does. That literature does it because a *remote* server holds
//! the data. Here the server is the user's own disk, and the threat model already assumes the
//! adversary cannot read the unlocked process's memory. So the disk can be taken out of the query
//! path entirely, and once it is, the leakage that Mnemos-P has to state as residual disappears.
//!
//! ## What is different
//!
//! | | Mnemos-P | Mnemos-M |
//! |---|---|---|
//! | Placement | nearest centroid, then a keyed permutation | **uniformly random free slot** |
//! | Search | reads `s·k` whole regions from disk | **runs in the locked arena; no disk read** |
//! | Flush | when `w` records are staged, or on a timer | **every Δ, written or not** |
//! | Capacity | per partition, so a full topic refuses | **global only** |
//!
//! Each row closes a channel. Random placement means a write reveals a region drawn from a
//! distribution that does not depend on content, so the write channel carries nothing and a diff
//! between two file versions carries nothing. An in-memory search issues no disk access, so the
//! search, result and query-equality channels are empty rather than mitigated. A clocked flush
//! means the *existence* and *timing* of a write are independent of whether the agent wrote, which
//! is what closes the derived-write channel that no leaky store can close. A global capacity
//! depends only on the record count, which the security game already equalises, so there is no
//! refusal channel and no partition that fills while the store is a third empty.
//!
//! ## On-disk layout
//!
//! ```text
//! slab.bin       [ 64-byte header ][ region 0 ][ region 1 ] ... [ region R-1 ]
//!                region = C slots; slot = sealed header ‖ sealed payload
//!                header plaintext  = id (16) ‖ int8 codes (dim) ‖ f32 scale (4)
//!                payload plaintext = u32 length ‖ text ‖ 0 ‖ meta ‖ zero padding
//! manifest.enc   sealed, fixed size: epoch, cursor, occupancy bitmap, tombstones
//! table.enc      sealed centroids — used only by the content-placement ablation and to
//!                compute evaluation ground truth; the shipped engine never consults it
//! wrapped.key    master key wrapped by the key store
//! ```
//!
//! The layout is deliberately the same shape as Mnemos-P's, so a comparison between the two is a
//! comparison of designs rather than of geometries.
//!
//! ## The rules the security argument rests on
//!
//! 1. **A record's slot is drawn uniformly from the free slots**, at `put` time, before anything
//!    about its content has influenced anything. Its region is that slot's region.
//! 2. **Every flush rewrites exactly `w + 1` whole regions**, and every one of those regions is
//!    uniformly distributed: the `w` record and padding rewrites are draws from the same
//!    distribution, and the last is the maintenance cursor.
//! 3. **A flush happens every Δ while unlocked**, whether or not anything was staged.
//! 4. **A search touches no disk.** Headers, and by default payloads, are decrypted once at unlock
//!    into the locked arena and ranked there.
//! 5. **A deletion writes nothing when it happens**, and its slot is freed when the cursor reaches
//!    its region, which is within `B·Δ` milliseconds rather than within `B` unpredictable flushes.
//!
//! Together these make the leakage function empty beyond the public parameters and the clock.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod engine;
pub mod geometry;
pub mod slabs;

pub use engine::{EngineM, MStats};
pub use geometry::MGeometry;

/// Evaluation-only switches, each removing one of the design's guarantees.
///
/// The shipped engine is `Ablation::default()`. Each switch turns Mnemos-M into something closer
/// to Mnemos-P along one axis, so an experiment can attribute a channel to the mechanism that
/// closes it instead of asserting the attribution.
// Each switch removes one of Mnemos-M's rules and is toggled independently by the audit and the
// experiments; a combination is a configuration to measure, not a state to transition.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ablation {
    /// Rule 1 removed: a record is placed in its nearest centroid's region, as Mnemos-P does, so
    /// the write channel returns.
    pub content_placement: bool,
    /// Rule 3 removed: a flush fires when `w` records are staged rather than on the clock, so
    /// write timing and the derived-write channel return.
    pub on_demand_flush: bool,
    /// Rule 4 removed: a search reads its candidate regions from disk, so the read-side channels
    /// return.
    pub disk_search: bool,
    /// Rule 2 removed: a flush rewrites only the regions its real records need, so the number of
    /// writes reveals how many records were real.
    pub no_padding: bool,
}

impl Ablation {
    /// Whether this is the shipped configuration.
    #[must_use]
    pub fn is_shipped(&self) -> bool {
        *self == Self::default()
    }
}
