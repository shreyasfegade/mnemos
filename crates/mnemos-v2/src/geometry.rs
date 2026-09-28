//! Sizes, derived once from `params.toml`, and the validation that keeps them consistent.
//!
//! Every length on disk is a function of these numbers and nothing else, which is condition S1.

use mnemos_core::error::{Error, Result};
use mnemos_core::Params;

/// XChaCha20-Poly1305 nonce length.
pub const NONCE_LEN: usize = 24;
/// Poly1305 tag length.
pub const MAC_LEN: usize = 16;
/// Record identifier length.
pub const ID_LEN: usize = 16;
/// Length of the plain header at the start of `partitions.bin`.
pub const FILE_HEADER_LEN: u64 = 64;
/// Length prefix inside a payload plaintext.
pub const LEN_PREFIX: usize = 4;

/// Store geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    /// Partitions, `B`.
    pub buckets: u32,
    /// Slots per partition, `C`.
    pub capacity: u32,
    /// Embedding dimension.
    pub dim: usize,
    /// Header plaintext length: `16 + dim + 4`.
    pub header_pt: usize,
    /// Payload plaintext length: `payload_blocks * block_size`.
    pub payload_pt: usize,
    /// Results per search.
    pub top_k: usize,
    /// Region rewrites per flush before the maintenance rewrite, `w`.
    pub epoch_writes: u32,
    /// Genuine probes, `k`.
    pub probe_width: u32,
    /// Probe superset multiplier, `s`.
    pub probe_spread: u32,
    /// Retention probability of a genuine probe, `p`.
    pub probe_retention: f32,
    /// Whether searches read one fixed group instead of a randomised superset.
    pub grouped: bool,
    /// Partitions per group.
    pub group_size: u32,
    /// Tombstone capacity of the manifest.
    pub tombstone_capacity: usize,
}

impl Geometry {
    /// Derive and validate.
    ///
    /// # Errors
    /// [`Error::Params`] on an inconsistent configuration.
    pub fn from_params(p: &Params) -> Result<Self> {
        let v = &p.v2;
        let bad = |m: &str| Err(Error::Params(format!("v2: {m}")));
        if p.index.buckets == 0 || p.index.bucket_capacity == 0 {
            return bad("buckets and bucket_capacity must be non-zero");
        }
        if v.payload_blocks == 0 {
            return bad("payload_blocks must be at least 1");
        }
        if v.tombstone_capacity == 0 {
            return bad("tombstone_capacity must be at least 1");
        }
        if p.index.top_k == 0 || p.index.probe_width == 0 || p.index.probe_spread == 0 {
            return bad("top_k, probe_width and probe_spread must be non-zero");
        }
        if p.epoch.epoch_writes == 0 {
            return bad("epoch_writes must be at least 1");
        }
        if !(p.index.probe_retention > 0.0 && p.index.probe_retention <= 1.0) {
            return bad("probe_retention must be in (0, 1]");
        }
        let grouped = match v.probe_mode.as_str() {
            "spread" => false,
            "grouped" => true,
            other => return bad(&format!("unknown probe_mode {other:?}")),
        };
        if grouped && (v.group_size == 0 || p.index.buckets % v.group_size != 0) {
            return bad("group_size must divide buckets");
        }
        if !grouped && p.index.probe_width.saturating_mul(p.index.probe_spread) > p.index.buckets {
            return bad("probe_width * probe_spread exceeds buckets");
        }
        if !matches!(v.flush_policy.as_str(), "on_demand" | "periodic") {
            return bad(&format!("unknown flush_policy {:?}", v.flush_policy));
        }
        let block = p.crypto.block_size_bytes as usize;
        if !block.is_power_of_two() {
            return bad("block_size_bytes must be a power of two");
        }
        Ok(Self {
            buckets: p.index.buckets,
            capacity: p.index.bucket_capacity,
            dim: p.embedding.dim,
            header_pt: ID_LEN + p.embedding.dim + 4,
            payload_pt: v.payload_blocks as usize * block,
            top_k: p.index.top_k as usize,
            epoch_writes: p.epoch.epoch_writes,
            probe_width: p.index.probe_width,
            probe_spread: p.index.probe_spread,
            probe_retention: p.index.probe_retention,
            grouped,
            group_size: v.group_size,
            tombstone_capacity: v.tombstone_capacity as usize,
        })
    }

    /// Sealed header length.
    #[must_use]
    pub const fn header_ct(&self) -> usize {
        NONCE_LEN + self.header_pt + MAC_LEN
    }

    /// Sealed payload length.
    #[must_use]
    pub const fn payload_ct(&self) -> usize {
        NONCE_LEN + self.payload_pt + MAC_LEN
    }

    /// One slot on disk.
    #[must_use]
    pub const fn slot_len(&self) -> usize {
        self.header_ct() + self.payload_ct()
    }

    /// One region on disk.
    #[must_use]
    pub const fn region_len(&self) -> usize {
        self.capacity as usize * self.slot_len()
    }

    /// Total slots.
    #[must_use]
    pub const fn slots(&self) -> usize {
        self.buckets as usize * self.capacity as usize
    }

    /// Byte offset of a region in `partitions.bin`.
    #[must_use]
    pub const fn region_offset(&self, region: u32) -> u64 {
        FILE_HEADER_LEN + region as u64 * self.region_len() as u64
    }

    /// Regions read by one search: `k * s`, or one group.
    #[must_use]
    pub const fn regions_per_search(&self) -> u32 {
        if self.grouped {
            self.group_size
        } else {
            self.probe_width * self.probe_spread
        }
    }

    /// Largest text plus metadata a slot can hold.
    #[must_use]
    pub const fn max_record_bytes(&self) -> usize {
        self.payload_pt - LEN_PREFIX - 1
    }
}
