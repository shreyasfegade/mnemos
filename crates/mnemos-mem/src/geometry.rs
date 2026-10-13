//! Sizes, derived once from `params.toml`.
//!
//! Every length on disk is a function of these numbers alone, which is condition S1. The slot and
//! region shapes are deliberately identical to Mnemos-P's, so the two designs can be measured
//! against each other at one configuration.

use mnemos_core::error::{Error, Result};
use mnemos_core::Params;
use mnemos_v2::geometry::{ID_LEN, LEN_PREFIX, MAC_LEN, NONCE_LEN};

/// Length of the plain header at the start of `slab.bin`.
pub const FILE_HEADER_LEN: u64 = 64;

/// Bytes per resident partition id under `m.search = "ivf"`.
pub const PART_ID_LEN: usize = 2;

/// Store geometry.
// Each flag is an independent configuration switch read from `params.toml`, as in `Ablation`.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MGeometry {
    /// Regions, `R`. The rewrite unit; not a partition, because nothing is partitioned.
    pub regions: u32,
    /// Slots per region, `C`.
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
    /// Whether payloads are decrypted at unlock and held resident.
    pub payload_resident: bool,
    /// Δ, the clocked flush interval in milliseconds.
    pub flush_ms: u64,
    /// Tombstone capacity of the manifest.
    pub tombstone_capacity: usize,
    /// `m.search = "ivf"`: partition ids are kept resident and a search probes `probe` of them.
    pub index_ivf: bool,
    /// Partitions probed per search when `index_ivf`.
    pub probe: u32,
    /// `flush_mode = "sweep"`: a flush rewrites `sweep_slots` slots after a cursor.
    pub sweep: bool,
    /// Slots rewritten per flush under the sweep.
    pub sweep_slots: u32,
    /// Staged records the sealed journal holds under the sweep.
    pub journal_slots: u32,
    /// `m.backlog = "refuse"`: a `put` that would stage more than `journal_slots` records is
    /// refused to the caller, so a lock is always one flush (C1 without exception).
    pub backlog_refuse: bool,
    /// Whether headers are decrypted at unlock and held resident. `false` is Mnemos-M/disk.
    pub header_resident: bool,
}

impl MGeometry {
    /// Derive and validate.
    ///
    /// # Errors
    /// [`Error::Params`] on an inconsistent configuration.
    pub fn from_params(p: &Params) -> Result<Self> {
        let m = &p.m;
        let bad = |s: &str| Err(Error::Params(format!("m: {s}")));
        if p.index.buckets == 0 || p.index.bucket_capacity == 0 {
            return bad("buckets and bucket_capacity must be non-zero");
        }
        if m.payload_blocks == 0 {
            return bad("payload_blocks must be at least 1");
        }
        if m.tombstone_capacity == 0 {
            return bad("tombstone_capacity must be at least 1");
        }
        if p.index.top_k == 0 {
            return bad("top_k must be non-zero");
        }
        if p.epoch.epoch_writes == 0 {
            return bad("epoch_writes must be at least 1");
        }
        if m.flush_ms == 0 {
            return bad("flush_ms must be at least 1");
        }
        let payload_resident = match m.payload_residency.as_str() {
            "memory" => true,
            "disk" => false,
            other => return bad(&format!("unknown payload_residency {other:?}")),
        };
        let block = p.crypto.block_size_bytes as usize;
        if !block.is_power_of_two() {
            return bad("block_size_bytes must be a power of two");
        }
        let index_ivf = match m.search.as_str() {
            "exact" => false,
            "ivf" => true,
            other => return bad(&format!("unknown search {other:?}")),
        };
        if index_ivf && (m.probe_width == 0 || m.probe_width > p.index.buckets) {
            return bad("probe_width must be in 1..=index.buckets");
        }
        if index_ivf && p.index.buckets > u32::from(u16::MAX) {
            return bad("ivf search stores partition ids as u16");
        }
        let sweep = match m.flush_mode.as_str() {
            "regions" => false,
            "sweep" => true,
            other => return bad(&format!("unknown flush_mode {other:?}")),
        };
        if sweep && (m.sweep_slots == 0 || m.journal_slots == 0) {
            return bad("sweep_slots and journal_slots must be at least 1");
        }
        let slots = u64::from(p.index.buckets) * u64::from(p.index.bucket_capacity);
        if sweep && u64::from(m.sweep_slots) > slots {
            return bad("sweep_slots cannot exceed the number of slots");
        }
        let backlog_refuse = match m.backlog.as_str() {
            "drain" => false,
            "refuse" => true,
            other => return bad(&format!("unknown backlog {other:?}")),
        };
        let header_resident = match m.header_residency.as_str() {
            "memory" => true,
            "disk" => false,
            other => return bad(&format!("unknown header_residency {other:?}")),
        };
        if !header_resident && (!index_ivf || payload_resident || !sweep) {
            return bad("header_residency = \"disk\" needs search = \"ivf\", \
                 payload_residency = \"disk\" and flush_mode = \"sweep\"");
        }
        Ok(Self {
            regions: p.index.buckets,
            capacity: p.index.bucket_capacity,
            dim: p.embedding.dim,
            header_pt: ID_LEN + p.embedding.dim + 4,
            payload_pt: m.payload_blocks as usize * block,
            top_k: p.index.top_k as usize,
            epoch_writes: p.epoch.epoch_writes,
            payload_resident,
            flush_ms: m.flush_ms,
            tombstone_capacity: m.tombstone_capacity as usize,
            index_ivf,
            probe: m.probe_width,
            sweep,
            sweep_slots: m.sweep_slots,
            journal_slots: m.journal_slots,
            backlog_refuse,
            header_resident,
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
        self.regions as usize * self.capacity as usize
    }

    /// Byte offset of an absolute slot in `slab.bin`. Regions are contiguous, so slot `g` of the
    /// whole file sits at `FILE_HEADER_LEN + g * slot_len` whatever its region.
    #[must_use]
    pub const fn slot_offset(&self, g: u64) -> u64 {
        FILE_HEADER_LEN + g * self.slot_len() as u64
    }

    /// Bytes of resident headers at the start of the arena: all of them, or none for M/disk.
    #[must_use]
    pub const fn resident_header_bytes(&self) -> usize {
        if self.header_resident {
            self.slots() * self.header_pt
        } else {
            0
        }
    }

    /// Byte offset of a region in `slab.bin`.
    #[must_use]
    pub const fn region_offset(&self, region: u32) -> u64 {
        FILE_HEADER_LEN + region as u64 * self.region_len() as u64
    }

    /// Largest text plus metadata a slot can hold.
    #[must_use]
    pub const fn max_record_bytes(&self) -> usize {
        self.payload_pt - LEN_PREFIX - 1
    }

    /// Bytes the resident index occupies in the locked arena.
    ///
    /// This is the quantity that bounds where Mnemos-M is usable, so it is computed rather than
    /// estimated, and reported in the statistics an experiment reads.
    #[must_use]
    pub const fn resident_bytes(&self) -> usize {
        let h = if self.header_resident {
            self.header_pt
        } else {
            0
        };
        let per = if self.payload_resident {
            h + self.payload_pt
        } else {
            h
        };
        let ids = if self.index_ivf { PART_ID_LEN } else { 0 };
        self.slots() * (per + ids)
    }

    /// Where the resident partition ids start, measured from the end of the resident headers.
    ///
    /// They sit after the payloads (or directly after the headers when payloads are not
    /// resident) and inside the locked arena, not on the ordinary heap: a map from slot to
    /// partition groups records by topic, which is exactly what must never reach a page file.
    #[must_use]
    pub const fn part_ids_offset(&self) -> usize {
        if self.payload_resident {
            self.slots() * self.payload_pt
        } else {
            0
        }
    }
}
