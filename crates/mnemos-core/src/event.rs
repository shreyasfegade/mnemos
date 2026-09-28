//! The adversary's-eye view.
//!
//! Every operation that produces a disk-visible effect emits exactly one [`LeakEvent`]. This
//! is **not** telemetry and **not** debug logging — it is a first-class part of the security
//! argument, with three consumers:
//!
//! 1. **The attack harness** (`python/eval/attacks/`) treats the event stream *as* the
//!    transcript from `THREAT-MODEL.md` §2.2. Attacks import `mnemos.trace`, never
//!    `mnemos.engine`, so an attack structurally cannot see more than the adversary sees.
//! 2. **`crates/mnemos-engine/tests/leakage_profile.rs`** asserts the emitted variants match
//!    `docs/leakage-profile.md` row for row. **Leak something undocumented and CI fails.**
//! 3. **The demo** renders it live.
//!
//! ## The rule for adding a variant
//!
//! A new variant means the implementation leaks something new. Adding one **requires** a
//! matching row in `docs/leakage-profile.md` in the same commit. That is the whole point.
//!
//! ## What must never appear in an event
//!
//! Plaintext, key material, query text, `MemoryId`, or anything the adversary could not
//! compute from the disk and the clock. If you are tempted, the design is wrong, not the event.

use serde::{Deserialize, Serialize};

use crate::types::{BucketId, Epoch, SlotAddr};

/// One disk-visible effect, as the adversary would observe it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LeakEvent {
    /// Store created. Only public parameters are observable.
    Setup {
        /// Declared capacity; the store is preallocated to this size.
        n_max: u64,
        /// Number of IVF partitions.
        buckets: u32,
        /// Slots per partition.
        bucket_capacity: u32,
        /// Record padding granularity.
        block_size: u32,
    },

    /// An epoch flushed. Exactly `record_count == params.epoch.epoch_writes` slot writes
    /// occurred; the adversary cannot tell which were real.
    EpochFlush {
        /// Epoch index.
        epoch: Epoch,
        /// Always `params.epoch.epoch_writes`. If this ever varies, the padding is broken
        /// and `tests/leakage_profile.rs` will say so.
        record_count: u32,
    },

    /// A content slot was written. Emitted once per slot in a flush batch, in PRF-shuffled
    /// order — the emission order deliberately does **not** match the logical write order.
    SlotWrite {
        /// Physical address.
        addr: SlotAddr,
        /// Epoch this write belongs to.
        epoch: Epoch,
    },

    /// An index slot within a bucket was read during a search.
    BucketRead {
        /// Which partition.
        bucket: BucketId,
        /// Whether this read was a decoy. **Adversary-invisible** — present only so the
        /// evaluation harness can measure how well decoys work. Attacks MUST NOT read it;
        /// `mnemos.trace` strips this field before handing events to `python/eval/attacks/`.
        #[serde(skip)]
        decoy: bool,
    },

    /// An **index** slot was written during a flush — one per record placed into a partition.
    ///
    /// This variant exists because the leakage profile did not model index writes at all, and
    /// the hole was load-bearing: the flush path writes each record into its semantic
    /// partition at a deterministic file offset, so an adversary watching the index file sees
    /// both *how many* records a flush really contained and *which partitions* they went to.
    /// That is the structure-transfer signal in full, and no event described it.
    ///
    /// The invariant is that exactly `params.epoch.epoch_writes` of these occur per flush --
    /// dummies included -- so the count reveals nothing about how many records were real. See
    /// `the findings notes`.
    IndexWrite {
        /// Which partition received the write.
        bucket: BucketId,
        /// Epoch this write belongs to.
        epoch: Epoch,
    },

    /// A content slot was read (a rerank winner). Always exactly `top_k` per search,
    /// dummy-padded when fewer entries matched.
    SlotRead {
        /// Physical address.
        addr: SlotAddr,
    },

    /// Search tokens advanced for a partition — the forward-privacy step.
    TokenAdvance {
        /// Which partition.
        bucket: BucketId,
        /// Epoch after the advance.
        epoch: Epoch,
    },

    /// Compaction touched its fixed, data-independent slot budget.
    Compact {
        /// Epoch.
        epoch: Epoch,
        /// Always `params.epoch.compaction_slots`, regardless of how much was reclaimable.
        slots_touched: u32,
        /// The slots actually rewritten.
        ///
        /// The count alone used to be the whole event, which described less than the disk did:
        /// compaction writes at specific offsets and an adversary watching the slab sees which.
        /// The cursor is data-independent so naming them discloses nothing an adversary could
        /// not predict --- but that is a property of a *modelled* channel, and an unmodelled
        /// one cannot be argued about at all. This is the F5 lesson applied a second time.
        slots: Vec<SlotAddr>,
    },

    /// The manifest was rewritten. Happens every epoch whether or not anything changed, so
    /// its occurrence carries no information.
    ManifestWrite {
        /// Epoch.
        epoch: Epoch,
    },

    /// Capacity was exceeded and the store doubled. **A loud, unhidden event** — declared as
    /// a non-goal in `the specification` §5.3 and as a row in the leakage profile.
    Grow {
        /// Capacity before.
        old_n_max: u64,
        /// Capacity after.
        new_n_max: u64,
    },

    /// A hardware key operation occurred (unlock or lock). The adversary sees the TPM call
    /// and its timing, never the key.
    KeyOp {
        /// `true` for unlock, `false` for lock.
        unlock: bool,
    },

    /// Mnemos v2: a partition region of `partitions.bin` was rewritten.
    ///
    /// `region` is the **physical** region index, which is what a storage-level observer sees.
    /// It is related to the logical partition only through the keyed permutation. `slots` is
    /// empty when the whole region was rewritten, which is the shipped behaviour; it lists
    /// individual slots only under the evaluation ablation that disables full-region rewrites.
    RegionWrite {
        /// Physical region index.
        region: u32,
        /// Flush this write belongs to.
        epoch: Epoch,
        /// Slots written, or empty for the whole region.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        slots: Vec<u32>,
    },

    /// Mnemos v2: a whole partition region was read, by a search or by the unlock scan.
    RegionRead {
        /// Physical region index.
        region: u32,
    },

    /// Mnemos v2, **evaluation ablation only**: one slot re-read after ranking. The shipped
    /// engine never emits this; it exists so experiment E5 can measure the result-dependent
    /// read channel that v1's separate record store has.
    RegionSlotRead {
        /// Physical region index.
        region: u32,
        /// Slot within the region.
        slot: u32,
    },
}

/// Sink for [`LeakEvent`]s.
///
/// The engine holds one. Production uses a no-op; the demo uses a WebSocket broadcaster;
/// the evaluation harness uses a recorder that serialises to JSONL.
pub trait EventSink: Send + Sync {
    /// Record one event. Implementations must not block the write path — buffer or drop.
    fn emit(&self, event: LeakEvent);
}

/// Discards everything. The default in production, where the event bus is pure overhead.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSink;

impl EventSink for NullSink {
    fn emit(&self, _event: LeakEvent) {}
}

/// Buffers events in memory. Used by tests and by the evaluation harness.
#[derive(Debug, Default)]
pub struct RecordingSink {
    events: parking_lot::Mutex<Vec<LeakEvent>>,
}

impl RecordingSink {
    /// Empty recorder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of everything recorded so far.
    #[must_use]
    pub fn drain(&self) -> Vec<LeakEvent> {
        std::mem::take(&mut *self.events.lock())
    }
}

impl EventSink for RecordingSink {
    fn emit(&self, event: LeakEvent) {
        self.events.lock().push(event);
    }
}
