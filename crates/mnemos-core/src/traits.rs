//! The seams.
//!
//! Four traits, each drawn where a *substitution* is genuinely useful:
//!
//! - [`KeyStore`] — because there are four real implementations (three platforms plus the CI
//!   fallback) and swapping them must not touch anything else.
//! - [`BlobStore`] and [`Index`] — because `mnemos-baselines` implements the *same* operations
//!   with deliberately weaker properties, and the attack harness must be able to point A3 at
//!   either without knowing which it has.
//! - [`SearchableMemory`] — the surface the CLI, the FFI, the demo, and every experiment share.
//!
//! Nothing else gets a trait. Abstraction that exists only in anticipation of a second
//! implementation is cost without benefit.

use crate::error::Result;
use crate::types::{BucketId, Epoch, Hit, Memory, MemoryId, QuantVec, SlotAddr, Tag};

/// Custody of the root key.
///
/// Deliberately tiny — four methods — so that three platform implementations stay cheap to
/// keep correct. See ADR-0005.
///
/// The root key **never crosses this boundary**. `unwrap_master_key` returns the *master* key,
/// which the hardware decrypts; the root key itself stays inside the TPM or enclave.
pub trait KeyStore: Send + Sync {
    /// Human-readable provider name, e.g. `windows-cng-pcp`. Appears in experiment metadata so
    /// a benchmark run always records which custody path produced it.
    fn provider_name(&self) -> &'static str;

    /// Whether this provider is hardware-backed. `false` only for the CI fallback, which is
    /// double-gated (compile feature **and** `MNEMOS_I_ACCEPT_INSECURE=1`).
    fn is_hardware(&self) -> bool;

    /// Ask the hardware to unwrap the master key into caller-owned memory.
    ///
    /// The caller is responsible for placing the result in the locked arena and zeroising it.
    ///
    /// # Errors
    /// [`crate::Error::KeyCustody`] if the device is unavailable or refuses.
    fn unwrap_master_key(&self, wrapped: &[u8]) -> Result<Vec<u8>>;

    /// Wrap a freshly generated master key for storage in the manifest.
    ///
    /// # Errors
    /// [`crate::Error::KeyCustody`] on device failure.
    fn wrap_master_key(&self, master: &[u8]) -> Result<Vec<u8>>;

    /// Platform attestation of the running binary, if available.
    ///
    /// Returns `None` where unsupported. Present so the seam exists for the future-work item
    /// in ADR-0005 (sealing the master key to a PCR set), not because v1 uses it.
    ///
    /// # Errors
    /// [`crate::Error::KeyCustody`] if the platform supports attestation but the call fails.
    /// The default implementation never errors.
    fn attest(&self) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }
}

/// Fixed-slot encrypted blob storage.
///
/// Implementations must satisfy: **the physical address of a record is a function of its
/// [`Tag`] alone**, never of when it was written. Everything else in the history-independence
/// argument rests on that one sentence.
///
/// ## Why the key is a parameter
///
/// Slot addressing needs `K_idx`, but key material lives in the engine's locked arena and
/// `mnemos-core` cannot depend on `mnemos-crypto` (that is the wrong dependency direction).
/// Passing raw key bytes per call keeps the arena the single owner and makes every point that
/// touches key material visible in the signature rather than hidden in a field.
pub trait BlobStore: Send + Sync {
    /// Address a tag maps to, before collision resolution.
    fn home_slot(&self, k_idx: &[u8; 32], tag: Tag) -> SlotAddr;

    /// Write a batch. **Always exactly `epoch_writes` records**, dummy-padded by the caller,
    /// emitted in PRF-shuffled order.
    ///
    /// Returns **every slot actually written**, including slots touched by Robin Hood
    /// displacement. The caller emits one `SlotWrite` per returned address: emitting home slots
    /// instead would make the event stream disagree with the disk, and the attack harness
    /// consumes that stream *as the adversary's transcript* — so a mismatch means every attack
    /// result is measured against fiction.
    ///
    /// # Errors
    /// I/O or capacity failures.
    fn write_batch(
        &mut self,
        k_idx: &[u8; 32],
        epoch: Epoch,
        records: &[Record],
    ) -> Result<Vec<SlotAddr>>;

    /// Read one slot. The caller decrypts inside the locked arena.
    ///
    /// # Errors
    /// I/O failures. A missing entry is not an error here — dummy slots read successfully and
    /// are indistinguishable from live ones, which is the point.
    fn read_slot(&self, addr: SlotAddr) -> Result<Vec<u8>>;

    /// Rewrite exactly `compaction_slots` slots, regardless of how many are reclaimable.
    ///
    /// # Errors
    /// I/O failures.
    fn compact(&mut self, epoch: Epoch) -> Result<()>;

    /// Declared capacity in slots.
    fn capacity(&self) -> u64;
}

/// One record in a flush batch. Live and dummy records are byte-indistinguishable on disk.
#[derive(Clone)]
pub struct Record {
    /// Slot tag. Random for dummies.
    pub tag: Tag,
    /// Padded ciphertext. Random bytes for dummies.
    pub ciphertext: Vec<u8>,
    /// Target partition. Uniform-random for dummies.
    pub bucket: BucketId,
}

/// Encrypted approximate-nearest-neighbour index.
///
/// The **constant-volume invariant** is a contract of this trait, not an implementation
/// detail: [`Index::search`] must read exactly `params.slots_read_per_search()` index slots
/// for every query, whatever was asked and however many entries match.
pub trait Index: Send + Sync {
    /// Partition a quantised vector belongs to.
    fn assign(&self, v: &QuantVec) -> BucketId;

    /// Insert into a partition. Called only from a flush batch, never directly.
    ///
    /// Returns the partition the record was **actually** written to, which is not necessarily
    /// `bucket`: an implementation may spill a record whose assigned partition is full into a
    /// nearby one. The caller needs the real destination because the write is observable at
    /// that offset, and a leakage event naming the *assigned* partition would describe a write
    /// that did not happen --- the exact failure mode of finding F5.
    ///
    /// # Errors
    /// Capacity or I/O failures. Capacity here means every candidate partition was full, i.e.
    /// the record could not be indexed at all; it must not be discarded silently.
    fn insert(
        &mut self,
        bucket: BucketId,
        tag: Tag,
        v: &QuantVec,
        epoch: Epoch,
    ) -> Result<BucketId>;

    /// Return candidate tags for a query.
    ///
    /// Implementations **must** read `probe_width + decoy_buckets` partitions in full. Returning
    /// early because enough candidates were found would make I/O volume query-dependent — the
    /// exact leakage Xu et al. (CCS 2023) show survives forward and backward privacy.
    ///
    /// # Errors
    /// I/O failures.
    fn search(&self, q: &QuantVec) -> Result<Vec<(Tag, f32)>>;

    /// Emit a revocation token. Type-II backward privacy.
    ///
    /// # Errors
    /// I/O failures.
    fn revoke(&mut self, bucket: BucketId, tag: Tag, epoch: Epoch) -> Result<()>;

    /// Advance search tokens for every touched partition. The forward-privacy step; called
    /// once per epoch flush.
    ///
    /// # Errors
    /// I/O failures.
    fn advance_tokens(&mut self, epoch: Epoch) -> Result<()>;
}

/// The surface everything else talks to: the CLI, the FFI, the demo, and every experiment.
///
/// `mnemos-engine` implements it, and so does each baseline in `mnemos-baselines` — which is
/// what lets the attack harness point A3 at Sophos, Diana, or Mnemos without knowing which.
pub trait SearchableMemory: Send + Sync {
    /// Store a memory. Returns immediately; the write becomes durable at the next epoch flush.
    ///
    /// # Errors
    /// [`crate::Error::Locked`], capacity, or I/O failures.
    fn put(&mut self, text: &str, meta: &str, embedding: &QuantVec) -> Result<MemoryId>;

    /// Retrieve the `top_k` nearest memories. Constant-volume.
    ///
    /// # Errors
    /// [`crate::Error::Locked`] or I/O failures.
    fn search(&self, query: &QuantVec) -> Result<Vec<Hit>>;

    /// Replace a memory. Implemented as delete-then-put and therefore **returns a new
    /// [`MemoryId`]** — an in-place rewrite is a same-slot write, which is precisely the
    /// correlation signal a live adversary wants. See ADR-0004.
    ///
    /// # Errors
    /// [`crate::Error::NotFound`] or I/O failures.
    fn update(
        &mut self,
        id: MemoryId,
        text: &str,
        meta: &str,
        embedding: &QuantVec,
    ) -> Result<MemoryId>;

    /// Delete a memory. Type-II backward privacy.
    ///
    /// # Errors
    /// [`crate::Error::NotFound`] or I/O failures.
    fn delete(&mut self, id: MemoryId) -> Result<()>;

    /// Force an epoch flush regardless of the schedule. Tests and the demo only — calling this
    /// from application code defeats the batching that bounds Δ.
    ///
    /// # Errors
    /// I/O failures.
    fn flush(&mut self) -> Result<Epoch>;

    /// Non-secret shape data: entry count, current epoch, declared capacity.
    fn stat(&self) -> Stat;

    /// Unwrap the master key into the locked arena.
    ///
    /// # Errors
    /// [`crate::Error::KeyCustody`].
    fn unlock(&mut self) -> Result<()>;

    /// Zeroise the master key and the arena.
    fn lock(&mut self);
}

/// The only shape information the API exposes. Everything here is already in the leakage
/// profile as observable, so returning it costs nothing.
#[derive(Clone, Copy, Debug)]
pub struct Stat {
    /// Live entries.
    pub live_entries: u64,
    /// Current epoch.
    pub epoch: Epoch,
    /// Declared capacity.
    pub capacity: u64,
    /// Whether the master key is currently resident.
    pub unlocked: bool,
}

/// Retrieved memory alias, re-exported for trait users.
pub type RetrievedMemory = Memory;
