//! Loader for `params.toml` — **the single source of truth**.
//!
//! Rust reads it here, Python reads it via `mnemos.params`, and the paper's configuration
//! table is generated from it. If a number appears in the paper,
//! or in any source file, it comes from here.
//!
//! Why this matters beyond tidiness: a paper and an implementation that disagree about
//! `k_probe` is a finding a reader makes *for* you. [`Params::hash`] is recorded in every
//! experiment's output JSON so a figure can always be traced to the exact configuration that
//! produced it.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// The full parameter set. Mirrors `params.toml` field for field.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Params {
    /// Schema version; bumped when the file's shape changes incompatibly.
    pub schema_version: u32,
    /// Cryptographic primitives and sizes.
    pub crypto: CryptoParams,
    /// Locked-arena settings.
    pub memory: MemoryParams,
    /// Key-custody provider preferences.
    pub keystore: KeystoreParams,
    /// Embedding model and quantisation.
    pub embedding: EmbeddingParams,
    /// IVF index geometry.
    pub index: IndexParams,
    /// The epoch schedule — the knob Theorem 1's Δ depends on.
    pub epoch: EpochParams,
    /// Mnemos v2 settings. Geometry (`B`, `C`, `k`, `s`, `top_k`, `dim`, block size, `w`,
    /// `epoch_ms`) is shared with the sections above; this holds only what v2 adds.
    #[serde(default)]
    pub v2: V2Params,
    /// Mnemos-M settings. Shares the same geometry so M and P are directly comparable at one
    /// configuration; this holds only what the memory-resident design adds.
    #[serde(default)]
    pub m: MParams,
}

/// See `params.toml` `[m]`.
///
/// Mnemos-M reuses `index.buckets` as its region count and `index.bucket_capacity` as slots per
/// region, so a measurement of M against P at one geometry is a measurement of the *design*
/// rather than of two different shapes. What differs is placement (uniformly random rather than
/// by content), where search runs (memory rather than disk), and when a flush happens (a clock
/// rather than a write).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MParams {
    /// Payload capacity of every slot, in blocks of `crypto.block_size_bytes`.
    pub payload_blocks: u32,
    /// `"memory"`: payloads are decrypted at unlock and held in the locked arena, so a search
    /// performs no disk read at all. `"disk"`: only headers are resident and a result's payload
    /// is read by position, which is safe against a write-observing adversary and reopens the
    /// result channel against a read-observing one.
    pub payload_residency: String,
    /// Δ, the clocked flush interval in milliseconds. A flush happens every Δ while the store is
    /// unlocked whether or not anything was written, which is what removes write timing and the
    /// derived-write channel, and what bounds deletion latency in wall-clock time.
    pub flush_ms: u64,
    /// Pending deletions the sealed manifest can hold before a flush purges them.
    pub tombstone_capacity: u32,
    /// `fsync` the slab file and manifest at the end of every flush.
    pub durable: bool,
    /// How a search ranks the resident index. `"exact"` scores every resident record.
    /// `"ivf"` keeps each record's nearest-centroid partition in the locked arena and scores only
    /// the records in the query's `probe_width` nearest partitions.
    ///
    /// The partition table costs Mnemos-P its write channel because P's partitions are *physical*:
    /// a record's region on disk is its partition. Mnemos-M's disk layout stays uniformly random
    /// either way, and probing happens in memory, where neither adversary class observes it, so
    /// the in-memory index buys sublinear search at no cost in leakage -- only in recall, which
    /// E11 measures. The logical index and the physical layout are separate choices.
    pub search: String,
    /// Partitions probed per search under `search = "ivf"`.
    pub probe_width: u32,
    /// What a clocked flush writes.
    ///
    /// `"regions"`: `epoch_writes + 1` whole regions, each uniformly distributed (the original
    /// design). `"sweep"`: the next `sweep_slots` slots after a cursor, a
    /// deterministic function of the flush count, in the manner of a write-only ORAM; a staged
    /// record is placed in the first free slot the cursor reaches and is held until then in a
    /// fixed-size sealed journal rewritten every flush. The sweep writes `C` times fewer bytes per
    /// flush and makes the write pattern independent of everything but the clock.
    pub flush_mode: String,
    /// Slots rewritten per flush under `flush_mode = "sweep"`.
    pub sweep_slots: u32,
    /// Staged records the sealed journal can hold under `flush_mode = "sweep"`.
    pub journal_slots: u32,
    /// What the sweep does when the agent stages faster than the cursor finds free slots.
    ///
    /// `"drain"`: `put` keeps accepting, and a lock flushes extra times until at most
    /// `journal_slots` records remain staged. The number of lock-time flushes then depends on the
    /// backlog, which is the sweep's one departure from a clock-only schedule (C1). E21 measures
    /// how often a session reaches it. `"refuse"`: `put` returns `CapacityExceeded` to the caller
    /// once `journal_slots` records are staged, so the backlog never exceeds the journal and a
    /// lock is always exactly one flush; the refusal is visible to the caller and to nothing on
    /// disk. This is back-pressure, the sweep's analogue of Mnemos-P's padded refusal.
    pub backlog: String,
    /// `"memory"`: every header is decrypted at unlock and searched in the locked arena.
    /// `"disk"`: only a position map and each slot's logical partition are resident, and a search
    /// reads the headers of its probed partitions' records by position. The write pattern is
    /// unchanged, so this is invisible to the W class and visible to RW; it is the variant for
    /// stores larger than memory. Requires `search = "ivf"`, `payload_residency = "disk"` and
    /// `flush_mode = "sweep"`.
    pub header_residency: String,
}

impl Default for MParams {
    fn default() -> Self {
        Self {
            payload_blocks: 1,
            payload_residency: "memory".into(),
            flush_ms: 500,
            tombstone_capacity: 4096,
            durable: true,
            search: "exact".into(),
            probe_width: 8,
            flush_mode: "regions".into(),
            sweep_slots: 8,
            journal_slots: 4,
            backlog: "drain".into(),
            header_residency: "memory".into(),
        }
    }
}

/// See `params.toml` `[v2]`.
// Each of these is an independent switch read from `params.toml`, not a state: any combination
// is a valid configuration, and the paper's ablations toggle them one at a time. A state machine
// or a set of two-variant enums would misdescribe that and make the TOML harder to read.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct V2Params {
    /// Payload capacity of every slot, in blocks of `crypto.block_size_bytes`. Fixed per store,
    /// so a record's size is never observable.
    pub payload_blocks: u32,
    /// `"spread"`: `k` genuine probes plus `(s - 1) k` fresh random partitions.
    /// `"grouped"`: partitions are grouped at setup into fixed groups of `group_size`, and a
    /// search reads exactly one whole group.
    pub probe_mode: String,
    /// Partitions per group under `probe_mode = "grouped"`. Must divide `index.buckets`.
    pub group_size: u32,
    /// Pending deletions the sealed manifest can hold before a flush purges them.
    pub tombstone_capacity: u32,
    /// `"on_demand"`: flush when `epoch_writes` records are staged, or on a `put` arriving more
    /// than `epoch_ms` after the last flush. `"periodic"`: flush exactly once per `epoch_ms`,
    /// driven by `tick`, whether or not anything is staged.
    pub flush_policy: String,
    /// Bound on records staged under the periodic policy.
    pub max_staged: u32,
    /// Apply the keyed permutation from logical partition to physical region.
    pub permute_partitions: bool,
    /// `fsync` the partition file and manifest at the end of every flush.
    pub durable: bool,
    /// What a `put` whose partition is full does on the wire.
    ///
    /// `"padded"`: the record is refused to the caller, but the store performs the same
    /// observable work an accepted record would have, so a refusal is not visible on disk.
    /// `"silent"`: the refusal returns without writing, which lets an observer who has labelled
    /// regions see that a topic's region did not receive the write it expected, and so learn
    /// that the topic's load exceeds `C` — a direct statement about the pre-compromise cohort.
    /// `"silent"` reproduces the earlier behaviour and exists so the channel can be measured.
    pub refusal_mode: String,
    /// Re-encrypt the whole partition file when the store is locked.
    ///
    /// Against an adversary holding two file versions from different sessions, the diff between
    /// them otherwise names every region written in between, which is the write channel applied
    /// before τ. A full rewrite at lock makes the between-session diff total and therefore
    /// uninformative. It costs one full file write per session.
    pub rewrite_at_lock: bool,
    /// Draw the maintenance cursor's starting position afresh at each unlock.
    ///
    /// The cursor advances one region per flush, so a persistent observer who identifies it
    /// learns the number of pre-τ flushes modulo `B`, which is the temporal-activity asset.
    /// A random start at unlock removes that inference and preserves the `B`-flush deletion
    /// bound within each session.
    pub randomise_cursor: bool,
}

impl Default for V2Params {
    fn default() -> Self {
        Self {
            payload_blocks: 1,
            probe_mode: "spread".into(),
            group_size: 8,
            tombstone_capacity: 4096,
            flush_policy: "on_demand".into(),
            max_staged: 1024,
            permute_partitions: true,
            durable: true,
            refusal_mode: "padded".into(),
            rewrite_at_lock: true,
            randomise_cursor: true,
        }
    }
}

/// See `params.toml` `[crypto]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CryptoParams {
    /// AEAD used for entry contents.
    pub content_aead: String,
    /// Content key size in bits.
    pub content_key_bits: u32,
    /// Content nonce size in bits.
    pub content_nonce_bits: u32,
    /// AEAD used for metadata structures.
    pub metadata_aead: String,
    /// Key-derivation function.
    pub kdf: String,
    /// PRF used for tags and tokens.
    pub prf: String,
    /// Truncated PRF tag width in bits.
    pub prf_tag_bits: u32,
    /// Puncturable PRF construction.
    pub pprf: String,
    /// Puncturable PRF tree depth.
    pub pprf_depth: u32,
    /// Record padding granularity in bytes.
    pub block_size_bytes: u32,
    /// Maximum entry size before chunk chaining.
    pub max_entry_bytes: u32,
}

/// See `params.toml` `[memory]`. Enforces threat-model assumption TA-4.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryParams {
    /// Locked arena size in bytes. Plaintext lives nowhere else inside the TCB.
    pub arena_bytes: usize,
    /// Whether to `VirtualLock`/`mlock` the arena. **Never set false outside a test.**
    pub lock_pages: bool,
    /// Zeroisation deadline in milliseconds, asserted by E1.
    pub zeroise_deadline_ms: u64,
}

/// See `params.toml` `[keystore]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeystoreParams {
    /// Providers in preference order; the first available wins.
    pub providers: Vec<String>,
    /// When true, refuse to start if no hardware provider is available. See the paper, Section 6.4.
    pub require_hardware: bool,
}

/// See `params.toml` `[embedding]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingParams {
    /// Model identifier.
    pub model: String,
    /// Embedding dimensionality.
    pub dim: usize,
    /// Distance metric.
    pub metric: String,
    /// Quantisation scheme.
    pub quantisation: String,
    /// Faster model used for parameter sweeps.
    pub speed_model: String,
    /// Its dimensionality.
    pub speed_dim: usize,
}

/// `serde` default for [`IndexParams::probe_spread`]: no widening.
fn one() -> u32 {
    1
}

/// `serde` default for [`IndexParams::probe_retention`]: every real probe is read.
fn one_f32() -> f32 {
    1.0
}

/// `serde` default for [`IndexParams::index_write_padding`]: dummies reach the index.
fn yes() -> bool {
    true
}

/// `serde` default for [`IndexParams::overflow_depth`]: spill into the two next-nearest
/// partitions before giving up on indexing a record.
fn two() -> u32 {
    2
}

/// See `params.toml` `[index]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexParams {
    /// Number of IVF partitions, `B`.
    pub buckets: u32,
    /// Slots per partition, `C`. Fully padded — occupancy is never observable.
    pub bucket_capacity: u32,
    /// `k_probe`. **Always exactly this many buckets are read, query-independently.**
    pub probe_width: u32,
    /// `d`. Extra buckets read purely to blunt access-pattern inference.
    pub decoy_buckets: u32,
    /// `s`. **Probe superset multiplier.** A search reads `s * probe_width` partitions rather
    /// than `probe_width`, the extra ones drawn with *fresh* randomness on every call.
    ///
    /// This exists because `decoy_buckets` does not close query equality and cannot. Decoys are
    /// PRF-derived from the query so that repeating a query inside an epoch cannot cancel them
    /// by intersection — which necessarily means the whole read set is a deterministic function
    /// of the query, so two searches for the same thing are recognisable forever. Attack A5
    /// measures exactly that and recovers query equality at 1.000, across epochs.
    ///
    /// Fresh randomness breaks the determinism, at `s x` the read volume. `s = 1` reproduces
    /// the original behaviour exactly and is the default, so this knob never silently changes
    /// a measurement.
    #[serde(default = "one")]
    pub probe_spread: u32,
    /// `p`. Probability that each *genuine* top-`k_probe` partition is actually read.
    ///
    /// Superset widening alone defeats pairwise signature equality but not intersection: if the
    /// real probes are present in every repetition, an adversary who can group repeats
    /// intersects them out. Dropping each real probe independently with probability `1 - p`
    /// makes a real probe survive `r` intersected repetitions with probability only `p^r`.
    ///
    /// The cost is recall: a dropped probe is a partition genuinely not searched. `p = 1.0`
    /// reproduces the original behaviour and is the default.
    #[serde(default = "one_f32")]
    pub probe_retention: f32,
    /// **Evaluation only.** When `false`, dummy records are not written into the index, so a
    /// flush touches only the partitions its *real* records belong to.
    ///
    /// This reproduces finding F5 exactly, and exists so the cost of that defect can be
    /// measured rather than asserted. Experiment E9 sweeps it: with padding on, structure
    /// transfer against the live engine sits at chance; with it off, the flush hands the
    /// adversary the induced partition directly.
    ///
    /// Defaults to `true`. Setting it to `false` in a deployed store is a confidentiality
    /// break, which is why it is documented here rather than hidden behind a build flag --- a
    /// knob nobody can find is a knob nobody can check.
    #[serde(default = "yes")]
    pub index_write_padding: bool,
    /// How many *next-nearest* partitions an insert may spill into when its assigned partition
    /// is full. `0` restores the old behaviour: a record whose partition is full is silently
    /// not indexed.
    ///
    /// A full partition used to be tolerated in silence -- the entry stayed durable in the slab
    /// and simply never appeared in a search result again. That is a correctness hole with no
    /// symptom: the store reports the entry as live, `stat()` counts it, and only a recall
    /// measurement against known ground truth would ever notice.
    ///
    /// Spilling to the next-nearest centroid is the natural IVF repair, and it is the only one
    /// that keeps reads constant-volume. An overflow *area* -- the textbook chain -- would have
    /// to be read by any search that probes the overflowing partition, which makes read volume
    /// depend on occupancy and hands back exactly the channel `bucket_capacity` padding exists
    /// to close. Spilling instead relies on the searcher already reading `probe_width`
    /// partitions: a record displaced to its second-nearest centroid is still found by any
    /// query for which that centroid is in the top `k_probe`.
    ///
    /// The cost is paid in recall, not volume, and it is measured rather than assumed
    /// (`overflow_spill_keeps_the_record_findable`).
    ///
    /// **What this costs in history independence.** Which records spill is a function of
    /// arrival order: for a set that overflows, inserting `x` then `y` and inserting `y` then
    /// `x` place different records in the spill partition. The strong form of history
    /// independence -- one canonical plaintext state per content set -- therefore holds only
    /// while no partition overflows. The observable form still holds unconditionally, because
    /// every partition is padded to `bucket_capacity` and every slot is sealed, so no
    /// keyless observer can tell the two states apart. We record the distinction rather than
    /// eliding it; the alternative repairs (rank-by-PRF with eviction) reintroduce the
    /// displacement cascades that made per-flush write counts data-dependent, which is a
    /// strictly worse trade.
    #[serde(default = "two")]
    pub overflow_depth: u32,
    /// Results returned. Also the exact number of content-slot reads per search.
    pub top_k: u32,
    /// Declared capacity; the store is preallocated to this size at init.
    pub n_max: u64,
    /// Partitioning algorithm.
    pub partitioner: String,
    /// Collision policy. Must be uniquely representing — see the paper, Section 5.
    pub collision_policy: String,
}

/// See `params.toml` `[epoch]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochParams {
    /// Time-based flush trigger in milliseconds.
    pub epoch_ms: u64,
    /// Count-based flush trigger. Also the exact size of every flushed batch.
    pub epoch_writes: u32,
    /// Slots rewritten per epoch by compaction, regardless of what is reclaimable.
    pub compaction_slots: u32,
    /// WAL length in records. Always equal to `epoch_writes`.
    pub wal_records: u32,
    /// Fixed slot writes per flush. Every flush is padded up to exactly this many, so Robin
    /// Hood displacement cascades cannot make the write count data-dependent.
    pub slot_write_budget: u32,
    /// Occupancy above which cascades can exceed `slot_write_budget`, reopening the leak.
    ///
    /// **A security parameter.** `Grow` must fire before this is crossed; running past it
    /// silently restores a data-dependent write count.
    pub max_load_factor: f64,
}

impl Params {
    /// Load from a `params.toml` path.
    ///
    /// # Errors
    /// Returns [`Error::Params`] if the file is missing, malformed, or fails [`Self::validate`].
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| Error::Params(format!("reading {}: {e}", path.as_ref().display())))?;
        let params: Self = toml::from_str(&text)
            .map_err(|e| Error::Params(format!("parsing params.toml: {e}")))?;
        params.validate()?;
        Ok(params)
    }

    /// Locate `params.toml` by walking up from the current directory to the workspace root.
    ///
    /// # Errors
    /// Returns [`Error::Params`] if no `params.toml` is found.
    pub fn discover() -> Result<Self> {
        let mut dir =
            std::env::current_dir().map_err(|e| Error::Params(format!("cwd unavailable: {e}")))?;
        loop {
            let candidate = dir.join("params.toml");
            if candidate.is_file() {
                return Self::load(candidate);
            }
            if !dir.pop() {
                return Err(Error::Params(
                    "params.toml not found in any ancestor directory".into(),
                ));
            }
        }
    }

    /// Cross-field invariants that the type system cannot express.
    ///
    /// These are not style checks. Each one, if violated, silently breaks a security property
    /// while leaving the system apparently working — which is the worst possible failure mode
    /// for this project.
    ///
    /// # Errors
    /// Returns [`Error::Params`] describing the first violated invariant.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 2 {
            return Err(Error::Params(format!(
                "unsupported schema_version {}; this build expects 2",
                self.schema_version
            )));
        }

        // The WAL is always exactly one batch long. If it were longer, its fill level would
        // encode the number of pending writes — a direct update-pattern leak.
        if self.epoch.wal_records != self.epoch.epoch_writes {
            return Err(Error::Params(
                "epoch.wal_records must equal epoch.epoch_writes, or WAL fill level leaks \
                 pending-write count"
                    .into(),
            ));
        }

        // Robin Hood with a total order on tags is uniquely representing; plain linear probing
        // is not (the first of two colliding entries takes the home slot, so layout depends on
        // arrival order). This is history independence.
        if self.index.collision_policy != "robin-hood" {
            return Err(Error::Params(format!(
                "index.collision_policy = {:?} is not uniquely representing; history \
                 independence would be violated",
                self.index.collision_policy
            )));
        }

        // Probing more partitions than exist would make probe width query-dependent at the
        // boundary, breaking the constant-volume invariant.
        if self
            .index
            .probe_width
            .saturating_mul(self.index.probe_spread)
            + self.index.decoy_buckets
            > self.index.buckets
        {
            return Err(Error::Params(
                "probe_width + decoy_buckets exceeds bucket count; constant-volume reads \
                 cannot be maintained"
                    .into(),
            ));
        }

        if self.index.probe_spread == 0 {
            return Err(Error::Params(
                "index.probe_spread must be at least 1".into(),
            ));
        }

        // A record spilled past the probe width is unfindable *even by a search for its own
        // vector*: the search reads the top `k_probe` partitions, so a record sitting in its
        // own `(k_probe + 1)`-th nearest is never read. Allowing that would replace a silent
        // drop with a silent misplacement, which is the same bug wearing a hat.
        if self.index.overflow_depth >= self.index.probe_width {
            return Err(Error::Params(format!(
                "index.overflow_depth = {} must be less than probe_width = {}; a record                  spilled beyond the probe width cannot be found by any query",
                self.index.overflow_depth, self.index.probe_width
            )));
        }

        // Retention is a probability. Outside (0, 1] it is either meaningless or drops every
        // real probe, which would make search return nothing at all.
        if !(self.index.probe_retention > 0.0 && self.index.probe_retention <= 1.0) {
            return Err(Error::Params(format!(
                "index.probe_retention = {} is not in (0, 1]",
                self.index.probe_retention
            )));
        }

        if self.index.top_k == 0 || self.index.probe_width == 0 {
            return Err(Error::Params(
                "top_k and probe_width must be non-zero".into(),
            ));
        }

        if !self.memory.lock_pages {
            return Err(Error::Params(
                "memory.lock_pages = false violates threat-model assumption TA-4 (plaintext \
                 must never reach swap)"
                    .into(),
            ));
        }

        // A budget below epoch_writes could not even cover a cascade-free batch, and padding
        // "up" would silently become truncation.
        if self.epoch.slot_write_budget < self.epoch.epoch_writes {
            return Err(Error::Params(
                "epoch.slot_write_budget must be >= epoch.epoch_writes, or a flush cannot be                  padded to a constant width"
                    .into(),
            ));
        }

        if !(0.0..=1.0).contains(&self.epoch.max_load_factor) {
            return Err(Error::Params(
                "epoch.max_load_factor must be in [0, 1]".into(),
            ));
        }

        if !self.crypto.block_size_bytes.is_power_of_two() {
            return Err(Error::Params(
                "crypto.block_size_bytes must be a power of two".into(),
            ));
        }

        Ok(())
    }

    /// SHA-256 of the canonical serialisation, hex-encoded.
    ///
    /// Recorded alongside every experiment result so a figure can be traced to its exact
    /// configuration.
    #[must_use]
    pub fn hash(&self) -> String {
        use sha2::{Digest, Sha256};
        // Hash the canonical re-serialisation rather than the raw file, so formatting and
        // comment edits do not invalidate results while any semantic change does. The Python
        // side hashes raw bytes and is deliberately stricter — over-invalidating is safe,
        // under-invalidating would let two different configurations share a hash.
        let canonical = toml::to_string(self).unwrap_or_default();
        let digest = Sha256::digest(canonical.as_bytes());
        digest.iter().fold(String::with_capacity(64), |mut acc, b| {
            let _ = write!(acc, "{b:02x}");
            acc
        })
    }

    /// Whether `live` entries against `n_max` is within the safe load factor.
    ///
    /// Above it, Robin Hood cascades can exceed `slot_write_budget` and the per-flush write
    /// count becomes data-dependent again — see `docs/leakage-profile.md` §4.1.
    #[must_use]
    pub fn within_safe_load(&self, live: u64) -> bool {
        (live as f64) <= (self.index.n_max as f64) * self.epoch.max_load_factor
    }

    /// Total index slots read per search — `(s * k_probe + d) x C`.
    ///
    /// **This value must be independent of the query.** It is the constant-volume invariant
    /// expressed as one function, so there is exactly one place to check it.
    ///
    /// `probe_spread` and `probe_retention` both leave that independence intact, by design:
    /// widening multiplies the count by a constant, and retention substitutes a random
    /// partition for each dropped real one rather than reading fewer. A mitigation that let the
    /// count vary with the query would trade a query-equality leak for a volumetric one.
    #[must_use]
    pub fn slots_read_per_search(&self) -> u32 {
        (self.index.probe_width * self.index.probe_spread + self.index.decoy_buckets)
            * self.index.bucket_capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_params_are_valid() {
        let p = Params::discover().expect("params.toml should be discoverable from the workspace");
        p.validate()
            .expect("shipped params.toml must satisfy its own invariants");
    }

    #[test]
    fn params_hash_is_stable_and_sensitive() {
        let p = Params::discover().unwrap();
        assert_eq!(p.hash(), p.hash(), "hash must be deterministic");
        let mut q = p.clone();
        q.index.probe_width += 1;
        assert_ne!(p.hash(), q.hash(), "a semantic change must change the hash");
    }

    #[test]
    fn constant_volume_is_query_independent() {
        let p = Params::discover().unwrap();
        // Not a real test of the property — that is E4's job — but it pins the formula so a
        // future refactor cannot quietly make it depend on anything else.
        //
        // `probe_spread` belongs in the formula: widening multiplies the slot count by a
        // constant, which keeps the count independent of the query. This assertion was written
        // against the pre-mitigation formula and started failing the moment `params.toml`
        // shipped `s = 2` — which is the correct behaviour for it to have.
        assert_eq!(
            p.slots_read_per_search(),
            (p.index.probe_width * p.index.probe_spread + p.index.decoy_buckets)
                * p.index.bucket_capacity
        );
    }

    #[test]
    fn the_shipped_default_has_the_query_equality_mitigation_on() {
        // E10 measures pairwise query-equality recovery at Youden's J = 1.000 with `s = 1` and
        // 0.000 with `s = 2`. Shipping the measurable configuration rather than the safe one is
        // defensible for a benchmark and not for a store, so the default is pinned here: a
        // silent revert to `s = 1` should break a test, not a deployment.
        let p = Params::discover().unwrap();
        assert!(
            p.index.probe_spread >= 2,
            "params.toml shipped probe_spread = {}; query equality is recoverable at s = 1",
            p.index.probe_spread
        );
    }

    #[test]
    fn the_shipped_default_indexes_records_whose_partition_is_full() {
        // `overflow_depth = 0` is the old behaviour: a record whose partition is full is
        // durable, counted as live, and absent from every search. It has no symptom short of a
        // recall measurement, so it gets an assertion instead.
        let p = Params::discover().unwrap();
        assert!(
            p.index.overflow_depth >= 1,
            "params.toml shipped overflow_depth = 0"
        );
    }
}
