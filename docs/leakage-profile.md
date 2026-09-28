# Leakage Profile

**What a τ-arrival adversary observes from Mnemos, operation by operation, and what it cannot
infer.** This file is the source of Table 5 in the paper (`paper/sections/06-mnemos.tex`).

> **This table is executable.** `crates/mnemos-engine` emits a `LeakEvent` for every
> disk-visible operation. `crates/mnemos-engine/tests/leakage_profile.rs` asserts that every event
> variant is documented here, and `crates/mnemos-engine/tests/channel_enumeration.rs` diffs the
> store byte for byte around an operation and requires every changed range to be named by an
> event. **If the code writes something no event describes, CI fails** (for the operations the
> enumeration covers; see §4.9).
>
> **Updated 2026-09-15** to match the revised paper and finding F7. The adversary is storage-level
> (`THREAT-MODEL.md` §2.2): it reads every file, sees which byte ranges are written and read and
> when, and never reads the memory of the unlocked process.

Notation: `B` partitions, `C` = `bucket_capacity`, `k` = `probe_width`, `s` = `probe_spread`,
`p` = `probe_retention`, `d` = `decoy_buckets`, `W` = `epoch_writes`, `H` = pre-τ cohort, `P` =
post-τ cohort. Defaults: B = 256, C = 64, k = 4, s = 2, p = 1.0, d = 0, W = 16.

---

## 1. Snapshot at τ (cohort H)

| Observable | Mnemos | Unpadded partitioned index | Local forward-private DSSE (Sophos/Diana-style) |
|---|---|---|---|
| File sizes | constants of the public parameters | grow with the corpus | grow with the corpus |
| **Per-partition occupancy** | **none**: every partition is `C × slot_len` bytes | **records per partition** ⚠ | not grouped by partition |
| **Partition of each record** | **none**: slots sealed, regions padded | **visible from region layout** ⚠ | not stored by partition |
| Slot contents | sealed (index: AES-256-GCM; records: XChaCha20-Poly1305); free slots are random bytes | sealed in the E4 model | sealed |
| Record tags | 128-bit PRF outputs of uniformly random identifiers | — | pseudorandom labels |
| **Per-keyword counters** | none persisted | — | **update count per keyword** ⚠ (Proposition 1) |
| Insertion order | none; slab placement depends only on tags (Robin Hood, tag-ordered tie-break) | within-region order | append order of the log |
| Manifest | fixed length, AES-256-GCM, fresh random nonce | — | — |
| Staging file (`wal.enc`) | fixed length, random bytes | — | — |

**Measured:**
- Structure transfer against the unpadded layout (simulation, E4, D2, 2,000 induced writes):
  `Adv_hist` = **0.587** [0.574, 0.600], ratio to `Adv_post` **0.943** [0.916, 0.970].
- The padded layout in the same simulation is zero by construction; it is a model, not a
  measurement.
- Against the compiled engine (E9), with the adversary *given* cohort-H membership, partition
  labelling from padded index writes is not distinguishable from chance at ≤ 16 writes per topic
  (largest mean accuracy 0.124 [0.064, 0.185] against 0.100).
- Counting persisted per-keyword state (E3 sensitivity): keyword recovery mean 0.051, max 0.22;
  record-level accuracy below the most-frequent-class baseline in all 75 configurations.

---

## 2. After τ (cohort P)

| Operation | Events emitted | Adversary observes | Cannot infer | Mechanism |
|---|---|---|---|---|
| `Setup(n_max)` | `Setup` ×1 | public parameters `n_max`, `B`, `C`, block size | anything about future content | preallocation |
| `put(text)` | **none** | nothing until the flush | content; which task caused which record | records staged in memory; `wal.enc` stays dummy-filled |
| `flush(e)` | `SlotWrite` ×`slot_write_budget` (64), `IndexWrite` ×`W`, `TokenAdvance` ×`B`, `Compact` ×1, `ManifestWrite` ×1, `EpochFlush` ×1 | 64 record-store slot writes and their positions; `W` index writes, each rewriting one whole partition region; compaction positions; one manifest rewrite; flush time | how many records were real; order within the batch; which slot in a partition changed; partition occupancy | dummy padding to `W` with independent random tags and uniform partitions; PRF-shuffled batch; slot-write budget; full-region re-encryption |
| `search(q)` | `BucketRead` ×(`s·k + d`), `SlotRead` ×`top_k` | the `s·k + d` partitions read (including the `k` genuine probes); the `top_k` record-store positions read (result positions, plus random positions for padding reads) | the query; how many candidates resolved; which probes were genuine (from a single search) | fixed probe count; fresh random spread; padding reads at random positions |
| `search`, repeated query | `BucketRead`, `SlotRead` | with `s = 1`, identical partition reads forever; with `s ≥ 2`, the genuine probes are common to every repetition | pairwise equality when `s ≥ 2` | randomised probe superset (§4.2) |
| `update(id)` | delete events, then `put` | two operations; update returns a new identifier | old or new content | `update = delete + put`, never same-slot |
| `delete(id)` | `SlotWrite` ×1 | the record-store slot overwritten immediately; the partition region holding the record rewritten (**no event**, §4.9) | content | slot overwritten with random bytes; region re-encrypted |
| `Compact(e)` | `Compact` ×1, carrying the slot addresses rewritten | which free slots in the rolling 64-slot window were rewritten | contents | fixed window per epoch; only free slots are rewritten, so occupied positions become visible over time (§4.6) |
| `ManifestWrite(e)` | `ManifestWrite` ×1 | the manifest was rewritten | whether anything changed | unconditional rewrite, fixed size, random nonce |
| `Grow()` | `Grow` ×1 | capacity doubled | which records triggered it | declared event; growth is not implemented, and the engine refuses to flush past the load factor |
| `lock` / `unlock` | `KeyOp` ×1 | a key operation occurred, and when | key material | hardware custody; key hierarchy erased on lock |
| **Injection** (adversary supplies a task) | via `put` → `flush` | its chosen stimulus; the flush its record landed in; the partitions of that flush | which index write is its record, from one flush | batch padding, PRF shuffle, uniform dummy partitions |

`TokenAdvance` is emitted for every partition on every flush and writes nothing to disk: the
forward-privacy token chains are derived and advanced in memory, and search does not consult them.

No event carries plaintext, metadata, a `MemoryId`, or query text; `no_event_carries_a_secret`
plants canaries and searches the serialised stream.

---

## 3. What the cohort-H guarantee rests on

The paper's Theorem 1 (`paper/sections/05-dual-epoch.tex`) shows that a store satisfying four
conditions has negligible advantage in the cohort-H game:

| Condition | Mnemos mechanism | Rows above |
|---|---|---|
| S1 fixed-size representation | preallocated slab, padded partitions, fixed manifest and staging file | §1 file sizes |
| S2 sealed slots | AES-256-GCM / XChaCha20-Poly1305 with random nonces; random free slots; PRF tags | §1 slot contents, tags |
| S3 data-independent updates | `W` records per flush, 64 slot writes, full-region index rewrites, fixed compaction window, unconditional manifest | §2 `flush` |
| S4 independent partition table, no overflow | deployment condition; E9 violates it by training on the corpus | §4.8 |

The game contains **no searches**. The search rows in §2 are outside the theorem; see §4.10.

The gap between cohorts is determined by the post-τ leakage. For an unpadded layout both cohorts
are exposed almost equally (ratio 0.943 on D2); for Mnemos, cohort H is protected by S1 to S4
regardless of what cohort P reveals.

---

## 4. Known residuals

A leakage table that lists only the wins is not evidence of anything.

### 4.1 Robin Hood displacement cascades: fixed, bound is empirical

Displacement made the per-flush slot-write count depend on local crowding. Every flush is now
padded up to `slot_write_budget = 64` real disk writes, and the engine refuses to flush past
`max_load_factor = 0.60`.

| Load factor | Worst raw cascade | As × `epoch_writes` |
|---|---|---|
| 0.40 | 25 | 1.6× |
| 0.50 | 37 | 2.3× |
| **0.60** | **42** | **2.6×** |
| 0.70 | 78 | 4.9× |
| 0.875 | 244 | 15× |
| 0.94 | 1,211 | 76× |

The bound is measured (`crates/mnemos-engine/tests/cascade_measurement.rs`), not proven. If a
cascade ever exceeded 64 writes the count would again be data-dependent; nothing caps it.

### 4.2 Repeated-query equality: not closed by randomisation (corrected 2026-09-15)

> **Correction.** An earlier version of this section claimed the shipped `s = 2` spread "closes
> pairwise query equality". That is true only against an adversary that tests **exact** equality of
> read sets, which is a strawman. The re-run E10 (v2 live, B = 64, k = 4) drives an
> **overlap-threshold** adversary and a **graph-clustering** adversary as well:
>
> | Probe mechanism | exact-equality J | overlap-threshold J | cluster ARI |
> |---|---:|---:|---:|
> | Randomised superset `s = 2` | 0.000 | **0.983** (τ = k) | 0.207 |
> | Grouped (group size 8) | 0.864 | 0.864 | 0.502 |
>
> So spread defeats the exact-equality adversary but **not** the overlap-threshold one (J = 0.98).
> The grouped mechanism leaks pairwise equality at group granularity (J = 0.86) but with a
> deterministic anonymity set equal to the group size. **Neither hides query equality from a
> determined adversary; only oblivious access over partitions does.** The retention (`p`) rows
> below still hold against intersection of *grouped* repetitions but do not rescue the pairwise
> claim.

Without randomisation the probe set is a deterministic function of the query vector. E5
(`s = 1`, B = 16, 240 records, 60 searches over 12 queries) measured 4 partition and 10 slot reads
per search with zero variance, volume-based recovery at chance (1/12), and pairwise query
equality 1.000 within and across epochs at a false-positive rate of 0.050.

Mitigation, measured by E10 on the fixed build (B = 64, k = 4, 240 records for leakage, 400 for
recall):

| `s` | `p` | pairwise (TPR − FPR) | recovery under intersection | partitions read | Recall@10 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | **1.000** | 1.000 | 4 | 0.620 |
| 2 | 1.0 | **0.000** | 1.000 | 8 | 0.684 |
| 4 | 1.0 | 0.000 | 0.909 | 16 | 0.765 |
| 8 | 1.0 | 0.000 | 0.364 | 32 | 0.956 |
| 4 | 0.9 | 0.000 | 0.909 | 16 | 0.744 |
| 4 | 0.75 | 0.000 | 0.727 | 16 | 0.681 |
| 4 | 0.5 | 0.000 | 0.182 | 16 | 0.624 |

Intersection recovery is computed over 11 query groups (0.091 per group) and varies between runs
because spread and retention use fresh randomness: the previous run gave 0.727, 0.455 and 0.364
for the `s = 4`, `s = 8` and `p = 0.5` rows. Mnemos ships `s = 2`, `p = 1.0`: exact-equality closed
but overlap-threshold recovery open (see the correction box above). Closing query equality requires
ORAM over partitions.

### 4.3 F7: four disk-level distinguishers, fixed 2026-09-15

1. Dummy tags were copied from dummy payloads, so dummy slab slots were identifiable and their
   partition ids were in plaintext; an observer could subtract dummies from every flush. Now drawn
   from independent randomness.
2. The manifest nonce was the epoch counter, which restarts on open (AES-GCM nonce reuse). Now a
   fresh random nonce.
3. Search padding reads went to fixed slots `0..top_k−1`. Now uniformly random.
4. Index insertion and revocation wrote a single slot, revealing which slots of each partition
   were free, and so per-partition occupancy, over time. Every insert and revoke now rewrites the
   whole region. This was invisible to both event checks because `IndexWrite` names a partition,
   not a slot.

Write-up: `the findings notes`.

### 4.4 F5: index writes were not padded (fixed 2026-08-27)

Dummy partition ids were unbounded `u32`s, rejected by the index, with the error discarded; every
flush wrote only its real records' partitions. Kept as the `index_write_padding` ablation. Against
the compiled engine the defect gives accuracy 0.467 [0.435, 0.499] (advantage 0.367) at 8 induced
writes per topic. Write-up: `the findings notes`.

### 4.5 F4: index slots were not encrypted (fixed 2026-08-14)

Index slots were written as `tag ‖ int8 vector` in the clear. Regression test
`the_index_never_holds_a_plaintext_embedding`. Write-up:
`the findings notes`.

### 4.6 Compaction reveals occupied slab positions over time

Compaction rewrites only the *free* slots in its window, and F6 made the event name them, so a
persistent observer accumulates the slab occupancy map within about `n_max / 64` epochs. Slab
placement depends only on uniformly random identifiers, so this reveals the number of occupied
slots (real records plus never-reclaimed dummies), not contents or topics. Write-up of the event
fix: `the findings notes`.

### 4.7 Instrumentation fidelity (fixed 2026-08-12)

The event stream once reported home slots while the disk write went to the Robin-Hood-resolved
slot, so attacks consumed an inaccurate transcript. `BlobStore::write_batch` now returns the slots
actually touched.

### 4.8 Overflow and the partition table

A record whose partition is full spills into one of its two next-nearest partitions
(`overflow_depth = 2 < k`). Which records spill depends on arrival order, so placement is canonical
only while no partition overflows (condition S4). S4 also requires a partition table independent
of cohort H; a table trained on stored records makes later write partitions depend on them.

### 4.9 Deletion: immediate, and partly unmodelled

A delete overwrites its record-store slot at once (one `SlotWrite`) and rewrites the partition
region that held the record. **The region rewrite emits no event**, and
`channel_enumeration.rs` covers flush, manifest and search but not delete or update. The partition
rewritten reveals which partition held the deleted record. The slab deletion also clears the slot
without a backward shift, so displaced records can become unfindable and the canonical layout is
lost; deletion is not evaluated and the theorem covers put-only histories.

### 4.10 Search access pattern

A search reads the genuine probes of its query and the record-store positions of its results.
Result positions are stable, so repeated retrievals of a record are linkable, and a cohort-H
record returned by a post-τ search is associated with the partitions probed. Combined with
partition labels learned from writes (§4.11), this can reveal the topics of retrieved cohort-H
records. Theorem 1 excludes searches for this reason. Closing it requires oblivious access to
partitions and records.

### 4.11 Partition labels from post-τ writes

Real records are always written to their assigned partitions; uniform dummy partitions only add
noise. Frequency analysis over many induced writes therefore labels partitions and exposes cohort-P
topics. E9 tested ≤ 16 writes per topic; the signal-to-noise ratio grows with the budget.

### 4.12 Flush timing

Flushes are triggered at `put` time (16 staged records, or a `put` more than 500 ms after the last
flush). A `put` after an idle period is flushed immediately, which reveals when it happened.

### 4.13 Dummies are never reclaimed

Dummy records are stored as live records in the slab and the index. Every flush consumes `W` slots
whatever the number of real records, and there is no growth mechanism.

---

## 5. Filling in a new row

1. Enumerate the disk deltas the implementation produces **from the bytes**, not from intent (§4.3,
   §4.7).
2. Write the "observes" cell from that enumeration, at the granularity an adversary sees (slot, not
   only partition).
3. Write the "cannot infer" cell and name the mechanism that makes it true.
4. Add the assertion to `tests/leakage_profile.rs` and, for disk channels,
   `tests/channel_enumeration.rs`.
5. If (2) and (3) cannot be reconciled, the design is wrong. Fix the design, or record the residual
   in §4. Never adjust the table to match a leak.

---

## 6. Mnemos v2 events (`crates/mnemos-v2`)

The v2 engine replaces the separate record store and index with one sealed partition file of `B`
equal regions, so its disk-visible operations emit a different, smaller event set. These three
variants are asserted present here by the same drift test
(`crates/mnemos-engine/tests/leakage_profile.rs`), and every changed byte is checked against them
by `crates/mnemos-v2/tests/audit.rs`.

| Variant | Emitted when | Adversary observes | Cannot infer |
|---|---|---|---|
| **RegionWrite** | a partition region is (re)written during a flush, or by the S3-ablation slot write | the **physical** region index, and the epoch; `slots` empty for a whole-region rewrite (the shipped case), or the specific slots under the `slot_writes_only` ablation | which slot in the region changed; how full the region is; whether the write carried a real record or was padding; the logical partition (hidden by the keyed permutation) |
| **RegionRead** | a whole region is read by a search, or by the unlock scan | the physical region index read | the query; how many candidates matched; which slots are live |
| **RegionSlotRead** | **evaluation ablation only** (`result_reads`): one slot re-read after ranking | the physical region and slot | — (the shipped engine never emits this; it exists so E5 can measure the result-dependent read channel that v1's separate record store has) |

A v2 flush emits exactly `RegionWrite × (w+1)` (one per staged record, uniform padding regions up
to `w`, and one maintenance region on the round-robin cursor), one `ManifestWrite`, and one
`EpochFlush`. A v2 `delete` and `update` emit **nothing** until the next flush (tombstone in the
sealed manifest); a v2 `search` emits only `RegionRead × (s·k or one group)` and never a
result-dependent read. `Setup`, `ManifestWrite`, `EpochFlush`, and `KeyOp` are shared with v1 and
documented above; v2 does not emit `SlotWrite`, `IndexWrite`, `BucketRead`, `SlotRead`,
`TokenAdvance`, `Compact`, or `Grow`, because it has no separate record store, no forward-privacy
token chains, and no compaction.

### 6.1 Two additions to v2 (2026-09-16)

| Change | Why | Effect on the event stream |
|---|---|---|
| **Padded refusal** (`v2.refusal_mode = "padded"`) | A refusal that wrote nothing told an observer that a region had reached `C`, and therefore what cohort H's load on that topic was (paper Prop. 4). | A refused `put` now owes its intended region a rewrite, so the flush emits the same `RegionWrite × (w+1)` an accepted record would have. `stats.ghost_writes` counts them. Measured: the channel is at AUC 1.000 under `"silent"` and 0.500 under `"padded"`. |
| **Full rewrite at close** (`v2.rewrite_at_lock`) | Two file versions otherwise diff to exactly the regions written between them, which is the write channel applied *before* τ. | `close()` emits `RegionWrite × B` plus one `ManifestWrite`. A cross-session diff then covers every region: measured precision falls from 0.050 to 0.0156, which is exactly the 1/B null. |
| **Randomised cursor start** (`v2.randomise_cursor`) | The maintenance cursor's resting position counted pre-τ flushes mod `B`. | No new variant; the cursor's first region after `unlock` is drawn uniformly. |

---

## 7. Mnemos-M events (`crates/mnemos-mem`)

Mnemos-M reuses `RegionWrite`, `RegionRead`, `RegionSlotRead`, `ManifestWrite`, `EpochFlush`,
`Setup` and `KeyOp` — it introduces no new variant, because it does strictly *less* at the storage
interface than Mnemos-P. What differs is the distribution behind the same events.

| Operation | Events emitted | Adversary observes | Cannot infer |
|---|---|---|---|
| `put` | **none** | nothing; the record is staged in the locked arena | that a write happened at all, until a flush — and a flush happens on the clock regardless |
| `flush` (clocked, every Δ) | `RegionWrite × (w+1)`, `ManifestWrite × 1`, `EpochFlush × 1` | which regions changed | whether any real record was carried; **the regions are uniformly distributed whether the rewrite carries a record or is padding**, because a record's slot was drawn uniformly at `put` time |
| `search` | **none** | nothing — no disk access occurs | everything: the query, the results, the read volume. Headers and payloads are resident in the locked arena |
| `delete` / `update` | **none** until the next flush | nothing at call time | which region held the record |
| `unlock` | `RegionRead × R`, `KeyOp` | that an unlock happened, and that the whole store was read | anything content-dependent; the scan is total by construction |
| `search` under the `disk_search` **ablation** | `RegionSlotRead × top_k` | the slots holding the results | — (evaluation only; this is the result-dependent read channel, reproduced so it can be measured) |

**The property that distinguishes M from P.** In Mnemos-P a flush's `RegionWrite` set is
`{content-derived regions of the staged records} ∪ {uniform padding} ∪ {cursor}`, so the real
records' regions are drawn from a *different* distribution than the padding. In Mnemos-M both are
uniform, so the multiset of regions a flush writes has a distribution that does not depend on the
records it carried, on their content, or on whether it carried any. `tests/audit.rs` asserts this
directly (`the_regions_a_flush_writes_are_uniform_under_identical_content`), with the
`content_placement` ablation as its negative control.

**Residuals for Mnemos-M.** Resident memory is the bound: the whole index, and by default every
payload, lives in the locked arena, so the design is limited to stores that fit (see the paper's
performance section for the measured crossover). The clock is visible: flushes occur every Δ while
unlocked, so an observer learns when the store was open, though not whether anything was written.
An `unlock` reads the entire store, which reveals that an unlock happened. Deletion is bounded by
`R·Δ` in wall-clock time, and no storage-layer measure reaches copies a synchronisation provider
has already retained.

### 7.1 The sweep flush (`m.flush_mode = "sweep"`, the shipped configuration from 2026-09-28)

The region flush above rewrites `w + 1` whole regions every Δ, which at the shipped geometry is
about 5.4 MB per flush. The sweep replaces it with the write pattern of a deterministic write-only
ORAM (Roche et al., CCS 2017): each flush rewrites the next `sweep_slots` slots after a cursor, and
a fixed-size sealed journal holds staged records until the cursor reaches a free slot.

| Operation | Events emitted | Adversary observes | Cannot infer |
|---|---|---|---|
| `flush` (clocked, every Δ) | `RegionWrite` naming exactly the `sweep_slots` slots after the cursor (one event per region the range touches), `ManifestWrite × 1`, **`JournalWrite × 1`**, `EpochFlush × 1` | the cursor range, which is a function of the flush count and a start drawn at unlock | whether any record was placed, staged, re-encrypted or refreshed: every written slot is a fresh sealing of fixed length |
| **`JournalWrite`** | the journal (`journal.enc`, fixed size) was rewritten | that a flush happened | how many staged records it holds (none to `journal_slots`) |
| `lock` / `close` | exactly one flush; further flushes only while more than `journal_slots` records are staged | that the store was locked | whether anything was written in the session. **Residual:** extra flushes at lock reveal that the backlog exceeded `journal_slots`, which requires the agent to have written faster than `sweep_slots * (1 - load)` records per Δ |
| `flush` with non-resident content (`payload_residency = "disk"`) | additionally `RegionSlotRead` for each slot of the cursor range | the cursor range was read before it was written | anything: the read set equals the write set |

### 7.2 Mnemos-M/disk (`m.header_residency = "disk"`)

Headers and payloads stay on disk; the locked arena holds only each slot's logical partition and
the position map. Writes are the sweep's, unchanged. A search emits `RegionSlotRead` for the header
of every live record in its probed partitions and for each result's payload. **W** observes
nothing of a search. **RW** observes which slots were read, which groups records by logical
partition over repeated searches: the read channel of any on-disk index, stated rather than hidden.
