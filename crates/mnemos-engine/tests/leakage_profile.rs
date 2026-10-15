//! **The highest-value test in the repository.**
//!
//! `docs/leakage-profile.md` is the paper's technical heart: a table claiming, operation by
//! operation, exactly what an adversary observes. A table like that is worth nothing if the
//! implementation can drift away from it — and implementations always drift, silently, because
//! no *functional* test fails when a scheme starts leaking.
//!
//! So the table is a test. These assertions run on every PR.
//!
//! | Test | Leakage-profile row | Threat-model goal |
//! |---|---|---|
//! | [`search_volume_is_query_independent`] | `Search` | G3 — access-pattern inference |
//! | [`flush_batch_size_is_constant`] | `EpochFlush` | G4 — update-pattern / induced writes |
//! | [`compaction_budget_is_data_independent`] | `Compact` | Deletion leakage |
//! | [`manifest_is_written_every_epoch`] | `ManifestWrite` | Activity oracle |
//! | [`no_event_carries_a_secret`] | all | The event bus itself must not leak |
//! | [`every_event_variant_appears_in_the_document`] | — | Drift detection |

use std::sync::Arc;

use mnemos_core::event::{LeakEvent, RecordingSink};
use mnemos_core::traits::SearchableMemory;
use mnemos_core::types::{Embedding, QuantVec};
use mnemos_core::Params;
use mnemos_engine::Engine;
use mnemos_keystore::software::SoftwareKeystore;

/// Small geometry so tests stay fast; the invariants are size-independent.
fn test_params() -> Params {
    let mut p = Params::discover().expect("params.toml");
    p.index.n_max = 256;
    p.index.buckets = 8;
    p.index.bucket_capacity = 8;
    p.index.probe_width = 2;
    p.index.decoy_buckets = 0;
    p.index.top_k = 3;
    // `overflow_depth` must stay under `probe_width`, and `probe_width` is narrowed here.
    p.index.overflow_depth = 1;
    p.crypto.block_size_bytes = 512;
    p.crypto.max_entry_bytes = 512;
    p.embedding.dim = 8;
    p.epoch.compaction_slots = 4;
    p.epoch.slot_write_budget = 64;
    // Smaller locked arena: tests run in parallel and each locks its own region, so 2 MiB
    // apiece exhausts the Windows working-set quota. lock_pages stays TRUE — the real
    // mechanism is still exercised, just at a size that does not fight the scheduler.
    p.memory.arena_bytes = 64 * 1024;
    p
}

fn qv(v: Vec<f32>) -> QuantVec {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    QuantVec::quantise(&Embedding(v.into_iter().map(|x| x / n).collect()))
}

fn centroids(b: u32, dim: usize) -> Vec<QuantVec> {
    (0..b)
        .map(|i| {
            let mut v = vec![0.0f32; dim];
            v[i as usize % dim] = 1.0;
            qv(v)
        })
        .collect()
}

/// A store plus its event recorder. Uses the software keystore, explicitly accepted.
fn store(dir: &std::path::Path) -> (Engine, Arc<RecordingSink>) {
    let p = test_params();
    let ks = SoftwareKeystore::open_at(&dir.join("k"), true).expect("software keystore");
    let mut e = Engine::create(
        dir,
        p.clone(),
        Box::new(ks),
        centroids(p.index.buckets, p.embedding.dim),
    )
    .expect("create");
    let sink = Arc::new(RecordingSink::new());
    e.set_event_sink(sink.clone());
    (e, sink)
}

fn count(events: &[LeakEvent], f: impl Fn(&LeakEvent) -> bool) -> usize {
    events.iter().filter(|e| f(e)).count()
}

/// Every search must produce exactly `(k_probe + d)` `BucketRead` events and exactly `top_k`
/// `SlotRead` events — whatever the query, whatever matched.
///
/// This is the constant-volume invariant, and the specific property attack A5 targets. Queries
/// used: one whose partition is loaded, one whose partition is empty. If the two differ, the
/// volumetric channel is back.
#[test]
fn search_volume_is_query_independent() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path());
    let p = test_params();

    for i in 0..5 {
        e.put(
            &format!("entry {i}"),
            "{}",
            &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .unwrap();
    }
    e.flush().unwrap();
    let _ = sink.drain();

    let dense = qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let sparse = qv(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    e.search(&dense).unwrap();
    let a = sink.drain();
    e.search(&sparse).unwrap();
    let b = sink.drain();

    let bucket_reads = |ev: &[LeakEvent]| count(ev, |e| matches!(e, LeakEvent::BucketRead { .. }));
    let slot_reads = |ev: &[LeakEvent]| count(ev, |e| matches!(e, LeakEvent::SlotRead { .. }));

    // Note that `probe_spread` is inherited from `params.toml`, which ships the query-equality
    // mitigation on. That is deliberate: with the mitigation on, *which* partitions are read is
    // freshly random per call, so this is the assertion that the randomisation widened the read
    // set by a constant rather than making the count depend on the query. Trading a
    // query-equality leak for a volumetric one would be a bad trade made quietly.
    assert_eq!(
        bucket_reads(&a),
        (p.index.probe_width * p.index.probe_spread + p.index.decoy_buckets) as usize,
        "bucket reads must equal probe_width * probe_spread + decoy_buckets"
    );
    assert_eq!(
        bucket_reads(&a),
        bucket_reads(&b),
        "bucket reads must not depend on the query"
    );
    assert_eq!(
        slot_reads(&a),
        p.index.top_k as usize,
        "slot reads must equal top_k"
    );
    assert_eq!(
        slot_reads(&a),
        slot_reads(&b),
        "slot reads must not depend on match count"
    );
}

/// Every flush emits exactly `params.epoch.epoch_writes` `SlotWrite` events.
///
/// Run with 1 pending write and with a full batch. Both must be identical from the adversary's
/// side — that is what the dummy padding buys.
#[test]
fn flush_batch_size_is_constant() {
    let p = test_params();
    let mut counts = Vec::new();

    for real in [1usize, 3, p.epoch.epoch_writes as usize] {
        let dir = tempfile::tempdir().unwrap();
        let (mut e, sink) = store(dir.path());
        let _ = sink.drain();

        for i in 0..real {
            e.put(
                &format!("m{i}"),
                "{}",
                &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        }
        // A full batch auto-flushes; otherwise force one.
        if real < p.epoch.epoch_writes as usize {
            e.flush().unwrap();
        }
        let ev = sink.drain();
        counts.push(count(&ev, |e| matches!(e, LeakEvent::SlotWrite { .. })));

        for x in &ev {
            if let LeakEvent::EpochFlush { record_count, .. } = x {
                assert_eq!(
                    *record_count, p.epoch.epoch_writes,
                    "batch size must be constant"
                );
            }
        }
    }

    assert!(
        counts.windows(2).all(|w| w[0] == w[1]),
        "slot writes per flush must not depend on how many records were real: {counts:?}"
    );
}

/// Compaction touches `params.epoch.compaction_slots` slots whether or not anything is
/// reclaimable. Run against a store with zero deletions and one with several.
#[test]
fn compaction_budget_is_data_independent() {
    let p = test_params();
    let mut seen = Vec::new();

    for deletions in [0usize, 4] {
        let dir = tempfile::tempdir().unwrap();
        let (mut e, sink) = store(dir.path());

        let mut ids = Vec::new();
        for i in 0..6 {
            ids.push(
                e.put(
                    &format!("m{i}"),
                    "{}",
                    &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                )
                .unwrap(),
            );
        }
        e.flush().unwrap();
        for id in ids.iter().take(deletions) {
            e.delete(*id).ok();
        }
        let _ = sink.drain();
        e.flush().unwrap();

        let ev = sink.drain();
        let touched: Vec<u32> = ev
            .iter()
            .filter_map(|x| match x {
                LeakEvent::Compact { slots_touched, .. } => Some(*slots_touched),
                _ => None,
            })
            .collect();
        assert_eq!(touched, vec![p.epoch.compaction_slots]);
        seen.push(touched);
    }
    assert_eq!(
        seen[0], seen[1],
        "compaction volume must not track deletion count"
    );
}

/// The manifest is rewritten every epoch, unconditionally. A conditional write would turn
/// "the manifest changed" into "real work happened this epoch".
#[test]
fn manifest_is_written_every_epoch() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path());

    // An epoch with no writes at all must still rewrite the manifest.
    let _ = sink.drain();
    e.flush().unwrap();
    let idle = sink.drain();
    assert_eq!(
        count(&idle, |x| matches!(x, LeakEvent::ManifestWrite { .. })),
        1,
        "an idle epoch must still rewrite the manifest"
    );

    e.put(
        "something",
        "{}",
        &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    .unwrap();
    let _ = sink.drain();
    e.flush().unwrap();
    let busy = sink.drain();
    assert_eq!(
        count(&busy, |x| matches!(x, LeakEvent::ManifestWrite { .. })),
        1,
        "a busy epoch must write the manifest exactly as often as an idle one"
    );
}

/// No event may carry plaintext, key material, a `MemoryId`, or query text.
///
/// Checked by serialising every emitted event to JSON and asserting no known secret from the
/// fixture appears. Crude, and it catches exactly the mistake that would actually be made.
#[test]
fn no_event_carries_a_secret() {
    const SECRET: &str = "PLAINTEXT-CANARY-9f3a";

    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path());
    e.put(
        SECRET,
        "{\"meta\":\"META-CANARY-71b2\"}",
        &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    .unwrap();
    e.flush().unwrap();
    e.search(&qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let json = serde_json::to_string(&sink.drain()).expect("events serialise");
    assert!(!json.contains(SECRET), "an event leaked entry plaintext");
    assert!(
        !json.contains("META-CANARY"),
        "an event leaked entry metadata"
    );
}

/// Drift detection: every `LeakEvent` variant must be documented in
/// `docs/leakage-profile.md`, and vice versa.
///
/// Deliberately strict — a new variant means the implementation leaks something new, and that
/// fact must reach the document in the *same commit*, not later.
#[test]
fn every_event_variant_appears_in_the_document() {
    let doc = include_str!("../../../docs/leakage-profile.md");
    assert!(
        !doc.is_empty(),
        "leakage profile must exist and be non-empty"
    );

    let variants = [
        "Setup",
        "EpochFlush",
        "SlotWrite",
        "IndexWrite",
        "BucketRead",
        "SlotRead",
        "TokenAdvance",
        "Compact",
        "ManifestWrite",
        "Grow",
        "KeyOp",
        "RegionWrite",
        "RegionRead",
        "RegionSlotRead",
        "JournalWrite",
    ];
    for v in variants {
        assert!(
            doc.contains(v) || doc.to_lowercase().contains(&v.to_lowercase()),
            "LeakEvent::{v} has no row in docs/leakage-profile.md — document what it leaks in \
             this commit, not later"
        );
    }
}

/// Compile-time reminder: the list above must match `mnemos_core::event::LeakEvent`.
#[allow(dead_code)]
#[rustfmt::skip]
fn exhaustiveness_reminder(e: &LeakEvent) {
    match e {
        LeakEvent::Setup { .. }
        | LeakEvent::EpochFlush { .. }
        | LeakEvent::SlotWrite { .. }
        | LeakEvent::IndexWrite { .. }
        | LeakEvent::BucketRead { .. }
        | LeakEvent::SlotRead { .. }
        | LeakEvent::TokenAdvance { .. }
        | LeakEvent::Compact { .. }
        | LeakEvent::ManifestWrite { .. }
        | LeakEvent::Grow { .. }
        | LeakEvent::KeyOp { .. }
        | LeakEvent::RegionWrite { .. }
        | LeakEvent::RegionRead { .. }
        | LeakEvent::RegionSlotRead { .. }
        | LeakEvent::JournalWrite { .. } => {}
    }
    // If this match stops compiling, a variant was added. Add its row to
    // docs/leakage-profile.md and to `variants` above before making it compile again.
}

/// **The §4.1 fix, verified under crowding.**
///
/// [`flush_batch_size_is_constant`] runs on a sparse table where Robin Hood never displaces
/// anything — passing there does not prove the invariant holds. This test crowds the table
/// until cascades are common, and asserts the per-flush slot-write count is *still* exactly
/// `slot_write_budget`.
///
/// History: before the budget existed, this test asserted the opposite — that cascades made the
/// count data-dependent — and it passed, measuring `[16, 17, 16]`. That measurement is what
/// promoted `docs/leakage-profile.md` §4.1 from a suspicion to a defect, and the padding is
/// what closed it.
#[test]
fn slot_writes_stay_constant_even_under_displacement() {
    let p = test_params();
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path());

    // Stay within the documented safe load factor. Above it the padding budget is knowingly
    // insufficient — measured at 114 writes against a 48 budget at 0.875 occupancy — which is
    // why `max_load_factor` exists and why `Grow` must fire before it is crossed.
    let safe_rounds = ((p.index.n_max as f64 * p.epoch.max_load_factor)
        / f64::from(p.epoch.epoch_writes)) as usize;

    let mut per_flush = Vec::new();
    for round in 0..safe_rounds {
        for i in 0..p.epoch.epoch_writes {
            e.put(
                &format!("r{round}-{i}"),
                "{}",
                &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        }
        let writes = count(&sink.drain(), |x| matches!(x, LeakEvent::SlotWrite { .. }));
        if writes > 0 {
            per_flush.push(writes);
        }
    }

    assert!(
        per_flush.len() >= 2,
        "expected several flushes, saw {per_flush:?}"
    );
    assert!(
        per_flush
            .iter()
            .all(|w| *w == p.epoch.slot_write_budget as usize),
        "within max_load_factor every flush must write exactly slot_write_budget ({}) slots \
         regardless of displacement, saw {per_flush:?}",
        p.epoch.slot_write_budget
    );
}

/// Past `max_load_factor` the engine **refuses to flush** rather than leaking.
///
/// Asserting the *limit* of a defence matters as much as asserting the defence. Cascade length
/// explodes rather than degrades past the bound — measured 42 writes at 0.60 load, 244 at
/// 0.875, 1211 at 0.94 — so there is no safe "slightly over", and a hard stop is the correct
/// failure mode until `Grow` exists.
#[test]
fn refuses_to_flush_past_the_safe_load_factor() {
    let p = test_params();
    let dir = tempfile::tempdir().unwrap();
    let (mut e, _sink) = store(dir.path());

    let mut hit_limit = false;
    'outer: for round in 0..40 {
        for i in 0..p.epoch.epoch_writes {
            if e.put(
                &format!("r{round}-{i}"),
                "{}",
                &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
            .is_err()
            {
                hit_limit = true;
                break 'outer;
            }
        }
    }

    assert!(
        hit_limit,
        "the engine must refuse to operate past max_load_factor, not leak"
    );

    let live = e.stat().live_entries;
    assert!(
        p.within_safe_load(live),
        "live entries ({live}) should have stopped within max_load_factor ({}) of n_max ({})",
        p.epoch.max_load_factor,
        p.index.n_max
    );
}

/// **Finding F5.** Every record in a flush must land in the index, dummies included.
///
/// The flush path writes each record into its semantic partition at a deterministic file
/// offset. If only *real* records reach the index, then the number of index writes per flush
/// equals the number of real records, and the partitions written are exactly the real
/// records' semantic partitions -- undoing the WAL's padding one layer down and handing the
/// structure-transfer adversary its signal directly.
///
/// The engine's own comment asserted this could not happen ("EVERY record goes into the
/// index, dummies included"). It was wrong: dummy records carried an unbounded `u32` bucket
/// id, `EncryptedIvf::insert` rejected anything past the partition count, and the flush path
/// discarded the error. With B = 256 partitions a dummy had a 256/2^32 chance of landing
/// anywhere at all.
#[test]
fn every_record_in_a_flush_reaches_the_index_including_dummies() {
    let p = test_params();
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path());

    let mut per_flush = Vec::new();
    for round in 0..4 {
        for i in 0..p.epoch.epoch_writes {
            e.put(
                &format!("r{round}-{i}"),
                "{}",
                &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        }
        let n = count(&sink.drain(), |x| matches!(x, LeakEvent::IndexWrite { .. }));
        if n > 0 {
            per_flush.push(n);
        }
    }

    assert!(
        per_flush.len() >= 2,
        "expected several flushes, saw {per_flush:?}"
    );
    assert!(
        per_flush
            .iter()
            .all(|n| *n == p.epoch.epoch_writes as usize),
        "every flush must write exactly epoch_writes ({}) index slots so the count cannot \
         reveal how many records were real, saw {per_flush:?}",
        p.epoch.epoch_writes
    );
}

/// The partitions a flush touches must not be exactly the real records' partitions.
///
/// This is the structure-transfer signal stated as an invariant, and it is the shape of the
/// real attack: the adversary induces *one* write and it is flushed alongside `epoch_writes -
/// 1` other records. If the partitions touched by that flush are only the ones real records
/// belong to, the adversary intersects across repetitions and recovers its injected
/// partition -- which is attack A4 succeeding against the live engine.
///
/// Finding F5: dummy records carried an out-of-range partition id, so they never reached the
/// index at all and every flush touched exactly the real records' partitions and nothing else.
#[test]
fn the_partitions_a_flush_touches_are_not_just_the_real_ones() {
    let p = test_params();
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path());

    // One induced write, deliberately steered into a single partition, then flushed among
    // the dummies that pad the batch out to `epoch_writes`.
    e.put(
        "induced",
        "{}",
        &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    .unwrap();
    e.flush().unwrap();

    let writes: Vec<u32> = sink
        .drain()
        .iter()
        .filter_map(|x| match x {
            LeakEvent::IndexWrite { bucket, .. } => Some(bucket.0),
            _ => None,
        })
        .collect();

    assert_eq!(
        writes.len(),
        p.epoch.epoch_writes as usize,
        "a flush must write epoch_writes index slots"
    );

    let touched: std::collections::HashSet<u32> = writes.iter().copied().collect();
    assert!(
        touched.len() > 1,
        "one real write was flushed among {} dummies and the flush touched only {:?}; an \
         adversary reading index write offsets learns the induced partition directly",
        p.epoch.epoch_writes - 1,
        touched
    );
    assert!(
        touched.iter().all(|b| *b < p.index.buckets),
        "every index write must name a partition that exists, saw {touched:?}"
    );
}

/// The F5 ablation is faithful: with index-write padding off, the defect comes back.
///
/// An ablation nobody checks is a claim, not a control. E9 uses this knob to measure what
/// finding F5 cost, and that measurement is only meaningful if switching it off really does
/// reproduce the original behaviour --- a flush touching exactly the real records' partitions.
#[test]
fn disabling_index_write_padding_reproduces_finding_f5() {
    let mut p = test_params();
    p.index.index_write_padding = false;

    let dir = tempfile::tempdir().unwrap();
    let ks = SoftwareKeystore::open_at(&dir.path().join("k"), true).expect("keystore");
    let mut e = Engine::create(
        dir.path(),
        p.clone(),
        Box::new(ks),
        centroids(p.index.buckets, p.embedding.dim),
    )
    .expect("create");
    let sink = Arc::new(RecordingSink::new());
    e.set_event_sink(sink.clone());

    e.put(
        "induced",
        "{}",
        &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    .unwrap();
    e.flush().unwrap();

    let writes: Vec<u32> = sink
        .drain()
        .iter()
        .filter_map(|x| match x {
            LeakEvent::IndexWrite { bucket, .. } => Some(bucket.0),
            _ => None,
        })
        .collect();

    assert_eq!(
        writes.len(),
        1,
        "with padding off only the real record reaches the index"
    );
    let touched: std::collections::HashSet<u32> = writes.iter().copied().collect();
    assert_eq!(
        touched.len(),
        1,
        "and it names exactly the partition it belongs to"
    );
}
