//! Close-and-reopen behaviour.
//!
//! Every test here exists because building the CLI surfaced two defects that the entire rest of
//! the suite missed, both for the same structural reason: **nothing else ever reopened a
//! store.** Tests created an engine, exercised it, and dropped it. A whole class of bug lives
//! past that boundary.
//!
//! | Defect | Symptom | Caught by |
//! |---|---|---|
//! | Index occupancy reset to empty on `open` | Next flush overwrote live entries from slot 0. Writes kept succeeding; only *older* entries vanished. | [`entries_survive_a_reopen_and_a_second_flush`] |
//! | Centroid table recomputed per process | A different k-means run repartitions everything, so a store stops finding its own contents. | [`reopened_store_uses_the_same_partitioning`] |
//!
//! Both are silent. Neither produces an error, a panic, or a failing assertion anywhere else —
//! which is precisely the profile of a bug that ships.

use mnemos_core::traits::SearchableMemory;
use mnemos_core::types::{Embedding, MemoryId, QuantVec};
use mnemos_core::Params;
use mnemos_engine::Engine;
use mnemos_keystore::software::SoftwareKeystore;

const DIM: usize = 16;

fn test_params() -> Params {
    let mut p = Params::discover().expect("params.toml");
    p.index.n_max = 512;
    p.index.buckets = 8;
    p.index.bucket_capacity = 64;
    p.index.probe_width = 8;
    p.index.decoy_buckets = 0;
    // Larger than the number of entries any test writes. `search` returns at most `top_k`
    // hits, so a smaller value would make these tests measure RANKING competition between
    // epochs rather than retrievability — which is exactly the confound that made the first
    // version of `entries_survive_a_reopen_and_a_second_flush` fail for the wrong reason.
    p.index.top_k = 40;
    p.crypto.block_size_bytes = 512;
    p.crypto.max_entry_bytes = 512;
    p.embedding.dim = DIM;
    p.epoch.compaction_slots = 4;
    p.memory.arena_bytes = 64 * 1024;
    p
}

fn qv(v: Vec<f32>) -> QuantVec {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    QuantVec::quantise(&Embedding(v.into_iter().map(|x| x / n).collect()))
}

/// A vector pointing mostly along axis `i`, so partition assignment is predictable.
fn axis(i: usize, jitter: f32) -> QuantVec {
    let mut v = vec![0.0f32; DIM];
    v[i % DIM] = 1.0;
    v[(i + 1) % DIM] = jitter;
    qv(v)
}

fn centroids(b: u32) -> Vec<QuantVec> {
    (0..b).map(|i| axis(i as usize, 0.0)).collect()
}

fn store_dir(scratch: &std::path::Path) -> std::path::PathBuf {
    scratch.join("store")
}

fn create(scratch: &std::path::Path) -> Engine {
    let p = test_params();
    let ks = SoftwareKeystore::open_at(&scratch.join("keystore.bin"), true).expect("keystore");
    Engine::create(
        &store_dir(scratch),
        p.clone(),
        Box::new(ks),
        centroids(p.index.buckets),
    )
    .expect("create")
}

fn reopen(scratch: &std::path::Path) -> Engine {
    let p = test_params();
    let ks = SoftwareKeystore::open_at(&scratch.join("keystore.bin"), true).expect("keystore");
    let mut e = Engine::open(&store_dir(scratch), p, Box::new(ks)).expect("open");
    e.unlock().expect("unlock");
    e
}

/// Write one full epoch of entries, all in the same partition so they compete for slots.
fn fill_one_epoch(e: &mut Engine, tag: &str) -> Vec<String> {
    let p = test_params();
    let mut texts = Vec::new();
    for i in 0..p.epoch.epoch_writes {
        let text = format!("{tag}-{i}");
        e.put(&text, "{}", &axis(3, 0.01 * i as f32)).expect("put");
        texts.push(text);
    }
    e.flush().expect("flush");
    texts
}

fn found(e: &Engine, query: &QuantVec, wanted: &[String]) -> usize {
    let hits = e.search(query).expect("search");
    wanted
        .iter()
        .filter(|w| hits.iter().any(|h| h.memory.text == **w))
        .count()
}

#[test]
fn entries_survive_a_reopen_and_a_second_flush() {
    // THE regression test. Before the fix, `Engine::open` rebuilt index occupancy as all-false,
    // so the second flush inserted from slot 0 of each partition and overwrote the first
    // epoch's entries. Nothing errored — the first batch simply stopped being retrievable.
    let d = tempfile::tempdir().unwrap();

    let first = {
        let mut e = create(d.path());
        fill_one_epoch(&mut e, "first")
    };

    let mut e = reopen(d.path());
    let before = found(&e, &axis(3, 0.0), &first);
    assert!(
        before > 0,
        "nothing from the first epoch survived the reopen itself"
    );

    fill_one_epoch(&mut e, "second");

    let after = found(&e, &axis(3, 0.0), &first);
    assert_eq!(
        after,
        before,
        "the second flush destroyed {} of {before} first-epoch entries — index occupancy was \
         not restored on open",
        before - after
    );
}

#[test]
fn reopened_store_uses_the_same_partitioning() {
    // The centroid table has to come back byte-identical. A recomputed table would reassign
    // every entry to a different partition, so the store would keep working and stop finding
    // anything it had already written.
    let d = tempfile::tempdir().unwrap();
    let probe = axis(5, 0.02);

    let (written, before) = {
        let mut e = create(d.path());
        let mut written = Vec::new();
        for i in 0..8u8 {
            let text = format!("topic5-{i}");
            e.put(&text, "{}", &axis(5, 0.01 * f32::from(i)))
                .expect("put");
            written.push(text);
        }
        e.flush().expect("flush");
        let n = found(&e, &probe, &written);
        (written, n)
    };
    assert!(
        before > 0,
        "the store could not find its own entries before closing"
    );

    let e = reopen(d.path());
    assert_eq!(
        found(&e, &probe, &written),
        before,
        "a reopened store retrieved a different set — the partitioning changed across processes"
    );
}

#[test]
fn a_reopened_store_starts_locked() {
    // `open` must not leave the master key resident: process lifetime is not a bound we
    // control, so "unlocked by default" would mean "key in memory until reboot".
    let d = tempfile::tempdir().unwrap();
    {
        let mut e = create(d.path());
        fill_one_epoch(&mut e, "x");
    }

    let p = test_params();
    let ks = SoftwareKeystore::open_at(&d.path().join("keystore.bin"), true).expect("keystore");
    let e = Engine::open(&store_dir(d.path()), p, Box::new(ks)).expect("open");

    assert!(!e.stat().unlocked, "open() left the store unlocked");
    assert!(
        matches!(e.search(&axis(0, 0.0)), Err(mnemos_core::Error::Locked)),
        "a locked store answered a search"
    );
}

#[test]
fn deletes_survive_a_reopen() {
    // Backward privacy has to persist. If a delete were only reflected in memory, reopening
    // would resurrect the entry — a store that forgets its forgetting.
    let d = tempfile::tempdir().unwrap();

    let (id, text) = {
        let mut e = create(d.path());
        let text = "delete me".to_owned();
        let id = e.put(&text, "{}", &axis(2, 0.0)).expect("put");
        for i in 0..test_params().epoch.epoch_writes - 1 {
            e.put(&format!("keep-{i}"), "{}", &axis(2, 0.01 * i as f32))
                .expect("put");
        }
        e.flush().expect("flush");
        e.delete(id).expect("delete");
        e.flush().expect("flush");
        (id, text)
    };
    let _ = id;

    let e = reopen(d.path());
    let hits = e.search(&axis(2, 0.0)).expect("search");
    assert!(
        !hits.iter().any(|h| h.memory.text == text),
        "a deleted entry came back after a reopen"
    );
    assert!(
        !hits.is_empty(),
        "the reopen lost the entries that were NOT deleted"
    );
}

#[test]
fn live_count_survives_a_reopen() {
    let d = tempfile::tempdir().unwrap();
    let before = {
        let mut e = create(d.path());
        fill_one_epoch(&mut e, "count");
        e.stat().live_entries
    };
    assert!(before > 0);
    assert_eq!(reopen(d.path()).stat().live_entries, before);
}

#[test]
fn an_unknown_id_is_not_found_rather_than_a_panic() {
    let d = tempfile::tempdir().unwrap();
    let mut e = create(d.path());
    fill_one_epoch(&mut e, "x");
    assert!(matches!(
        e.delete(MemoryId::random()),
        Err(mnemos_core::Error::NotFound)
    ));
}

/// `actual_bucket` is what the evaluation scores an attack against, so it must report where a
/// record really is: nowhere while staged, and in exactly the bucket its content selects once
/// flushed. It finds the record by its tag, so a spill or a quantisation boundary cannot make it
/// disagree with the index -- which a recomputed nearest centroid can.
#[test]
fn actual_bucket_reports_where_a_record_is_indexed() {
    let scratch = tempfile::tempdir().expect("tempdir");
    let mut e = create(scratch.path());
    let ids: Vec<MemoryId> = (0..8)
        .map(|i| e.put(&format!("m{i}"), "{}", &axis(i, 0.05)).expect("put"))
        .collect();
    assert!(
        ids.iter()
            .all(|id| e.actual_bucket(*id).expect("lookup").is_none()),
        "a staged record was reported as indexed"
    );
    // A flush writes one batch; enough of them drain every staged record whatever the batch size.
    for _ in 0..8 {
        e.flush().expect("flush");
    }
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(
            e.actual_bucket(*id).expect("lookup"),
            Some(i as u32),
            "m{i} is not in the bucket its axis selects"
        );
    }
}
