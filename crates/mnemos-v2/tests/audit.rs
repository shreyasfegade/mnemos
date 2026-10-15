//! Byte-level audit of Mnemos v2: for every operation, diff the store directory and require that
//! every changed byte is explained by an event emitted during that operation, and that the
//! events describe nothing that did not happen.
//!
//! v1's channel enumeration covered flush, manifest and search but not delete or update, and the
//! delete path turned out to rewrite a partition region with no event. This file covers every
//! public operation, and each check has a negative control that must fail when the mechanism is
//! removed through an evaluation ablation.

// The audit's arithmetic uses the paper's own symbols -- B, C, w, g, r, s for buckets, capacity,
// epoch writes, slot, region, slot-in-region -- and renaming them to satisfy the lint would make
// each check harder to compare against the definition it tests.
#![allow(clippy::many_single_char_names)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use mnemos_core::event::{LeakEvent, RecordingSink};
use mnemos_core::traits::SearchableMemory;
use mnemos_core::types::{Embedding, QuantVec};
use mnemos_core::Params;
use mnemos_keystore::software::SoftwareKeystore;
use mnemos_v2::{Ablation, EngineV2};

const DIM: usize = 8;

fn params() -> Params {
    let mut p = Params::discover().expect("params.toml");
    p.index.buckets = 8;
    p.index.bucket_capacity = 6;
    p.index.probe_width = 2;
    p.index.probe_spread = 2;
    p.index.top_k = 3;
    p.embedding.dim = DIM;
    p.crypto.block_size_bytes = 128;
    p.epoch.epoch_writes = 4;
    p.epoch.epoch_ms = 3_600_000;
    p.memory.arena_bytes = 64 * 1024;
    p.v2.durable = false;
    p.v2.tombstone_capacity = 8;
    p
}

fn unit(i: usize) -> QuantVec {
    let mut v = vec![0.02f32; DIM];
    v[i % DIM] = 1.0;
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    QuantVec::quantise(&Embedding(v.into_iter().map(|x| x / n).collect()))
}

fn store(dir: &Path, abl: Ablation) -> (EngineV2, Arc<RecordingSink>) {
    store_with(dir, abl, params())
}

fn store_with(dir: &Path, abl: Ablation, p: Params) -> (EngineV2, Arc<RecordingSink>) {
    let ks = SoftwareKeystore::open_at(&dir.join("k"), true).unwrap();
    let cents = (0..p.index.buckets as usize).map(unit).collect();
    let mut e = EngineV2::create(dir, p, Box::new(ks), cents, abl).unwrap();
    let sink = Arc::new(RecordingSink::new());
    e.set_event_sink(sink.clone());
    (e, sink)
}

type Snapshot = BTreeMap<PathBuf, Vec<u8>>;

fn snapshot(root: &Path) -> Snapshot {
    fn walk(root: &Path, dir: &Path, out: &mut Snapshot) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                out.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    std::fs::read(&path).unwrap(),
                );
            }
        }
    }
    let mut out = Snapshot::new();
    walk(root, root, &mut out);
    out
}

fn changed_files(a: &Snapshot, b: &Snapshot) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for k in a.keys().chain(b.keys()) {
        if a.get(k) != b.get(k) {
            out.insert(k.to_string_lossy().replace('\\', "/"));
        }
    }
    out
}

/// Regions of `partitions.bin` whose bytes changed, and for each the slots that changed.
fn changed_regions(e: &EngineV2, a: &Snapshot, b: &Snapshot) -> BTreeMap<u32, BTreeSet<u32>> {
    let g = e.geometry();
    let key = PathBuf::from("partitions.bin");
    let (x, y) = (&a[&key], &b[&key]);
    assert_eq!(x.len(), y.len(), "partitions.bin changed size");
    let mut out = BTreeMap::new();
    for r in 0..g.buckets {
        for s in 0..g.capacity {
            let off = g.region_offset(r) as usize + s as usize * g.slot_len();
            if x[off..off + g.slot_len()] != y[off..off + g.slot_len()] {
                out.entry(r).or_insert_with(BTreeSet::new).insert(s);
            }
        }
    }
    out
}

fn region_writes(events: &[LeakEvent]) -> Vec<(u32, Vec<u32>)> {
    events
        .iter()
        .filter_map(|ev| match ev {
            LeakEvent::RegionWrite { region, slots, .. } => Some((*region, slots.clone())),
            _ => None,
        })
        .collect()
}

/// Every changed slot is inside an announced write, and every announced whole-region write
/// changed every slot of its region.
fn assert_explained(e: &EngineV2, before: &Snapshot, after: &Snapshot, events: &[LeakEvent]) {
    let cap = e.geometry().capacity;
    let changed = changed_regions(e, before, after);
    let mut announced: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for (r, slots) in region_writes(events) {
        let set = announced.entry(r).or_default();
        if slots.is_empty() {
            set.extend(0..cap);
        } else {
            set.extend(slots);
        }
    }
    for (r, slots) in &changed {
        let ok = announced.get(r).is_some_and(|a| slots.is_subset(a));
        assert!(
            ok,
            "region {r} slots {slots:?} changed without an event; announced {announced:?}"
        );
    }
    for (r, slots) in &announced {
        assert_eq!(
            changed.get(r),
            Some(slots),
            "announced write to region {r} did not change exactly those slots"
        );
    }
    let files = changed_files(before, after);
    let manifest_announced = events
        .iter()
        .any(|ev| matches!(ev, LeakEvent::ManifestWrite { .. }));
    assert_eq!(
        files.contains("manifest.enc"),
        manifest_announced,
        "manifest change and event disagree"
    );
    for f in &files {
        assert!(
            f == "partitions.bin" || f == "manifest.enc",
            "unmodelled file changed: {f}"
        );
    }
}

fn warm(e: &mut EngineV2, n: usize) {
    for i in 0..n {
        e.put(&format!("warm {i}"), "{}", &unit(i)).unwrap();
    }
    e.flush().unwrap();
}

#[test]
fn a_flush_writes_exactly_w_plus_one_whole_regions_whatever_it_carries() {
    for real in [0usize, 1, 3] {
        let dir = tempfile::tempdir().unwrap();
        let (mut e, sink) = store(dir.path(), Ablation::default());
        warm(&mut e, 4);
        let _ = sink.drain();

        for i in 0..real {
            e.put(&format!("r{i}"), "{}", &unit(i + 2)).unwrap();
        }
        let before = snapshot(dir.path());
        e.flush().unwrap();
        let after = snapshot(dir.path());
        let events = sink.drain();

        let writes = region_writes(&events);
        assert_eq!(
            writes.len(),
            4 + 1,
            "flush with {real} real records wrote {} regions",
            writes.len()
        );
        assert!(
            writes.iter().all(|(_, s)| s.is_empty()),
            "a partial region write in the shipped engine"
        );
        assert_explained(&e, &before, &after, &events);
        assert_eq!(
            events
                .iter()
                .filter(|ev| matches!(ev, LeakEvent::EpochFlush { .. }))
                .count(),
            1
        );
    }
}

#[test]
fn bytes_written_per_flush_do_not_depend_on_the_number_of_real_records() {
    let mut totals = Vec::new();
    for real in [0usize, 2, 4] {
        let dir = tempfile::tempdir().unwrap();
        let (mut e, _sink) = store(dir.path(), Ablation::default());
        warm(&mut e, 2);
        let s0 = e.stats();
        for i in 0..real {
            e.put(&format!("r{i}"), "{}", &unit(i)).unwrap();
        }
        if real < 4 {
            e.flush().unwrap();
        }
        let s1 = e.stats();
        totals.push((
            s1.region_bytes_written - s0.region_bytes_written,
            s1.region_writes - s0.region_writes,
        ));
    }
    assert!(
        totals.windows(2).all(|w| w[0] == w[1]),
        "flush volume depends on content: {totals:?}"
    );
}

#[test]
fn put_below_the_batch_size_writes_nothing_and_emits_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path(), Ablation::default());
    warm(&mut e, 1);
    let _ = sink.drain();
    let before = snapshot(dir.path());
    e.put("staged", "{}", &unit(1)).unwrap();
    let after = snapshot(dir.path());
    assert!(changed_files(&before, &after).is_empty());
    assert!(sink.drain().is_empty());
}

#[test]
fn a_search_reads_whole_regions_only_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path(), Ablation::default());
    warm(&mut e, 4);
    let _ = sink.drain();
    let before = snapshot(dir.path());
    for i in 0..6 {
        let hits = e.search(&unit(i)).unwrap();
        assert!(hits.len() <= 3);
    }
    let after = snapshot(dir.path());
    let events = sink.drain();
    assert!(
        changed_files(&before, &after).is_empty(),
        "a search changed the store"
    );
    let reads = events
        .iter()
        .filter(|ev| matches!(ev, LeakEvent::RegionRead { .. }))
        .count();
    assert_eq!(reads, 6 * e.geometry().regions_per_search() as usize);
    assert!(
        events
            .iter()
            .all(|ev| matches!(ev, LeakEvent::RegionRead { .. })),
        "a search emitted something other than whole-region reads: {events:?}"
    );
}

#[test]
fn result_reads_ablation_is_detected() {
    let dir = tempfile::tempdir().unwrap();
    let abl = Ablation {
        result_reads: true,
        ..Ablation::default()
    };
    let (mut e, sink) = store(dir.path(), abl);
    warm(&mut e, 4);
    let _ = sink.drain();
    e.search(&unit(0)).unwrap();
    assert!(sink
        .drain()
        .iter()
        .any(|ev| matches!(ev, LeakEvent::RegionSlotRead { .. })));
}

#[test]
fn delete_and_update_write_nothing_until_the_next_flush() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path(), Ablation::default());
    let ids: Vec<_> = (0..4)
        .map(|i| e.put(&format!("h{i}"), "{}", &unit(i)).unwrap())
        .collect();
    let _ = sink.drain();

    let before = snapshot(dir.path());
    e.delete(ids[0]).unwrap();
    e.update(ids[1], "h1 revised", "{}", &unit(1)).unwrap();
    let after = snapshot(dir.path());
    assert!(
        changed_files(&before, &after).is_empty(),
        "delete or update wrote immediately"
    );
    assert!(sink.drain().is_empty(), "delete or update emitted an event");

    let before = snapshot(dir.path());
    e.flush().unwrap();
    let after = snapshot(dir.path());
    let events = sink.drain();
    assert_eq!(region_writes(&events).len(), 5);
    assert_explained(&e, &before, &after, &events);
}

#[test]
fn immediate_delete_ablation_names_the_region_at_deletion_time() {
    let dir = tempfile::tempdir().unwrap();
    let abl = Ablation {
        immediate_delete: true,
        ..Ablation::default()
    };
    let (mut e, sink) = store(dir.path(), abl);
    let ids: Vec<_> = (0..4)
        .map(|i| e.put(&format!("h{i}"), "{}", &unit(i)).unwrap())
        .collect();
    let _ = sink.drain();
    let before = snapshot(dir.path());
    e.delete(ids[2]).unwrap();
    let after = snapshot(dir.path());
    let events = sink.drain();
    assert_eq!(
        region_writes(&events).len(),
        1,
        "ablation should rewrite exactly the record's region"
    );
    assert_explained(&e, &before, &after, &events);
}

#[test]
fn slot_writes_only_ablation_writes_partial_regions_and_is_still_explained() {
    let dir = tempfile::tempdir().unwrap();
    let abl = Ablation {
        slot_writes_only: true,
        ..Ablation::default()
    };
    let (mut e, sink) = store(dir.path(), abl);
    warm(&mut e, 4);
    let _ = sink.drain();
    e.put("x", "{}", &unit(5)).unwrap();
    let before = snapshot(dir.path());
    e.flush().unwrap();
    let after = snapshot(dir.path());
    let events = sink.drain();
    let writes = region_writes(&events);
    assert_eq!(
        writes.len(),
        1,
        "only the placed slot should be written: {writes:?}"
    );
    assert_eq!(writes[0].1.len(), 1);
    assert_explained(&e, &before, &after, &events);
}

#[test]
fn lock_flushes_the_staged_remainder_and_unlock_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path(), Ablation::default());
    warm(&mut e, 4);
    e.put("pending", "{}", &unit(3)).unwrap();
    let _ = sink.drain();
    let before = snapshot(dir.path());
    e.lock();
    let after = snapshot(dir.path());
    let events = sink.drain();
    assert_eq!(
        region_writes(&events).len(),
        5,
        "lock did not flush the staged record"
    );
    assert_explained(&e, &before, &after, &events);

    let before = snapshot(dir.path());
    e.unlock().unwrap();
    let after = snapshot(dir.path());
    let events = sink.drain();
    assert!(
        changed_files(&before, &after).is_empty(),
        "unlock changed the store"
    );
    let reads = events
        .iter()
        .filter(|ev| matches!(ev, LeakEvent::RegionRead { .. }))
        .count();
    assert_eq!(
        reads,
        e.geometry().buckets as usize,
        "the unlock scan reads every region once"
    );
    assert!(e
        .search(&unit(3))
        .unwrap()
        .iter()
        .any(|h| h.memory.text == "pending"));
}

#[test]
fn file_sizes_do_not_depend_on_content() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let (mut a, _) = store(dir_a.path(), Ablation::default());
    let (mut b, _) = store(dir_b.path(), Ablation::default());
    a.flush().unwrap();
    for i in 0..24 {
        b.put(&format!("entry {i}"), "{\"k\":1}", &unit(i)).unwrap();
    }
    b.flush().unwrap();
    let (sa, sb) = (snapshot(dir_a.path()), snapshot(dir_b.path()));
    for k in ["partitions.bin", "manifest.enc", "table.enc"] {
        assert_eq!(
            sa[Path::new(k)].len(),
            sb[Path::new(k)].len(),
            "{k} size depends on content"
        );
    }
}

#[test]
fn growing_regions_ablation_reveals_occupancy_through_file_sizes() {
    let dir = tempfile::tempdir().unwrap();
    let abl = Ablation {
        growing_regions: true,
        ..Ablation::default()
    };
    let (mut e, _) = store(dir.path(), abl);
    for i in 0..3 {
        e.put(&format!("same {i}"), "{}", &unit(0)).unwrap();
    }
    e.flush().unwrap();
    let sizes: Vec<u64> = (0..8)
        .map(|r| {
            std::fs::metadata(dir.path().join(format!("regions/r{r:05}.bin")))
                .unwrap()
                .len()
        })
        .collect();
    let slot = e.geometry().slot_len() as u64;
    assert!(
        sizes.contains(&(3 * slot)),
        "negative control: occupancy should be visible, got {sizes:?}"
    );
}

/// A refused `put` must produce the bytes an accepted one would have.
///
/// Otherwise an observer who has labelled regions sees that the region it expected to receive a
/// write did not, and learns that the topic's live load has reached `C`. Since cohort P's
/// contribution to that load is already visible to it, the remainder is cohort H's load on that
/// topic, which is the asset G5 protects.
#[test]
fn a_refused_put_still_rewrites_the_region_it_wanted() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path(), Ablation::default());
    let intended = e.ground_truth_region(&unit(0)).unwrap();

    // Fill that partition to capacity (C = 6).
    for i in 0..6 {
        e.put(&format!("f{i}"), "{}", &unit(0)).unwrap();
    }
    e.flush().unwrap();
    let _ = sink.drain();

    let before = snapshot(dir.path());
    assert!(
        e.put("refused", "{}", &unit(0)).is_err(),
        "the partition should be full"
    );
    assert_eq!(e.stats().refused, 1);
    assert_eq!(
        e.stats().ghost_writes,
        1,
        "the refusal must owe its region a rewrite"
    );
    e.flush().unwrap();
    let after = snapshot(dir.path());
    let events = sink.drain();

    let writes = region_writes(&events);
    assert_eq!(
        writes.len(),
        5,
        "a refusal must not change the shape of the flush"
    );
    assert!(
        writes.iter().any(|(r, _)| *r == intended),
        "the flush after a refusal wrote {writes:?} and never touched the intended region {intended}"
    );
    assert_explained(&e, &before, &after, &events);
}

/// The negative control for the above: `refusal_mode = "silent"` owes nothing, which is the
/// behaviour the refusal channel is measured against.
#[test]
fn silent_refusal_mode_reproduces_the_channel() {
    let dir = tempfile::tempdir().unwrap();
    let mut p = params();
    p.v2.refusal_mode = "silent".into();
    let (mut e, _sink) = store_with(dir.path(), Ablation::default(), p);
    for i in 0..6 {
        e.put(&format!("f{i}"), "{}", &unit(0)).unwrap();
    }
    e.flush().unwrap();
    assert!(e.put("refused", "{}", &unit(0)).is_err());
    assert_eq!(e.stats().refused, 1);
    assert_eq!(
        e.stats().ghost_writes,
        0,
        "negative control: silent mode owes no rewrite"
    );
}

/// Closing re-encrypts the whole file, so a diff between two sessions' versions is total.
///
/// Without this, an adversary holding two pre-τ versions diffs them and reads off exactly the
/// regions written in between, which is the write channel applied before τ.
#[test]
fn closing_re_encrypts_every_region() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, _sink) = store(dir.path(), Ablation::default());
    for i in 0..4 {
        e.put(&format!("m{i}"), "{}", &unit(i)).unwrap();
    }
    e.flush().unwrap();

    let before = snapshot(dir.path());
    e.close().unwrap();
    let after = snapshot(dir.path());

    let g = params();
    let changed = changed_files(&before, &after);
    assert!(changed.contains("partitions.bin"));
    // Every region's bytes must differ, not merely the file as a whole.
    let key = PathBuf::from("partitions.bin");
    let (x, y) = (&before[&key], &after[&key]);
    let region_len = {
        let (b, c) = (g.index.buckets, g.index.bucket_capacity);
        assert!(b > 0 && c > 0);
        (x.len() - 64) / b as usize
    };
    for r in 0..g.index.buckets as usize {
        let off = 64 + r * region_len;
        assert_ne!(
            x[off..off + region_len],
            y[off..off + region_len],
            "region {r} was not re-encrypted at close"
        );
    }
}

#[test]
fn tombstoned_records_leave_the_disk_within_one_cursor_sweep() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, _) = store(dir.path(), Ablation::default());
    let id = e.put("to delete", "{}", &unit(2)).unwrap();
    e.flush().unwrap();
    e.delete(id).unwrap();
    for _ in 0..e.geometry().buckets {
        e.flush().unwrap();
    }
    assert_eq!(e.tombstones_len(), 0);
    assert_eq!(e.stats().purged, 1);
}

/// Every 16-byte chunk of every rewritten region differs from what was there before.
///
/// Slot granularity is what `assert_explained` checks. This is finer: if a whole-region rewrite
/// re-sealed live slots under fresh nonces but left, say, a free slot's bytes or a payload tail
/// untouched, a byte diff inside the region would still say which slots were live. Two sealings of
/// anything agree on a given 16-byte chunk with probability 2^-128, so one unchanged chunk is a
/// defect, not bad luck.
fn unchanged_chunks(e: &EngineV2, a: &Snapshot, b: &Snapshot, region: u32) -> usize {
    let g = e.geometry();
    let key = PathBuf::from("partitions.bin");
    let (x, y) = (&a[&key], &b[&key]);
    let off = g.region_offset(region) as usize;
    let len = g.capacity as usize * g.slot_len();
    x[off..off + len]
        .chunks(16)
        .zip(y[off..off + len].chunks(16))
        .filter(|(p, q)| p == q)
        .count()
}

#[test]
fn a_whole_region_rewrite_changes_every_byte_of_the_region() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path(), Ablation::default());
    // A mix of live, deleted and never-used slots in the regions about to be rewritten.
    let ids: Vec<_> = (0..8)
        .map(|i| e.put(&format!("live {i}"), "{}", &unit(i)).unwrap())
        .collect();
    e.flush().unwrap();
    e.delete(ids[1]).unwrap();
    e.delete(ids[4]).unwrap();
    let _ = sink.drain();
    for round in 0..6 {
        e.put(&format!("new {round}"), "{}", &unit(round)).unwrap();
        let before = snapshot(dir.path());
        e.flush().unwrap();
        let after = snapshot(dir.path());
        for (r, slots) in region_writes(&sink.drain()) {
            assert!(slots.is_empty());
            assert_eq!(
                unchanged_chunks(&e, &before, &after, r),
                0,
                "round {round}: region {r} was 'rewritten' but kept some of its bytes"
            );
        }
    }
}

#[test]
fn slot_writes_only_ablation_leaves_unchanged_bytes_in_the_region() {
    // Negative control: a per-slot write leaves the rest of its region byte-identical, which is
    // exactly what the full-churn test above must be able to see.
    let dir = tempfile::tempdir().unwrap();
    let abl = Ablation {
        slot_writes_only: true,
        ..Ablation::default()
    };
    let (mut e, sink) = store(dir.path(), abl);
    warm(&mut e, 4);
    let _ = sink.drain();
    e.put("x", "{}", &unit(5)).unwrap();
    let before = snapshot(dir.path());
    e.flush().unwrap();
    let after = snapshot(dir.path());
    let writes = region_writes(&sink.drain());
    assert!(
        unchanged_chunks(&e, &before, &after, writes[0].0) > 0,
        "negative control: a per-slot write must leave most of its region untouched"
    );
}
