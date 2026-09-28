//! Byte-level audit of Mnemos-M: every changed byte must be explained by an event emitted during
//! the operation that changed it, and every event must correspond to a real change.
//!
//! The same methodology as `crates/mnemos-v2/tests/audit.rs`, applied to the properties that are
//! specific to this design: that a flush happens and writes the same bytes whether or not anything
//! was staged, that a search touches no disk at all, and that placement carries no content. Each
//! check has a negative control that fails when its mechanism is ablated away.

// The audit's arithmetic uses the paper's symbols -- R, C, w, g, r, s for regions, capacity, epoch
// writes, slot, region, slot-in-region -- and renaming them to satisfy the lint would make each
// check harder to compare against the rule it tests.
#![allow(clippy::many_single_char_names)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use mnemos_core::event::{LeakEvent, RecordingSink};
use mnemos_core::traits::SearchableMemory;
use mnemos_core::types::{Embedding, QuantVec};
use mnemos_core::Params;
use mnemos_keystore::software::SoftwareKeystore;
use mnemos_mem::{Ablation, EngineM};

const DIM: usize = 8;

fn params() -> Params {
    let mut p = Params::discover().expect("params.toml");
    p.index.buckets = 8;
    p.index.bucket_capacity = 6;
    p.index.top_k = 3;
    p.embedding.dim = DIM;
    p.crypto.block_size_bytes = 128;
    p.epoch.epoch_writes = 4;
    p.memory.arena_bytes = 64 * 1024;
    p.m.durable = false;
    p.m.tombstone_capacity = 8;
    p.m.flush_ms = 3_600_000;
    p
}

fn unit(i: usize) -> QuantVec {
    let mut v = vec![0.02f32; DIM];
    v[i % DIM] = 1.0;
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    QuantVec::quantise(&Embedding(v.into_iter().map(|x| x / n).collect()))
}

fn store(dir: &Path, abl: Ablation) -> (EngineM, Arc<RecordingSink>) {
    store_with(dir, abl, params())
}

fn store_with(dir: &Path, abl: Ablation, p: Params) -> (EngineM, Arc<RecordingSink>) {
    let ks = SoftwareKeystore::open_at(&dir.join("k"), true).unwrap();
    let cents = (0..p.index.buckets as usize).map(unit).collect();
    let mut e = EngineM::create(dir, p, Box::new(ks), cents, abl).unwrap();
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

fn changed_regions(e: &EngineM, a: &Snapshot, b: &Snapshot) -> BTreeMap<u32, BTreeSet<u32>> {
    let g = e.geometry();
    let key = PathBuf::from("slab.bin");
    let (x, y) = (&a[&key], &b[&key]);
    assert_eq!(x.len(), y.len(), "slab.bin changed size");
    let mut out = BTreeMap::new();
    for r in 0..g.regions {
        for s in 0..g.capacity {
            let off = g.region_offset(r) as usize + s as usize * g.slot_len();
            if x[off..off + g.slot_len()] != y[off..off + g.slot_len()] {
                out.entry(r).or_insert_with(BTreeSet::new).insert(s);
            }
        }
    }
    out
}

fn region_writes(events: &[LeakEvent]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|ev| match ev {
            LeakEvent::RegionWrite { region, .. } => Some(*region),
            _ => None,
        })
        .collect()
}

fn assert_explained(e: &EngineM, before: &Snapshot, after: &Snapshot, events: &[LeakEvent]) {
    let cap = e.geometry().capacity;
    let changed = changed_regions(e, before, after);
    let mut announced: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for r in region_writes(events) {
        announced.entry(r).or_default().extend(0..cap);
    }
    for (r, slots) in &changed {
        let ok = announced.get(r).is_some_and(|a| slots.is_subset(a));
        assert!(
            ok,
            "region {r} slots {slots:?} changed with no event; announced {announced:?}"
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
            f == "slab.bin" || f == "manifest.enc",
            "unmodelled file changed: {f}"
        );
    }
}

#[test]
fn a_flush_writes_w_plus_one_whole_regions_whether_or_not_anything_was_staged() {
    for real in [0usize, 1, 3] {
        let dir = tempfile::tempdir().unwrap();
        let (mut e, sink) = store(dir.path(), Ablation::default());
        for i in 0..4 {
            e.put(&format!("warm {i}"), "{}", &unit(i)).unwrap();
        }
        e.flush().unwrap();
        let _ = sink.drain();

        for i in 0..real {
            e.put(&format!("r{i}"), "{}", &unit(i + 2)).unwrap();
        }
        let before = snapshot(dir.path());
        e.flush().unwrap();
        let after = snapshot(dir.path());
        let events = sink.drain();

        assert_eq!(
            region_writes(&events).len(),
            5,
            "flush carrying {real} real records wrote {} regions",
            region_writes(&events).len()
        );
        assert_explained(&e, &before, &after, &events);
    }
}

#[test]
fn bytes_written_per_flush_do_not_depend_on_the_number_of_real_records() {
    let mut totals = Vec::new();
    for real in [0usize, 2, 4] {
        let dir = tempfile::tempdir().unwrap();
        let (mut e, _sink) = store(dir.path(), Ablation::default());
        let s0 = e.stats();
        for i in 0..real {
            e.put(&format!("r{i}"), "{}", &unit(i)).unwrap();
        }
        e.flush().unwrap();
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
fn a_search_reads_nothing_from_disk_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path(), Ablation::default());
    for i in 0..8 {
        e.put(&format!("m{i}"), "{}", &unit(i)).unwrap();
    }
    e.flush().unwrap();
    let _ = sink.drain();

    let before = snapshot(dir.path());
    for i in 0..6 {
        let hits = e.search(&unit(i)).unwrap();
        assert!(!hits.is_empty());
    }
    let after = snapshot(dir.path());
    let events = sink.drain();

    assert!(
        changed_files(&before, &after).is_empty(),
        "a search changed the store"
    );
    assert!(
        events.is_empty(),
        "a search must be invisible at the storage layer, but emitted {events:?}"
    );
}

#[test]
fn an_ivf_search_reads_nothing_from_disk_and_writes_nothing() {
    // The in-memory partition index must not change the one property that makes Mnemos-M's
    // search channel absent rather than mitigated. Probing happens against the resident table.
    let dir = tempfile::tempdir().unwrap();
    let mut p = params();
    p.m.search = "ivf".into();
    p.m.probe_width = 2;
    let (mut e, sink) = store_with(dir.path(), Ablation::default(), p);
    for i in 0..8 {
        e.put(&format!("m{i}"), "{}", &unit(i)).unwrap();
    }
    while e.staged_len() > 0 {
        e.flush().unwrap();
    }
    let _ = sink.drain();

    let before = snapshot(dir.path());
    for i in 0..6 {
        assert!(!e.search(&unit(i)).unwrap().is_empty());
    }
    let after = snapshot(dir.path());
    let events = sink.drain();
    assert!(
        changed_files(&before, &after).is_empty(),
        "an ivf search changed the store"
    );
    assert!(events.is_empty(), "an ivf search emitted {events:?}");
}

#[test]
fn disk_search_ablation_reopens_the_result_channel() {
    let dir = tempfile::tempdir().unwrap();
    let abl = Ablation {
        disk_search: true,
        ..Ablation::default()
    };
    let (mut e, sink) = store(dir.path(), abl);
    for i in 0..8 {
        e.put(&format!("m{i}"), "{}", &unit(i)).unwrap();
    }
    e.flush().unwrap();
    let _ = sink.drain();
    e.search(&unit(0)).unwrap();
    let events = sink.drain();
    assert!(
        events
            .iter()
            .any(|ev| matches!(ev, LeakEvent::RegionSlotRead { .. })),
        "negative control: the ablated search must produce result-dependent reads"
    );
}

#[test]
fn delete_and_update_write_nothing_until_the_next_flush() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path(), Ablation::default());
    let ids: Vec<_> = (0..4)
        .map(|i| e.put(&format!("h{i}"), "{}", &unit(i)).unwrap())
        .collect();
    e.flush().unwrap();
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
fn file_sizes_do_not_depend_on_content() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let (mut a, _) = store(dir_a.path(), Ablation::default());
    let (mut b, _) = store(dir_b.path(), Ablation::default());
    a.flush().unwrap();
    for i in 0..24 {
        b.put(&format!("entry {i}"), "{\"k\":1}", &unit(i)).unwrap();
        b.flush().unwrap();
    }
    let (sa, sb) = (snapshot(dir_a.path()), snapshot(dir_b.path()));
    for k in ["slab.bin", "manifest.enc", "table.enc"] {
        assert_eq!(
            sa[Path::new(k)].len(),
            sb[Path::new(k)].len(),
            "{k} size depends on content"
        );
    }
}

/// The distribution of regions a flush writes must not shift when every record has the same
/// content. This is the property that makes the write channel empty, stated as a test.
#[test]
fn the_regions_a_flush_writes_are_uniform_under_identical_content() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path(), Ablation::default());
    // The store holds 8 * 6 = 48 slots; stay under it so the run measures placement rather than
    // the global capacity bound.
    let mut counts = [0u32; 8];
    for i in 0..40 {
        e.put(&format!("same {i}"), "{}", &unit(0)).unwrap();
        e.flush().unwrap();
        for r in region_writes(&sink.drain()) {
            counts[r as usize] += 1;
        }
    }
    let total: u32 = counts.iter().sum();
    let expected = f64::from(total) / 8.0;
    // A content-placed design would drive one region's share to roughly 1/(w+1) of every flush,
    // which at w = 4 is 0.2 against a uniform 0.125. Three times the uniform share is a wide
    // margin that still catches that.
    for (r, c) in counts.iter().enumerate() {
        assert!(
            f64::from(*c) < expected * 3.0,
            "region {r} took {c} of {total} writes against a uniform {expected:.1}; \
             placement is tracking content"
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
    for _ in 0..e.geometry().regions {
        e.flush().unwrap();
    }
    assert_eq!(e.tombstones_len(), 0);
    assert_eq!(e.stats().purged, 1);
}

#[test]
fn unlock_reads_every_region_once_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink) = store(dir.path(), Ablation::default());
    for i in 0..6 {
        e.put(&format!("m{i}"), "{}", &unit(i)).unwrap();
    }
    e.close().unwrap();
    let _ = sink.drain();

    let p = params();
    let ks = SoftwareKeystore::open_at(&dir.path().join("k"), true).unwrap();
    let mut e = EngineM::open(dir.path(), p, Box::new(ks), Ablation::default()).unwrap();
    let sink = Arc::new(RecordingSink::new());
    e.set_event_sink(sink.clone());

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
        e.geometry().regions as usize,
        "the unlock scan reads every region once"
    );
    assert!(e
        .search(&unit(3))
        .unwrap()
        .iter()
        .any(|h| h.memory.text == "m3"));
}
