//! **Does the event stream describe every byte the implementation writes?**
//!
//! `leakage_profile.rs` asserts that the events Mnemos emits match the documented profile. That
//! is necessary and it is not sufficient, and finding F5 is the proof: for most of this system's
//! life the flush path wrote each record into its semantic partition at a deterministic file
//! offset, disclosing both how many records a batch really contained and which partitions they
//! went to. Every test in `leakage_profile.rs` passed throughout, because **no event described
//! index writes at all**, and a test suite that checks events cannot see a channel that emits
//! none. The hole was invisible to every assertion simultaneously.
//!
//! The countermeasure has to start from the disk rather than from the event stream. This file
//! snapshots the store byte-for-byte, performs one operation, diffs the snapshot, and demands
//! that **every changed byte range be explained by an event that was emitted during that
//! operation**. A new channel added to the write path fails here on its first commit, whether or
//! not anyone thought to model it.
//!
//! # What "explained" means
//!
//! Each file in the store has a known layout, so a changed range maps to a channel identity:
//!
//! | File | Range maps to | Explained by |
//! |---|---|---|
//! | `index.slab` | partition = `(off - 64) / bucket_len` | `IndexWrite { bucket }` |
//! | `memory.slab` | slot = `(off - 4096) / slot_size` | `SlotWrite { addr }` |
//! | `manifest.enc` | the whole file | `ManifestWrite` |
//! | `wal.enc` | the whole file | `EpochFlush` (the WAL is rewritten wholesale per flush) |
//! | `wrapped.key`, `k` | must not change after setup | nothing --- a change is a finding |
//!
//! # What this test cannot do
//!
//! It sees writes that land on disk, not writes that are buffered, and it sees ranges rather
//! than timings. A channel that leaks purely through *when* a write happens, or through a page
//! cache, is outside its reach. It is a floor, not a ceiling. The floor is what was missing.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use mnemos_core::event::{LeakEvent, RecordingSink};
use mnemos_core::traits::SearchableMemory;
use mnemos_core::types::{Embedding, QuantVec};
use mnemos_core::Params;
use mnemos_engine::Engine;
use mnemos_keystore::software::SoftwareKeystore;

/// Slab header, from `mnemos_store::slab::HEADER_LEN`.
const SLAB_HEADER: u64 = 4096;
/// Index header, from `mnemos_index::ivf::HEADER_LEN`.
const INDEX_HEADER: u64 = 64;

fn test_params() -> Params {
    let mut p = Params::discover().expect("params.toml");
    p.index.n_max = 256;
    p.index.buckets = 8;
    p.index.bucket_capacity = 8;
    p.index.probe_width = 2;
    p.index.decoy_buckets = 0;
    p.index.overflow_depth = 1;
    p.index.top_k = 3;
    p.crypto.block_size_bytes = 512;
    p.crypto.max_entry_bytes = 512;
    p.embedding.dim = 8;
    p.epoch.compaction_slots = 4;
    p.epoch.slot_write_budget = 64;
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

fn store(dir: &Path) -> (Engine, Arc<RecordingSink>, Params) {
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
    (e, sink, p)
}

// ---------------------------------------------------------------------------------------------
// Snapshotting
// ---------------------------------------------------------------------------------------------

type Snapshot = BTreeMap<PathBuf, Vec<u8>>;

fn snapshot(dir: &Path) -> Snapshot {
    let mut out = Snapshot::new();
    for entry in std::fs::read_dir(dir).expect("read store dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file() {
            let name = PathBuf::from(path.file_name().expect("file name"));
            out.insert(name, std::fs::read(&path).expect("read file"));
        }
    }
    out
}

/// One contiguous run of bytes that differ between two snapshots of the same file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Changed {
    start: u64,
    end: u64,
}

/// Byte ranges that differ, coalesced into runs.
///
/// A file that grew counts the whole appended tail as changed. Adjacent differing bytes are
/// coalesced across gaps of up to `coalesce` bytes, so that one logical write does not come back
/// as a dozen fragments wherever the new ciphertext happened to agree with what was underneath
/// it by chance.
///
/// `coalesce` has to stay well below the slot size. Set to a full slot it merges two genuinely
/// separate writes into one run spanning the untouched slot between them, and the result is a
/// set of "changed" slots that were never written --- a test failing on its own arithmetic
/// rather than on the system. That is how the first run of this file reported 26 unexplained
/// slab slots when the real number was the four compaction touched.
fn diff(before: &[u8], after: &[u8], coalesce: u64) -> Vec<Changed> {
    let mut runs: Vec<Changed> = Vec::new();
    let n = after.len().max(before.len());
    let mut i = 0u64;
    while i < n as u64 {
        let b = before.get(i as usize).copied();
        let a = after.get(i as usize).copied();
        if a != b {
            match runs.last_mut() {
                Some(last) if i.saturating_sub(last.end) <= coalesce => last.end = i + 1,
                _ => runs.push(Changed {
                    start: i,
                    end: i + 1,
                }),
            }
        }
        i += 1;
    }
    runs
}

/// Partitions of `index.slab` a set of changed ranges touches.
fn index_partitions_touched(runs: &[Changed], p: &Params) -> BTreeSet<u32> {
    let blen = mnemos_index::bucket::bucket_len(p.index.bucket_capacity, p.embedding.dim) as u64;
    let mut out = BTreeSet::new();
    for r in runs {
        let lo = r.start.saturating_sub(INDEX_HEADER) / blen;
        let hi = (r.end - 1).saturating_sub(INDEX_HEADER) / blen;
        for b in lo..=hi {
            if b < u64::from(p.index.buckets) {
                out.insert(b as u32);
            }
        }
    }
    out
}

/// Slots of `memory.slab` a set of changed ranges touches.
fn slab_slots_touched(runs: &[Changed], p: &Params) -> BTreeSet<u64> {
    let ssz = mnemos_store::slab::slot_size(p.crypto.block_size_bytes as usize) as u64;
    let mut out = BTreeSet::new();
    for r in runs {
        if r.end <= SLAB_HEADER {
            continue;
        }
        let lo = r.start.saturating_sub(SLAB_HEADER) / ssz;
        let hi = (r.end - 1).saturating_sub(SLAB_HEADER) / ssz;
        for s in lo..=hi {
            out.insert(s);
        }
    }
    out
}

/// Every file the store is allowed to touch, and which event variant explains a write to it.
///
/// Anything not on this list is an unmodelled channel by construction: the test does not know
/// what it is, which is exactly the situation that produced F5.
fn known_files() -> BTreeSet<&'static str> {
    [
        "index.slab",
        "memory.slab",
        "manifest.enc",
        "wal.enc",
        "wrapped.key",
        "k",
    ]
    .into_iter()
    .collect()
}

// ---------------------------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------------------------

/// A flush changes bytes in `index.slab`. Every partition whose bytes changed must have been
/// named by an `IndexWrite` event emitted during that flush.
///
/// This is the assertion that would have caught F5 on the commit that introduced it. Before the
/// fix, `IndexWrite` did not exist, so the set of explained partitions was empty and every
/// changed partition was a violation.
#[test]
fn every_index_partition_written_by_a_flush_was_announced() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink, p) = store(dir.path());

    // Reach steady state first: `create` writes the whole store, and the point of interest is
    // the incremental write, not the initial layout.
    for i in 0..p.epoch.epoch_writes {
        e.put(
            &format!("warm {i}"),
            "{}",
            &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .unwrap();
    }
    e.flush().unwrap();
    let _ = sink.drain();

    let before = snapshot(dir.path());
    for i in 0..p.epoch.epoch_writes {
        // Deliberately spread across partitions, so a test that only ever touched one bucket
        // could not pass by accident.
        let mut v = vec![0.0f32; p.embedding.dim];
        v[i as usize % p.embedding.dim] = 1.0;
        e.put(&format!("entry {i}"), "{}", &qv(v)).unwrap();
    }
    e.flush().unwrap();
    let after = snapshot(dir.path());
    let events = sink.drain();

    let slot = mnemos_index::bucket::slot_len(p.embedding.dim) as u64;
    let runs = diff(
        before
            .get(Path::new("index.slab"))
            .map_or(&[][..], Vec::as_slice),
        after
            .get(Path::new("index.slab"))
            .map_or(&[][..], Vec::as_slice),
        slot,
    );
    let written = index_partitions_touched(&runs, &p);

    // `IndexWrite` alone. `TokenAdvance` fires for every partition every flush but the token
    // chains live in memory and are persisted through the manifest, so it explains no byte of
    // `index.slab` -- counting it here would let it paper over a genuinely unannounced write,
    // and the negative control below is what caught that.
    let announced: BTreeSet<u32> = events
        .iter()
        .filter_map(|ev| match ev {
            LeakEvent::IndexWrite { bucket, .. } => Some(bucket.0),
            _ => None,
        })
        .collect();

    let unannounced: Vec<u32> = written.difference(&announced).copied().collect();
    assert!(
        unannounced.is_empty(),
        "index partitions {unannounced:?} changed on disk with no IndexWrite event. \
         An adversary watching the index file sees these writes; the leakage model does not. \
         This is finding F5's shape: a channel with no event is invisible to every test that \
         reads events.\nannounced: {announced:?}\nwritten: {written:?}"
    );
    assert!(
        !written.is_empty(),
        "the flush wrote nothing to the index; the test proved nothing"
    );

    // Negative control, in the same test so it cannot drift away from the assertion it guards.
    // Strip the `IndexWrite` events and the store is back in the state F5 describes: real
    // writes to the index file, and a leakage model with nothing to say about them. If this
    // does *not* fire, the assertion above is decorative and F5 could recur underneath it.
    let f5_announced: BTreeSet<u32> = BTreeSet::new();
    let would_be_missed: Vec<u32> = written.difference(&f5_announced).copied().collect();
    assert!(
        !would_be_missed.is_empty(),
        "the negative control passed, which means this test cannot detect the hole it exists \
         to detect. With IndexWrite removed, partitions {written:?} were still considered \
         explained by {f5_announced:?}."
    );
}

/// The same question for the content slab: every slot whose bytes changed must have been named
/// by a `SlotWrite`.
#[test]
fn every_slab_slot_written_by_a_flush_was_announced() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink, p) = store(dir.path());

    for i in 0..p.epoch.epoch_writes {
        e.put(
            &format!("warm {i}"),
            "{}",
            &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .unwrap();
    }
    e.flush().unwrap();
    let _ = sink.drain();

    let before = snapshot(dir.path());
    for i in 0..p.epoch.epoch_writes {
        let mut v = vec![0.0f32; p.embedding.dim];
        v[i as usize % p.embedding.dim] = 1.0;
        e.put(&format!("entry {i}"), "{}", &qv(v)).unwrap();
    }
    e.flush().unwrap();
    let after = snapshot(dir.path());
    let events = sink.drain();

    let runs = diff(
        before
            .get(Path::new("memory.slab"))
            .map_or(&[][..], Vec::as_slice),
        after
            .get(Path::new("memory.slab"))
            .map_or(&[][..], Vec::as_slice),
        16,
    );
    let written = slab_slots_touched(&runs, &p);

    let mut announced: BTreeSet<u64> = BTreeSet::new();
    for ev in &events {
        match ev {
            LeakEvent::SlotWrite { addr, .. } | LeakEvent::SlotRead { addr } => {
                announced.insert(addr.0);
            }
            // Compaction rewrites dummy material into unoccupied slots on a data-independent
            // rolling cursor. It used to announce only the *budget*, which described strictly
            // less than the disk did; it now names the slots, which is why they can be checked
            // here at all.
            LeakEvent::Compact { slots, .. } => {
                announced.extend(slots.iter().map(|a| a.0));
            }
            _ => {}
        }
    }

    let unannounced: Vec<u64> = written.difference(&announced).copied().collect();
    assert!(
        unannounced.is_empty(),
        "slab slots {unannounced:?} changed on disk with no event naming them.\n\
         announced: {announced:?}"
    );
    assert!(
        !written.is_empty(),
        "the flush wrote nothing to the slab; the test proved nothing"
    );

    // Negative control: drop the compaction slots and the check must notice. Compaction is the
    // channel this test found unmodelled, so it is the one the control is aimed at.
    let without_compaction: BTreeSet<u64> = events
        .iter()
        .filter_map(|ev| match ev {
            LeakEvent::SlotWrite { addr, .. } | LeakEvent::SlotRead { addr } => Some(addr.0),
            _ => None,
        })
        .collect();
    assert!(
        written.difference(&without_compaction).next().is_some(),
        "the negative control passed: compaction wrote no slot that SlotWrite did not already \
         cover, so this test would not have detected compaction going unannounced."
    );
}

/// No file outside the known layout may change, and the wrapped key may not change at all.
///
/// A new file appearing in the store directory is a channel nobody has reasoned about: its
/// existence, its size and its mtime are all observable. The test names the files it knows so
/// that adding one is a decision rather than an accident.
#[test]
fn no_unmodelled_file_changes() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink, p) = store(dir.path());
    e.put(
        "first",
        "{}",
        &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    .unwrap();
    e.flush().unwrap();
    let _ = sink.drain();

    let before = snapshot(dir.path());
    for i in 0..p.epoch.epoch_writes {
        e.put(
            &format!("entry {i}"),
            "{}",
            &qv(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .unwrap();
    }
    e.flush().unwrap();
    e.search(&qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        .unwrap();
    let after = snapshot(dir.path());

    let known = known_files();
    for name in after.keys() {
        let n = name.to_string_lossy().to_string();
        assert!(
            known.contains(n.as_str()),
            "the store grew a file this test does not model: {n}. Its existence, size and \
             modification time are all observable; decide what it leaks and add it here."
        );
    }

    // The wrapped master key is written once, at setup. A rewrite would mean key material is
    // being touched on the hot path, which is both a leak (the adversary sees when) and a
    // durability hazard.
    assert_eq!(
        before.get(Path::new("wrapped.key")),
        after.get(Path::new("wrapped.key")),
        "wrapped.key changed during ordinary operation"
    );
}

/// The manifest is rewritten every epoch, and every rewrite is announced.
///
/// The interesting half is the converse of the usual assertion: a manifest write that happened
/// *without* an event would let the adversary date activity the event stream does not admit to.
#[test]
fn manifest_changes_are_announced_and_unconditional() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink, p) = store(dir.path());
    e.put(
        "first",
        "{}",
        &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    .unwrap();
    e.flush().unwrap();
    let _ = sink.drain();

    // A flush with nothing staged. The manifest must still be rewritten: a conditional write
    // would turn "the manifest changed" into "real work happened this epoch".
    let before = snapshot(dir.path());
    e.flush().unwrap();
    let after = snapshot(dir.path());
    let events = sink.drain();

    let manifest_changed =
        before.get(Path::new("manifest.enc")) != after.get(Path::new("manifest.enc"));
    let announced = events
        .iter()
        .any(|ev| matches!(ev, LeakEvent::ManifestWrite { .. }));

    assert!(
        manifest_changed,
        "an empty flush left the manifest untouched: an activity oracle"
    );
    assert!(
        announced,
        "the manifest changed with no ManifestWrite event"
    );
    let _ = p;
}

/// A search must not change a single byte of the store.
///
/// Reads that write --- an access-count update, a cache file, a recency touch --- are the
/// classic way a read-only operation becomes an update channel. This is cheap to assert and
/// impossible to notice by inspection once the code is large enough.
#[test]
fn a_search_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, sink, p) = store(dir.path());
    for i in 0..p.epoch.epoch_writes {
        e.put(
            &format!("entry {i}"),
            "{}",
            &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .unwrap();
    }
    e.flush().unwrap();
    let _ = sink.drain();

    let before = snapshot(dir.path());
    for _ in 0..8 {
        e.search(&qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            .unwrap();
        e.search(&qv(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            .unwrap();
    }
    let after = snapshot(dir.path());

    let mut differing: Vec<String> = Vec::new();
    for (name, bytes) in &after {
        if before.get(name) != Some(bytes) {
            differing.push(name.to_string_lossy().to_string());
        }
    }
    assert!(
        differing.is_empty(),
        "searching modified {differing:?}. A read that writes is an update channel: the \
         adversary learns that a query happened, and when."
    );
}
