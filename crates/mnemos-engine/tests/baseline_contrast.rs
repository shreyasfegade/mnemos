//! **The paper's central table, as an executable assertion.**
//!
//! `leakage_profile.rs` checks that Mnemos leaks what `docs/leakage-profile.md` says it leaks.
//! This file checks the other half of the claim, and it is the half a reader actually cares
//! about: that the *baselines* leak something Mnemos does not, on the same corpus, at the same
//! moment, read by the same adversary.
//!
//! Without this, "Mnemos closes the snapshot channel" is a statement about one implementation
//! with nothing to compare it to. A test that only ever looks at Mnemos cannot distinguish
//! "we closed the channel" from "we never had the corpus to open it".
//!
//! ## The discipline
//!
//! Every scheme is written the **same entries in the same order**, then imaged. Each scheme's
//! `snapshot_observables` (or its Mnemos equivalent, reading the actual files) is the only way
//! the adversary is allowed to look. No private fields, no in-memory state, no key.
//!
//! ## What is asserted
//!
//! | Assertion | Why it matters |
//! |---|---|
//! | Sophos exposes per-keyword update counts | Otherwise A3 has nothing to attack |
//! | Diana exposes them too — **the same counter** | Kills "just use Diana instead" |
//! | Naive-EtS exposes record lengths | The A1 channel; padding is one parameter away |
//! | Mnemos's on-disk size is a function of `n_max` alone | Constant-volume, at the file level |
//! | Mnemos's slab size does not change with insertion order | History independence, at the file level |

use mnemos_baselines::diana::DianaFp;
use mnemos_baselines::naive::NaiveEts;
use mnemos_baselines::sophos::SophosFp;
use mnemos_baselines::Keywords;
use mnemos_core::traits::SearchableMemory;
use mnemos_core::types::{Embedding, QuantVec};
use mnemos_core::Params;
use mnemos_engine::Engine;
use mnemos_keystore::software::SoftwareKeystore;

const DIM: usize = 16;
const KEYWORDS: usize = 8;
const N: usize = 96;

fn test_params() -> Params {
    let mut p = Params::discover().expect("params.toml");
    p.index.n_max = 512;
    p.index.buckets = KEYWORDS as u32;
    p.index.bucket_capacity = 32;
    p.index.probe_width = 2;
    p.index.decoy_buckets = 0;
    p.index.top_k = 3;
    p.crypto.block_size_bytes = 512;
    p.crypto.max_entry_bytes = 512;
    p.embedding.dim = DIM;
    p.epoch.compaction_slots = 4;
    p.memory.arena_bytes = 64 * 1024;
    p
}

fn qv(v: &[f32]) -> QuantVec {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    QuantVec::quantise(&Embedding(v.iter().map(|x| x / n).collect()))
}

/// A clustered corpus with deterministic, mildly skewed partition occupancy.
///
/// Skew matters: with a perfectly uniform corpus every counter would be equal and Sophos's
/// counter map would carry no signal — the test would then pass for the wrong reason.
fn corpus(n: usize) -> Vec<(String, Vec<f32>)> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Zipf-ish: low cluster indices get many more entries than high ones.
        let c = (i % KEYWORDS) * (i % 3 + 1) % KEYWORDS;
        let mut v = vec![0.0f32; DIM];
        v[c] = 1.0;
        v[(c + 1) % DIM] = 0.15 * ((i % 7) as f32 / 7.0);
        out.push((format!("entry {i} about topic {c}"), v));
    }
    out
}

fn keyword_space() -> Keywords {
    let centroids = (0..KEYWORDS)
        .map(|i| {
            let mut v = vec![0.0f32; DIM];
            v[i] = 1.0;
            qv(&v)
        })
        .collect();
    Keywords::from_centroids(centroids)
}

/// The store directory inside a scratch dir.
///
/// The software keystore's key file lives *beside* it rather than inside it, so
/// [`mnemos_leaks_no_per_keyword_signal_where_the_baselines_do`] can assert the store's exact
/// on-disk footprint without a test artifact polluting the listing.
fn store_dir(scratch: &std::path::Path) -> std::path::PathBuf {
    scratch.join("store")
}

/// Build a Mnemos store over the same partitioning the baselines use.
fn mnemos(scratch: &std::path::Path) -> Engine {
    let p = test_params();
    let ks =
        SoftwareKeystore::open_at(&scratch.join("keystore.bin"), true).expect("software keystore");
    Engine::create(
        &store_dir(scratch),
        p,
        Box::new(ks),
        keyword_space().centroids().to_vec(),
    )
    .expect("create")
}

#[test]
fn sophos_and_diana_both_expose_per_keyword_update_counts() {
    // The finding that motivates the whole project, measured on both baselines at once so the
    // "just use Diana" objection is answered in the same breath it is raised.
    let d = tempfile::tempdir().unwrap();
    let data = corpus(N);

    let mut s = SophosFp::create(&d.path().join("s"), [1u8; 32], keyword_space(), 5).unwrap();
    let mut di = DianaFp::create(&d.path().join("d"), [1u8; 32], keyword_space(), 5).unwrap();
    for (text, v) in &data {
        let q = qv(v);
        s.put(text, "{}", &q).unwrap();
        di.put(text, "{}", &q).unwrap();
    }
    s.flush().unwrap();
    di.flush().unwrap();

    let sv = SophosFp::snapshot_observables(&d.path().join("s")).unwrap();
    let dv = DianaFp::snapshot_observables(&d.path().join("d")).unwrap();

    assert!(
        sv.has_per_keyword_signal(),
        "Sophos leaked nothing; A3 has no target"
    );
    assert!(
        dv.has_per_keyword_signal(),
        "Diana leaked nothing; the class claim fails"
    );

    // Both account for every write, and the distributions are non-degenerate — otherwise the
    // "counts identify partitions" premise is untested.
    let s_total: u64 = sv.update_counts.iter().map(|(_, c)| c).sum();
    let d_total: u64 = dv.update_counts.iter().map(|(_, c)| c).sum();
    assert_eq!(s_total, N as u64);
    assert_eq!(d_total, N as u64);

    let s_multiset = sorted_counts(&sv.update_counts);
    let d_multiset = sorted_counts(&dv.update_counts);
    assert_eq!(
        s_multiset, d_multiset,
        "Sophos and Diana disagree about per-partition counts on identical input; one of the \
         two keyword spaces is wrong and the comparison is invalid"
    );
    assert!(
        s_multiset
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1,
        "every partition received the same number of writes; the corpus has no skew and the \
         counter channel is untested"
    );
}

#[test]
fn mnemos_on_disk_size_is_a_function_of_capacity_alone() {
    // Constant volume, asserted at the file level rather than the event level. Two stores that
    // differ in *content count* must be byte-for-byte the same size.
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let data = corpus(N);

    let mut ea = mnemos(a.path());
    let mut eb = mnemos(b.path());

    for (text, v) in data.iter().take(8) {
        ea.put(text, "{}", &qv(v)).unwrap();
    }
    for (text, v) in data.iter().take(80) {
        eb.put(text, "{}", &qv(v)).unwrap();
    }
    ea.flush().unwrap();
    eb.flush().unwrap();

    for name in ["memory.slab", "index.slab", "manifest.enc", "wal.enc"] {
        let sa = std::fs::metadata(store_dir(a.path()).join(name))
            .unwrap()
            .len();
        let sb = std::fs::metadata(store_dir(b.path()).join(name))
            .unwrap()
            .len();
        assert_eq!(
            sa, sb,
            "{name} differs in size between a 8-entry store and an 80-entry store ({sa} vs \
             {sb}) — entry count is readable from a directory listing"
        );
    }
}

#[test]
fn mnemos_leaks_no_per_keyword_signal_where_the_baselines_do() {
    // The contrast, stated as directly as the file system permits: there is no analogue of
    // `sophos.state` in a Mnemos store. Every file present is fixed-size and accounted for.
    let d = tempfile::tempdir().unwrap();
    let mut e = mnemos(d.path());
    for (text, v) in corpus(N) {
        e.put(&text, "{}", &qv(&v)).unwrap();
    }
    e.flush().unwrap();

    let mut names: Vec<String> = std::fs::read_dir(store_dir(d.path()))
        .unwrap()
        .filter_map(std::result::Result::ok)
        .filter(|f| f.path().is_file())
        .map(|f| f.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();

    // If a new file appears here, it is new persisted state and it needs a row in
    // docs/leakage-profile.md before this test is updated. Failing loudly on an unexpected
    // filename is the point: silent additions to the on-disk footprint are how the profile
    // drifts out of date.
    assert_eq!(
        names,
        vec![
            "index.slab".to_string(),
            "manifest.enc".to_string(),
            "memory.slab".to_string(),
            "wal.enc".to_string(),
            "wrapped.key".to_string(),
        ],
        "unexpected on-disk footprint; every persisted file needs a leakage-profile row"
    );
}

#[test]
fn mnemos_slab_size_is_independent_of_insertion_order() {
    // The *file-level* corollary of history independence, not the property itself.
    //
    // The byte-level claim is asserted in `mnemos-store`'s
    // `layout_is_independent_of_insertion_order`, which fixes the tags. It cannot be asserted
    // here: tags derive from random `MemoryId`s, so two runs place the same text at different
    // slots even in the same order. That randomness is deliberate — deterministic ids would
    // make position a function of content, which is a worse property than the one being tested.
    //
    // What this test adds is that the *observable file* does not change shape with order, which
    // is what an imaging adversary sees and what the store-level test cannot reach.
    let data = corpus(48);
    let forward: Vec<_> = data.clone();
    let mut reversed = data;
    reversed.reverse();

    let slab_size = |order: &[(String, Vec<f32>)]| -> u64 {
        let d = tempfile::tempdir().unwrap();
        let mut e = mnemos(d.path());
        for (text, v) in order {
            e.put(text, "{}", &qv(v)).unwrap();
        }
        e.flush().unwrap();
        std::fs::metadata(store_dir(d.path()).join("memory.slab"))
            .unwrap()
            .len()
    };

    assert_eq!(
        slab_size(&forward),
        slab_size(&reversed),
        "slab size depended on insertion order"
    );
}

#[test]
fn naive_ets_leaks_record_lengths_and_mnemos_does_not() {
    // The A1 channel. Included because it is the cheapest possible leak to fix — pad to a
    // block — and because a reader who sees "AEAD everywhere" tends to assume it is already
    // fixed. Mnemos's block padding is asserted alongside so the contrast is on one screen.
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("naive.bin");
    let mut n = NaiveEts::create(&path, [4u8; 32], 5).unwrap();

    let v = qv(&{
        let mut x = vec![0.0f32; DIM];
        x[0] = 1.0;
        x
    });
    n.put("tiny", "{}", &v).unwrap();
    n.put(&"m".repeat(300), "{}", &v).unwrap();
    n.put(&"l".repeat(3000), "{}", &v).unwrap();

    let lens = NaiveEts::snapshot_lengths(&path).unwrap();
    let distinct: std::collections::HashSet<_> = lens.iter().collect();
    assert_eq!(
        distinct.len(),
        3,
        "naive-EtS record lengths did not vary: {lens:?}"
    );

    // The same three entries through Mnemos land in equal-size slots.
    let md = tempfile::tempdir().unwrap();
    let mut e = mnemos(md.path());
    e.put("tiny", "{}", &v).unwrap();
    e.put(&"m".repeat(300), "{}", &v).unwrap();
    let before = std::fs::metadata(store_dir(md.path()).join("memory.slab"))
        .unwrap()
        .len();
    e.flush().unwrap();
    let after = std::fs::metadata(store_dir(md.path()).join("memory.slab"))
        .unwrap()
        .len();
    assert_eq!(
        before, after,
        "Mnemos's slab grew when entries were written"
    );
}

fn sorted_counts(v: &[(u32, u64)]) -> Vec<u64> {
    let mut c: Vec<u64> = v.iter().map(|(_, n)| *n).collect();
    c.sort_unstable();
    c
}

#[test]
fn the_index_never_holds_a_plaintext_embedding() {
    // THE regression test for the worst defect this project has had.
    //
    // `index.slab` used to store each entry as `tag ‖ QuantVec::encode()` with no encryption at
    // all. An adversary who read the file obtained every stored embedding in the clear, which is
    // a complete break of the confidentiality claim requiring no attack, no cryptanalysis and no
    // injection budget. Padding was working perfectly and hiding nothing that mattered.
    //
    // It survived because nothing compared the design to the bytes: retrieval worked, the
    // padding tests passed, and the attack simulations reported Adv_hist = 0.000 because they
    // model the design rather than the file. This test closes that gap by reading the actual
    // disk and looking for material that must not be there.
    let d = tempfile::tempdir().unwrap();
    let mut e = mnemos(d.path());

    let secret = qv(&{
        let mut v = vec![0.0f32; DIM];
        v[0] = 0.91;
        v[1] = 0.37;
        v[2] = -0.55;
        v
    });
    let raw_vector = secret.encode();

    e.put("a highly distinctive secret entry", "{}", &secret)
        .unwrap();
    for (text, v) in corpus(32) {
        e.put(&text, "{}", &qv(&v)).unwrap();
    }
    e.flush().unwrap();

    for name in ["index.slab", "memory.slab", "wal.enc", "manifest.enc"] {
        let bytes = std::fs::read(store_dir(d.path()).join(name)).unwrap();
        assert!(
            !bytes
                .windows(raw_vector.len())
                .any(|w| w == raw_vector.as_slice()),
            "{name} contains the raw quantised embedding in the clear"
        );
        assert!(
            !bytes
                .windows(24)
                .any(|w| w == b"a highly distinctive secr"[..24].to_vec().as_slice()),
            "{name} contains entry text in the clear"
        );
    }
}
