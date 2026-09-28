//! Criterion benchmarks feeding experiment **E2**.
//!
//! Every number the paper quotes for latency, throughput or overhead comes from here. Which
//! means these are not developer conveniences — they are figure inputs, and they carry the
//! obligations that come with that:
//!
//! - **Record the keystore provider.** A run on `software-INSECURE` is not comparable to a run
//!   on `windows-cng-pcp`, and six weeks later nobody will remember which produced which CSV.
//! - **Record `Params::hash()`.** A latency number without its configuration is not a
//!   measurement.
//! - **Report p50 and p95**, not means. Tail latency is what an interactive agent feels.
//!
//! ## The geometry bug this file used to have, and why it mattered
//!
//! An earlier version held `buckets = 64, bucket_capacity = 128` fixed across every corpus
//! size. That gives 8,192 index slots. At `n = 10,000` it therefore had **1,808 entries with
//! nowhere to go** — the index was saturated and 18% of the corpus was silently unindexed,
//! because `Engine::flush` tolerates a full partition rather than failing the write.
//!
//! The consequence was a benchmark that appeared to show Mnemos's search cost *flat* in `n`.
//! It was flat because the index had stopped growing, not because the design is free. The
//! measured "overhead falls from 14.7× to 2.4× as the corpus grows" was an artifact of the same
//! saturation.
//!
//! Correct IVF sizing is `buckets ≈ sqrt(n)` with `bucket_capacity` large enough to hold the
//! largest partition plus headroom. Under that rule, a constant-volume search reads
//! `k_probe × C ≈ k_probe × sqrt(n)` slots: **sub-linear in `n`, not constant.** That is still
//! asymptotically better than every baseline here (all linear), and it is what the code
//! actually does. [`geometry`] implements the rule and is the only place it is expressed.

use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mnemos_baselines::diana::DianaFp;
use mnemos_baselines::naive::NaiveEts;
use mnemos_baselines::plain::PlaintextIvf;
use mnemos_baselines::sophos::SophosFp;
use mnemos_baselines::Keywords;
use mnemos_bench::{quantise_all, query_set, synthetic_corpus, RunMetadata};
use mnemos_core::traits::SearchableMemory;
use mnemos_core::types::QuantVec;
use mnemos_core::Params;
use mnemos_engine::Engine;
use mnemos_keystore::software::SoftwareKeystore;

/// Default embedding width for a quick local run. **The paper's numbers are taken at 768**, the
/// deployed model's width, via `MNEMOS_BENCH_DIM=768`.
///
/// Scoring is linear in dimension, so a number measured at 128 understates the index-scan half
/// of search latency by roughly 6x. Anything quoted anywhere must say which width produced it;
/// the provenance line printed at the start of every run does.
const DIM: usize = 128;

const SEED: u64 = 20_260_813;

/// Corpus sizes. `MNEMOS_BENCH_SIZES=1000,10000,100000` overrides.
fn sizes() -> Vec<usize> {
    std::env::var("MNEMOS_BENCH_SIZES").ok().map_or_else(
        || vec![1_000usize, 10_000],
        |v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
    )
}

fn dim() -> usize {
    std::env::var("MNEMOS_BENCH_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DIM)
}

/// Partition count and slots-per-partition for a corpus of `n` entries.
///
/// `buckets ≈ sqrt(n)` rounded to a power of two is the standard IVF rule; capacity is twice the
/// mean occupancy so a moderately skewed corpus still fits without the index saturating. Both
/// numbers are returned together because getting them out of step is exactly the bug documented
/// in the module header.
fn geometry(n: usize) -> (u32, u32) {
    let buckets = ((n as f64).sqrt().round() as u32)
        .next_power_of_two()
        .clamp(16, 1024);
    let capacity = ((n as f64 / f64::from(buckets)) * 2.0).ceil().max(32.0) as u32;
    (buckets, capacity.next_power_of_two())
}

/// Declared capacity for a corpus of `n`, expressed once so the benchmark and the chunking that
/// keeps it inside that capacity cannot drift apart.
fn n_max_for(n: usize) -> u64 {
    ((n as u64 * 2).max(1024)).next_power_of_two()
}

fn bench_params(n: usize, decoys: u32) -> Params {
    let (buckets, capacity) = geometry(n);
    let mut p = Params::discover().expect("params.toml");
    p.embedding.dim = dim();
    p.index.buckets = buckets;
    p.index.bucket_capacity = capacity;
    p.index.probe_width = 4;
    p.index.decoy_buckets = decoys;
    // `MNEMOS_BENCH_SPREAD` overrides the query-equality mitigation, which `params.toml` now
    // ships **on** at `s = 2`. The paper reports both configurations, so both have to be
    // reachable from one binary; inheriting silently would also mean a change to `params.toml`
    // could move every latency number in the paper without anything saying so.
    if let Some(s) = std::env::var("MNEMOS_BENCH_SPREAD")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        p.index.probe_spread = s;
    }
    p.index.top_k = 10;
    // Slab headroom so the load factor stays inside `max_load_factor` and the engine does not
    // (correctly) refuse to flush mid-benchmark.
    p.index.n_max = n_max_for(n);
    p.crypto.block_size_bytes = 1024;
    p.crypto.max_entry_bytes = 1024;
    p.memory.arena_bytes = 256 * 1024;
    p
}

fn keyword_space(corpus: &[Vec<f32>], buckets: usize) -> Keywords {
    Keywords::train(corpus, buckets, dim(), SEED)
}

fn build_mnemos(
    n: usize,
    decoys: u32,
    corpus: &[QuantVec],
    kw: &Keywords,
) -> (tempfile::TempDir, Engine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = bench_params(n, decoys);
    let ks = SoftwareKeystore::open_at(&dir.path().join("keystore.bin"), true)
        .expect("software keystore — benchmarks are not security results");
    let mut e = Engine::create(
        &dir.path().join("store"),
        p,
        Box::new(ks),
        kw.centroids().to_vec(),
    )
    .expect("create");

    for (i, v) in corpus.iter().enumerate() {
        e.put(&format!("entry {i}"), "{}", v).expect("put");
    }
    e.flush().expect("flush");
    (dir, e)
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");
    group.measurement_time(Duration::from_secs(8));
    let d = dim();

    for n in sizes() {
        let (buckets, capacity) = geometry(n);
        eprintln!(
            "n={n}: buckets={buckets} capacity={capacity} index_slots={} \
             (must be >= n or the index saturates and the result is meaningless)",
            u64::from(buckets) * u64::from(capacity)
        );
        assert!(
            u64::from(buckets) * u64::from(capacity) >= n as u64,
            "index geometry cannot hold the corpus"
        );

        let raw = synthetic_corpus(n, d, SEED);
        let corpus = quantise_all(&raw);
        let queries = quantise_all(&query_set(64, d, SEED));
        let kw = keyword_space(&raw, buckets as usize);
        group.throughput(Throughput::Elements(1));

        // --- the latency floor every overhead multiple is quoted against ---
        let mut plain = PlaintextIvf::build(&raw, buckets as usize, d, 4, 10, SEED).expect("build");
        for (i, v) in corpus.iter().enumerate() {
            plain.put(&format!("entry {i}"), "{}", v).expect("put");
        }
        group.bench_with_input(BenchmarkId::new("plaintext_ivf", n), &n, |b, _| {
            let mut q = queries.iter().cycle();
            b.iter(|| plain.search(q.next().expect("cycle")).expect("search"));
        });

        // --- the O(n) strawman. Skipped past 10k: at 100k it is ~300 ms per query, so
        //     criterion's sampling alone would take an hour to say what O(n) already says. ---
        if n <= 10_000 {
            let ndir = tempfile::tempdir().expect("tempdir");
            let mut naive =
                NaiveEts::create(&ndir.path().join("n.bin"), [7u8; 32], 10).expect("create");
            for (i, v) in corpus.iter().enumerate() {
                naive.put(&format!("entry {i}"), "{}", v).expect("put");
            }
            group.bench_with_input(BenchmarkId::new("naive_ets", n), &n, |b, _| {
                let mut q = queries.iter().cycle();
                b.iter(|| naive.search(q.next().expect("cycle")).expect("search"));
            });
        }

        // --- the forward-private baselines ---
        let sdir = tempfile::tempdir().expect("tempdir");
        let mut sophos = SophosFp::create(sdir.path(), [7u8; 32], kw.clone(), 10).expect("create");
        let ddir = tempfile::tempdir().expect("tempdir");
        let mut diana = DianaFp::create(ddir.path(), [7u8; 32], kw.clone(), 10).expect("create");
        for (i, v) in corpus.iter().enumerate() {
            sophos.put(&format!("entry {i}"), "{}", v).expect("put");
            diana.put(&format!("entry {i}"), "{}", v).expect("put");
        }
        group.bench_with_input(BenchmarkId::new("sophos_fp", n), &n, |b, _| {
            let mut q = queries.iter().cycle();
            b.iter(|| sophos.search(q.next().expect("cycle")).expect("search"));
        });
        group.bench_with_input(BenchmarkId::new("diana_fp", n), &n, |b, _| {
            let mut q = queries.iter().cycle();
            b.iter(|| diana.search(q.next().expect("cycle")).expect("search"));
        });

        // --- Mnemos, with and without decoy partitions ---
        for decoys in [0u32, 4] {
            let (_dir, engine) = build_mnemos(n, decoys, &corpus, &kw);
            let name = format!("mnemos_d{decoys}");
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
                let mut q = queries.iter().cycle();
                b.iter(|| engine.search(q.next().expect("cycle")).expect("search"));
            });
        }
    }
    group.finish();
}

fn bench_write(c: &mut Criterion) {
    // Amortised put, and the epoch flush measured separately — they have different budgets
    // (write_p50_ms = 8, epoch_flush_ms = 30) and blending them would hide which one regressed.
    //
    // The flush is where the 4x write amplification lives. Measuring it apart from `put` is the
    // difference between reporting that cost and burying it in an average.
    //
    // ## Why this is parameterised by `n` now
    //
    // It was not. It measured one geometry — `entries = 1_000` — while the paper quoted the
    // resulting ratio as "33x plaintext write latency **at n = 10^4**". The read side was
    // measured at both sizes and the write side at one, and the sentence reporting them said
    // otherwise. The overhead is quoted per size here so the claim and the measurement agree.
    let mut group = c.benchmark_group("write");
    group.measurement_time(Duration::from_secs(8));

    let width = dim();
    for &entries in &sizes() {
        let (buckets, _) = geometry(entries);
        let raw = synthetic_corpus(entries, width, SEED);
        let corpus = quantise_all(&raw);
        let kw = keyword_space(&raw, buckets as usize);

        // A store has finite capacity and `b.iter` does not. Timed in chunks that fit inside
        // the load factor, rebuilding the store between chunks with the rebuild *outside* the
        // timed region.
        //
        // Written as a plain `b.iter` this benchmark filled its own slab and panicked with
        // `CapacityExceeded { live: 1216, n_max: 2048 }` -- after the paper had already quoted
        // its output. A benchmark that cannot run to completion at the geometry it claims to
        // measure is the same defect as the saturated index in the module header, one layer up.
        // Slots a chunk may consume before the store refuses to write. Deliberately well under
        // `max_load_factor` (0.6), because **a flush consumes more slots than the loop issues
        // puts**: a padded batch writes `epoch_writes` slots whether or not that many records
        // were real. Sizing this against puts-issued rather than slots-consumed is how the
        // first fix still panicked at `live: 1216` against a headroom of 1126.
        let headroom = (n_max_for(entries) * 35 / 100).max(64);
        // One iteration is one full batch: `epoch_writes` puts, the last of which triggers the
        // flush. Timing single puts let criterion choose iteration counts below the batch size,
        // in which case no flush fell inside the timed region and the "amortised" figure
        // measured staging alone. Divide by `epoch_writes` for the per-put cost.
        let batch = Params::discover().expect("params.toml").epoch.epoch_writes as usize;
        group.bench_with_input(
            BenchmarkId::new("put_amortised", entries),
            &entries,
            |b, _| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    let mut remaining = iters;
                    let mut i = 0usize;
                    let batches = (headroom / (2 * batch as u64)).max(1);
                    while remaining > 0 {
                        let (_dir, mut engine) = build_mnemos(entries, 0, &corpus[..0], &kw);
                        let chunk = remaining.min(batches);
                        let t0 = Instant::now();
                        for _ in 0..chunk {
                            for _ in 0..batch {
                                let v = &corpus[i % corpus.len()];
                                i += 1;
                                engine.put("benchmark entry", "{}", v).expect("put");
                            }
                        }
                        total += t0.elapsed();
                        remaining -= chunk;
                    }
                    total
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("epoch_flush", entries),
            &entries,
            |b, _| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    let mut remaining = iters;
                    // Each iteration writes a full batch *and* pays for the padding of that batch,
                    // so budget two `epoch_writes` worth of slots per iteration.
                    let batches = (headroom / 32).max(1);
                    while remaining > 0 {
                        let (_dir, mut engine) = build_mnemos(entries, 0, &corpus[..0], &kw);
                        let chunk = remaining.min(batches);
                        let t0 = Instant::now();
                        for _ in 0..chunk {
                            for v in corpus.iter().take(16) {
                                engine.put("benchmark entry", "{}", v).expect("put");
                            }
                            engine.flush().expect("flush");
                        }
                        total += t0.elapsed();
                        remaining -= chunk;
                    }
                    total
                });
            },
        );

        // Chunked exactly as Mnemos is, and for a reason that took two wrong turns to find.
        //
        // The obvious version -- one `PlaintextIvf`, a plain `b.iter` -- looks safe because this
        // structure grows on demand and has no slot budget to exhaust. It exhausts *memory*
        // instead: criterion runs millions of iterations, each appends a row holding a `String`
        // and a `dim`-wide vector, and at `n = 10^4` the benchmark was still going after twenty
        // minutes with gigabytes resident. It had stopped measuring insertion and started
        // measuring the allocator under pressure.
        //
        // Chunking it the first way was worse: `PlaintextIvf::build` trains k-means, so
        // rebuilding per chunk re-ran a full 10,000x768 clustering between every timed window.
        // `from_centroids` exists so the reset is an allocation rather than a training run.
        //
        // "It has no capacity to exhaust" was simply wrong, and it was wrong in the direction
        // that produces a plausible number rather than a crash.
        let plain_centroids =
            PlaintextIvf::build(&raw, buckets as usize, width, 4, 10, SEED).expect("build");
        let plain_centroids = plain_centroids.centroids().to_vec();
        group.bench_with_input(
            BenchmarkId::new("plaintext_put", entries),
            &entries,
            |b, _| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    let mut remaining = iters;
                    let mut i = 0usize;
                    while remaining > 0 {
                        let mut plain =
                            PlaintextIvf::from_centroids(plain_centroids.clone(), 4, 10)
                                .expect("from_centroids");
                        let chunk = remaining.min(headroom);
                        let t0 = Instant::now();
                        for _ in 0..chunk {
                            let v = &corpus[i % corpus.len()];
                            i += 1;
                            plain.put("benchmark entry", "{}", v).expect("put");
                        }
                        total += t0.elapsed();
                        remaining -= chunk;
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn bench_crypto(c: &mut Criterion) {
    // Primitive-level, so that when search latency misses its target we can tell whether the
    // cause is crypto or I/O without guessing.
    use mnemos_core::types::Tag;
    use mnemos_crypto::aead::seal_content;
    use mnemos_crypto::kdf::{DerivedKey, KeyHierarchy};
    use mnemos_crypto::pprf::{eval_leaf, tree_root};
    use mnemos_crypto::prf::mac16;

    let mut group = c.benchmark_group("crypto");
    let keys = KeyHierarchy::derive(&[42u8; 32]).expect("derive");
    let payload = vec![0xABu8; 4096];
    let tag = Tag([9u8; 16]);

    group.throughput(Throughput::Bytes(4096));
    group.bench_function("xchacha20poly1305_seal_4k", |b| {
        b.iter(|| seal_content(&keys.content, &payload, tag, 4096).expect("seal"));
    });

    group.bench_function("hmac_sha256_tag", |b| {
        b.iter(|| mac16(&keys.tag.0, tag.as_bytes()));
    });

    let root = tree_root(&DerivedKey([3u8; 32]), mnemos_core::types::BucketId(0)).expect("root");
    group.bench_function("ggm_pprf_eval_depth32", |b| {
        let mut leaf = 0u64;
        b.iter(|| {
            leaf = leaf.wrapping_add(1);
            eval_leaf(&root, 32, leaf & 0xFFFF_FFFF)
        });
    });

    group.finish();
}

/// Print provenance once, so a captured criterion log is self-describing.
fn bench_metadata(_c: &mut Criterion) {
    let p = bench_params(1_000, 0);
    let dir = tempfile::tempdir().expect("tempdir");
    let ks = SoftwareKeystore::open_at(&dir.path().join("k"), true).expect("keystore");
    let m = RunMetadata::collect(&p, &ks, SEED);
    eprintln!(
        "mnemos-bench provenance: {}",
        serde_json::to_string(&m).unwrap_or_default()
    );
    eprintln!("mnemos-bench dim={} sizes={:?}", dim(), sizes());
    if !m.is_security_grade() {
        eprintln!(
            "mnemos-bench: NOT a security-grade run ({}). Performance numbers are valid; \
             security claims from this run are not.",
            m.keystore_provider
        );
    }
}

criterion_group!(
    benches,
    bench_metadata,
    bench_search,
    bench_write,
    bench_crypto
);
criterion_main!(benches);
