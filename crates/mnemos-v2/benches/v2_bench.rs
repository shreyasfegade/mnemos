//! Criterion benchmarks for Mnemos v2: search and the amortized put, at n = 10^3 and 10^4,
//! at the shipped dim (768). Run with `cargo bench -p mnemos-v2`.

use std::path::Path;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use mnemos_core::traits::SearchableMemory;
use mnemos_core::types::{Embedding, QuantVec};
use mnemos_core::Params;
use mnemos_keystore::software::SoftwareKeystore;
use mnemos_v2::{Ablation, EngineV2};

const DIM: usize = 768;

fn params(n: usize) -> Params {
    let mut p = Params::discover().expect("params.toml");
    let buckets = (n as f64).sqrt().ceil() as u32;
    let buckets = buckets.next_power_of_two().max(16);
    let capacity = (((n as f64 / f64::from(buckets)) * 2.0).ceil() as u32)
        .next_power_of_two()
        .max(32);
    p.index.buckets = buckets;
    p.index.bucket_capacity = capacity;
    p.index.probe_width = 4;
    p.index.probe_spread = 2;
    p.index.top_k = 10;
    p.embedding.dim = DIM;
    p.crypto.block_size_bytes = 4096;
    p.epoch.epoch_writes = 16;
    p.epoch.epoch_ms = 3_600_000;
    p.memory.arena_bytes = 4 * 1024 * 1024;
    p.v2.durable = false;
    p
}

fn vec_at(i: usize) -> QuantVec {
    let mut v = vec![0.0f32; DIM];
    for (j, x) in v.iter_mut().enumerate() {
        *x = ((i * 131 + j * 17) % 251) as f32 / 251.0 - 0.5;
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    QuantVec::quantise(&Embedding(v.into_iter().map(|x| x / n).collect()))
}

fn build(dir: &Path, n: usize, p: &Params) -> EngineV2 {
    std::env::set_var("MNEMOS_I_ACCEPT_INSECURE", "1");
    let ks = SoftwareKeystore::open_at(&dir.join("k"), true).unwrap();
    let cents: Vec<QuantVec> = (0..p.index.buckets as usize).map(vec_at).collect();
    let mut e = EngineV2::create(dir, p.clone(), Box::new(ks), cents, Ablation::default()).unwrap();
    for i in 0..n {
        // Skip records whose partition is full; the geometry is sized so most land.
        let _ = e.put(
            &format!("record number {i} with some body text"),
            "{}",
            &vec_at(i + 1000),
        );
    }
    e.flush().unwrap();
    e
}

fn bench(c: &mut Criterion) {
    for &n in &[1000usize, 10000] {
        let p = params(n);
        let dir = tempfile::tempdir().unwrap();
        let e = build(dir.path(), n, &p);
        let q = vec_at(500);
        c.bench_function(&format!("v2_search/{n}"), |b| {
            b.iter(|| e.search(&q).unwrap());
        });

        c.bench_function(&format!("v2_put_amortised/{n}"), |b| {
            b.iter_batched(
                || (),
                |()| {
                    let dir = tempfile::tempdir().unwrap();
                    let mut e = build(dir.path(), n, &p);
                    for i in 0..16 {
                        let _ = e.put(&format!("amortised put {i}"), "{}", &vec_at(i + 90_000));
                    }
                    e.flush().unwrap();
                },
                BatchSize::PerIteration,
            );
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
