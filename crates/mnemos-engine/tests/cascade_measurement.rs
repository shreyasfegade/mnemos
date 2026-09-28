//! Measures Robin Hood cascade length against load factor.
//!
//! Not a pass/fail test — a **calibration harness**. `epoch.slot_write_budget` and
//! `epoch.max_load_factor` are a matched pair: the budget must exceed the worst cascade seen at
//! the load factor. Both were originally guessed ("3× is comfortable below 0.7") and the guess
//! was wrong — the budget was exceeded at 0.6875. This produces the numbers to set them from.
//!
//! Run with `cargo test -p mnemos-engine --test cascade_measurement -- --nocapture --ignored`.

use std::sync::Arc;

use mnemos_core::event::{LeakEvent, RecordingSink};
use mnemos_core::traits::SearchableMemory;
use mnemos_core::types::{Embedding, QuantVec};
use mnemos_core::Params;
use mnemos_engine::Engine;
use mnemos_keystore::software::SoftwareKeystore;

fn qv(v: Vec<f32>) -> QuantVec {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    QuantVec::quantise(&Embedding(v.into_iter().map(|x| x / n).collect()))
}

#[test]
#[ignore = "calibration harness, not a pass/fail test"]
fn measure_cascade_growth_against_load_factor() {
    println!("\n  load   flush#   raw slot writes (budget disabled)");
    println!("  ----   ------   ---------------------------------");

    let mut p = Params::discover().unwrap();
    p.index.n_max = 1024;
    p.index.buckets = 8;
    p.index.bucket_capacity = 256;
    p.index.probe_width = 2;
    p.index.decoy_buckets = 0;
    p.index.top_k = 3;
    p.crypto.block_size_bytes = 512;
    p.crypto.max_entry_bytes = 512;
    p.embedding.dim = 8;
    p.epoch.compaction_slots = 4;
    p.memory.arena_bytes = 64 * 1024;
    // Budget set to epoch_writes so padding never masks the raw cascade count.
    p.epoch.slot_write_budget = p.epoch.epoch_writes;

    let dir = tempfile::tempdir().unwrap();
    let ks = SoftwareKeystore::open_at(&dir.path().join("k"), true).unwrap();
    let centroids: Vec<QuantVec> = (0..p.index.buckets)
        .map(|i| {
            let mut v = vec![0.0f32; p.embedding.dim];
            v[i as usize % p.embedding.dim] = 1.0;
            qv(v)
        })
        .collect();
    let mut e = Engine::create(dir.path(), p.clone(), Box::new(ks), centroids).unwrap();
    let sink = Arc::new(RecordingSink::new());
    e.set_event_sink(sink.clone());

    let rounds = (p.index.n_max / u64::from(p.epoch.epoch_writes)) as usize;
    let mut worst_at = Vec::new();

    for round in 0..rounds {
        for i in 0..p.epoch.epoch_writes {
            if e.put(
                &format!("r{round}-{i}"),
                "{}",
                &qv(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
            .is_err()
            {
                break;
            }
        }
        let writes = sink
            .drain()
            .iter()
            .filter(|x| matches!(x, LeakEvent::SlotWrite { .. }))
            .count();
        if writes == 0 {
            continue;
        }
        let load = ((round + 1) * p.epoch.epoch_writes as usize) as f64 / p.index.n_max as f64;
        if load > 0.95 {
            break;
        }
        worst_at.push((load, writes));
        if round % 8 == 0 || writes > p.epoch.epoch_writes as usize * 2 {
            println!("  {load:.3}   {round:>6}   {writes}");
        }
    }

    println!("\n  Recommended (worst observed write count at or below each load):");
    for target in [0.40f64, 0.50, 0.60, 0.70, 0.80] {
        let worst = worst_at
            .iter()
            .filter(|(l, _)| *l <= target)
            .map(|(_, w)| *w)
            .max()
            .unwrap_or(0);
        let mult = worst as f64 / f64::from(p.epoch.epoch_writes);
        println!("    max_load_factor = {target:.2}  ->  slot_write_budget >= {worst}  ({mult:.1}x epoch_writes)");
    }
    println!();
}
