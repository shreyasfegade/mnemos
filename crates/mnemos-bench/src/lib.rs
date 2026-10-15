//! Shared benchmark fixtures.
//!
//! Corpus generation, query sets, and the metadata block every result is stamped with. Kept in
//! a library rather than inside the bench file so `python/eval/` can reproduce the exact same
//! fixtures when generating figures — a benchmark and a figure disagreeing about their input
//! is a mistake that is very hard to notice.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use mnemos_core::types::{Embedding, QuantVec};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::Serialize;

/// Provenance stamped onto every benchmark result.
///
/// Without this, a CSV six weeks old is unusable: there is no way to tell whether it came from
/// a TPM run or the insecure software fallback, or which parameter set produced it.
#[derive(Debug, Serialize)]
pub struct RunMetadata {
    /// `Params::hash()` — ties the number to an exact configuration.
    pub params_hash: String,
    /// Key-custody provider name. `software-INSECURE` here means the run is not a security
    /// result and must not be reported as one.
    pub keystore_provider: String,
    /// Whether the locked arena's pages were genuinely locked.
    pub arena_locked: bool,
    /// Git commit.
    pub git_sha: String,
    /// RNG seed.
    pub seed: u64,
    /// Free-text machine description.
    pub machine: String,
}

impl RunMetadata {
    /// Collect provenance from the environment.
    ///
    /// `git_sha` falls back to `"unknown"` rather than failing: a benchmark that refuses to run
    /// outside a git checkout would be worse than one that records an honest gap. The field is
    /// *labelled* unknown rather than silently omitted, so a result carrying it can be
    /// recognised as unreproducible at a glance.
    #[must_use]
    pub fn collect(
        params: &mnemos_core::Params,
        keystore: &dyn mnemos_core::traits::KeyStore,
        seed: u64,
    ) -> Self {
        Self {
            params_hash: params.hash(),
            keystore_provider: {
                let name = keystore.provider_name();
                // The software provider already names itself `software-INSECURE`; appending the
                // suffix unconditionally produced `software-INSECURE-INSECURE` in every logged
                // run. Cosmetic, but the field exists to be read months later and a mangled
                // value is one more thing to have to interpret.
                if keystore.is_hardware() || name.ends_with("-INSECURE") {
                    name.to_owned()
                } else {
                    format!("{name}-INSECURE")
                }
            },
            arena_locked: params.memory.lock_pages,
            git_sha: git_sha(),
            seed,
            machine: machine(),
        }
    }

    /// Whether results from this run may be reported as *security* results.
    ///
    /// False on the software keystore or with page-locking disabled. Checked rather than
    /// trusted to memory: the whole reason `RunMetadata` exists is that six weeks later nobody
    /// remembers which configuration produced which CSV.
    #[must_use]
    pub fn is_security_grade(&self) -> bool {
        !self.keystore_provider.ends_with("-INSECURE") && self.arena_locked
    }
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn machine() -> String {
    format!(
        "{} {} {}-core",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get)
    )
}

/// Generate a reproducible synthetic corpus.
///
/// Seeded `ChaCha`, so a figure regenerates byte-identically on another machine.
///
/// **Clustered, not uniform.** A uniform corpus on the unit sphere would make IVF partitioning
/// meaningless — every partition would be equally (ir)relevant to every query — and would
/// flatter the recall numbers by removing the thing approximate search is approximating. The
/// cluster count is deliberately much smaller than the partition count so partitions subdivide
/// clusters, which is the regime real embeddings are in.
#[must_use]
pub fn synthetic_corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    clustered_corpus(n, dim, (n / 200).clamp(4, 64), seed)
}

/// [`synthetic_corpus`] with an explicit cluster count.
#[must_use]
pub fn clustered_corpus(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let k = clusters.max(1);

    let centres: Vec<Vec<f32>> = (0..k)
        .map(|_| {
            unit(
                &mut (0..dim)
                    .map(|_| rng.gen_range(-1.0f32..1.0))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    // Zipf-ish cluster weights. Real corpora are skewed, and a uniform assignment would hide
    // exactly the occupancy variation the snapshot attacks feed on — the benchmark would then
    // be measuring an easier problem than the evaluation does.
    let weights: Vec<f64> = (1..=k).map(|i| 1.0 / i as f64).collect();
    let total: f64 = weights.iter().sum();

    (0..n)
        .map(|_| {
            let mut target = rng.gen_range(0.0..total);
            let mut c = k - 1;
            for (i, w) in weights.iter().enumerate() {
                target -= *w;
                if target <= 0.0 {
                    c = i;
                    break;
                }
            }
            let mut v: Vec<f32> = centres[c]
                .iter()
                .map(|x| x + rng.gen_range(-0.30f32..0.30))
                .collect();
            unit(&mut v)
        })
        .collect()
}

/// A query set drawn from the same distribution as the corpus but with a different seed, so
/// queries are near-but-not-equal to stored entries — the realistic retrieval regime.
#[must_use]
pub fn query_set(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    clustered_corpus(n, dim, (n / 20).clamp(4, 64), seed.wrapping_add(0x5EED))
}

/// Quantise through the production path.
#[must_use]
pub fn quantise_all(v: &[Vec<f32>]) -> Vec<QuantVec> {
    v.iter()
        .map(|x| QuantVec::quantise(&Embedding(x.clone())))
        .collect()
}

fn unit(v: &mut Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in &mut *v {
            *x /= n;
        }
    }
    v.clone()
}

/// Recall@k of `got` against `truth`, both as ordered id lists.
///
/// Set overlap, not rank correlation: a retrieval system that returns the right ten documents
/// in the wrong order has not lost recall, and conflating the two would understate every
/// approximate scheme including the plaintext baseline.
#[must_use]
pub fn recall_at_k<T: PartialEq>(got: &[T], truth: &[T], k: usize) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let t = &truth[..k.min(truth.len())];
    let hit = got.iter().take(k).filter(|g| t.contains(g)).count();
    hit as f64 / t.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_reproducible_across_runs() {
        // A figure regenerated six months later must match, byte for byte.
        assert_eq!(synthetic_corpus(200, 32, 7), synthetic_corpus(200, 32, 7));
        assert_ne!(synthetic_corpus(200, 32, 7), synthetic_corpus(200, 32, 8));
    }

    #[test]
    fn corpus_vectors_are_unit_norm() {
        for v in synthetic_corpus(100, 16, 3) {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "vector norm was {n}");
        }
    }

    #[test]
    fn corpus_is_actually_clustered() {
        // The negative control. If the generator drifted to uniform, IVF recall numbers would
        // silently become meaningless while every other test kept passing.
        let c = clustered_corpus(500, 16, 5, 11);
        let q = &c[0];
        let mut sims: Vec<f32> = c
            .iter()
            .map(|v| v.iter().zip(q).map(|(a, b)| a * b).sum())
            .collect();
        sims.sort_by(|a, b| b.partial_cmp(a).unwrap());
        // With genuine clusters, the 10th-nearest neighbour is much closer than the median.
        let near = sims[10];
        let median = sims[sims.len() / 2];
        assert!(
            near - median > 0.25,
            "corpus is not clustered: 10th-NN sim {near}, median {median}"
        );
    }

    #[test]
    fn recall_counts_set_overlap_not_order() {
        assert!((recall_at_k(&[1, 2, 3], &[3, 2, 1], 3) - 1.0).abs() < 1e-9);
        assert!((recall_at_k(&[1, 2, 9], &[1, 2, 3], 3) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn metadata_flags_an_insecure_run() {
        // The guard that stops a software-keystore number being quoted as a security result.
        let m = RunMetadata {
            params_hash: "x".into(),
            keystore_provider: "software-INSECURE".into(),
            arena_locked: true,
            git_sha: "abc".into(),
            seed: 1,
            machine: "test".into(),
        };
        assert!(!m.is_security_grade());
    }
}
