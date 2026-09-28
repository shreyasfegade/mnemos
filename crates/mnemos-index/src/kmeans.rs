//! Spherical k-means — the partitioner, shared by everything that needs partitions.
//!
//! One implementation, used by `EncryptedIvf`, by `mnemos-baselines::plain`, by the benchmark
//! fixtures and by `mnemos init`. That sharing is not tidiness: the paper quotes Mnemos's
//! latency and recall *relative to a plaintext IVF*, and if the two used different partitioners
//! the comparison would be measuring the partitioner rather than the security machinery.
//!
//! ## Spherical, not Euclidean
//!
//! Embeddings are L2-normalised and compared by cosine, so centroids are re-normalised after
//! every update. Skipping the renormalisation gives centroids that drift off the unit sphere,
//! at which point `cosine()` against them is no longer the quantity being minimised and
//! assignment quality degrades quietly — quietly being the problem.
//!
//! ## Determinism
//!
//! Seeded `ChaCha`, k-means++ initialisation, a fixed iteration count and a deterministic
//! tie-break. The same sample and seed produce byte-identical centroids on any machine, which
//! is what lets a figure be regenerated six months later and match.

use mnemos_core::types::{Embedding, QuantVec};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Iterations of Lloyd's algorithm. Fixed rather than convergence-based: an early exit on a
/// convergence test makes the result depend on floating-point details that vary across
/// machines, and reproducibility is worth more here than the last fraction of a percent of
/// quantisation error.
const ITERATIONS: usize = 25;

/// Partition a sample into `k` unit-norm centroids.
///
/// Returns exactly `k` centroids even when the sample is smaller than `k` or contains
/// duplicates — `EncryptedIvf::create` requires exactly `buckets` of them, and an index with a
/// short centroid table would silently reroute assignment. Deficient centroids are filled with
/// deterministic pseudo-random unit vectors, which are simply partitions that stay empty.
///
/// # Panics
/// Never. An empty sample yields `k` pseudo-random centroids.
#[must_use]
pub fn spherical_kmeans(sample: &[Vec<f32>], k: usize, dim: usize, seed: u64) -> Vec<QuantVec> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let points: Vec<Vec<f32>> = sample
        .iter()
        .map(|v| {
            let mut p = v.clone();
            p.resize(dim, 0.0);
            normalise(&mut p);
            p
        })
        .collect();

    if points.is_empty() {
        return (0..k)
            .map(|_| random_unit(&mut rng, dim))
            .map(quantise)
            .collect();
    }

    let mut centroids = kmeans_plus_plus(&points, k, dim, &mut rng);

    for _ in 0..ITERATIONS {
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for p in &points {
            let c = nearest(p, &centroids);
            counts[c] += 1;
            for (s, v) in sums[c].iter_mut().zip(p.iter()) {
                *s += *v;
            }
        }

        for i in 0..k {
            if counts[i] == 0 {
                // An empty partition is re-seeded rather than left in place. Leaving it would
                // permanently waste a partition, and with `buckets = 256` over a clustered
                // corpus that is a measurable recall loss attributable to nothing visible.
                centroids[i] = random_unit(&mut rng, dim);
                continue;
            }
            centroids[i].clone_from(&sums[i]);
            normalise(&mut centroids[i]);
        }
    }

    centroids.into_iter().map(quantise).collect()
}

/// k-means++ seeding: first centre uniform, each subsequent one sampled proportionally to
/// squared cosine distance from the nearest chosen centre.
fn kmeans_plus_plus(
    points: &[Vec<f32>],
    k: usize,
    dim: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<Vec<f32>> {
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(points[rng.gen_range(0..points.len())].clone());

    while centroids.len() < k {
        let weights: Vec<f32> = points
            .iter()
            .map(|p| {
                let best = centroids
                    .iter()
                    .map(|c| dot(p, c))
                    .fold(f32::NEG_INFINITY, f32::max);
                // Cosine distance, squared, floored at zero so numerical noise cannot make a
                // weight negative and corrupt the running total.
                let d = (1.0 - best).max(0.0);
                d * d
            })
            .collect();

        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            // Every point coincides with a chosen centre; the rest of the table is filler.
            centroids.push(random_unit(rng, dim));
            continue;
        }

        let mut target = rng.gen_range(0.0..total);
        let mut chosen = points.len() - 1;
        for (i, w) in weights.iter().enumerate() {
            target -= *w;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(points[chosen].clone());
    }
    centroids
}

fn nearest(p: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_sim = f32::NEG_INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let s = dot(p, c);
        // Strictly greater, so the lowest index wins a tie and the result is order-independent.
        if s > best_sim {
            best_sim = s;
            best = i;
        }
    }
    best
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalise(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in &mut *v {
            *x /= n;
        }
    } else if let Some(first) = v.first_mut() {
        *first = 1.0;
    }
}

fn random_unit(rng: &mut ChaCha20Rng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    normalise(&mut v);
    v
}

fn quantise(v: Vec<f32>) -> QuantVec {
    QuantVec::quantise(&Embedding(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three well-separated clusters on the unit sphere.
    fn clustered(n_per: usize, dim: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut pts = Vec::new();
        let mut labels = Vec::new();
        for c in 0..3usize {
            for _ in 0..n_per {
                let mut v = vec![0.0f32; dim];
                v[c] = 1.0;
                for x in &mut *v {
                    *x += rng.gen_range(-0.05f32..0.05);
                }
                normalise(&mut v);
                pts.push(v);
                labels.push(c);
            }
        }
        (pts, labels)
    }

    #[test]
    fn returns_exactly_k_centroids_even_from_a_degenerate_sample() {
        // EncryptedIvf::create rejects a short centroid table, so this is a hard requirement
        // and not a nicety.
        assert_eq!(spherical_kmeans(&[], 16, 8, 1).len(), 16);
        assert_eq!(
            spherical_kmeans(&[vec![1.0, 0.0, 0.0, 0.0]], 16, 4, 1).len(),
            16
        );
    }

    #[test]
    fn centroids_are_unit_norm_after_quantisation() {
        let (pts, _) = clustered(40, 16, 7);
        for c in spherical_kmeans(&pts, 8, 16, 7) {
            let self_sim = c.cosine(&c);
            assert!(
                (self_sim - 1.0).abs() < 0.05,
                "centroid self-similarity {self_sim} — it left the unit sphere, so cosine \
                 against it is no longer the quantity being minimised"
            );
        }
    }

    #[test]
    fn recovers_planted_clusters() {
        let (pts, labels) = clustered(60, 16, 11);
        let centroids = spherical_kmeans(&pts, 3, 16, 11);
        let assign: Vec<usize> = pts
            .iter()
            .map(|p| {
                let q = QuantVec::quantise(&Embedding(p.clone()));
                centroids
                    .iter()
                    .enumerate()
                    .max_by(|a, b| {
                        q.cosine(a.1)
                            .partial_cmp(&q.cosine(b.1))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map_or(0, |(i, _)| i)
            })
            .collect();

        // Purity rather than exact label match: cluster ids are arbitrary.
        let mut confusion = [[0usize; 3]; 3];
        for (a, l) in assign.iter().zip(labels.iter()) {
            confusion[*l][*a] += 1;
        }
        let pure: usize = confusion
            .iter()
            .map(|row| *row.iter().max().unwrap_or(&0))
            .sum();
        assert!(
            pure as f32 / assign.len() as f32 > 0.95,
            "purity {pure}/{} — the partitioner is not recovering planted structure",
            assign.len()
        );
    }

    #[test]
    fn is_deterministic_across_runs() {
        // A figure regenerated six months later must match. Byte-identical, not approximately.
        let (pts, _) = clustered(30, 12, 3);
        let a = spherical_kmeans(&pts, 5, 12, 99);
        let b = spherical_kmeans(&pts, 5, 12, 99);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.encode(), y.encode());
        }
    }
}
