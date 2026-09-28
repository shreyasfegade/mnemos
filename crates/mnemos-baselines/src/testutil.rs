//! Shared fixtures for this crate's tests.
//!
//! Deliberately *not* `mnemos-bench::synthetic_corpus`, even though the two generate the same
//! shape: depending on the bench crate from here would be a dependency cycle. The generator is
//! the same algorithm with the same seeding discipline, so a fixture built here and one built
//! there are comparable.

use mnemos_core::types::{Embedding, QuantVec};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::keyword::Keywords;

/// `n` unit vectors drawn from `clusters` isotropic Gaussians, plus their true cluster labels.
///
/// Clustered rather than uniform: a uniform corpus makes IVF partitioning meaningless and
/// flatters every recall number in the comparison.
#[must_use]
pub fn corpus(n: usize, dim: usize, clusters: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            normalise(&mut v);
            v
        })
        .collect();

    let mut pts = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let c = i % clusters;
        let mut v: Vec<f32> = centres[c]
            .iter()
            .map(|x| x + rng.gen_range(-0.25f32..0.25))
            .collect();
        normalise(&mut v);
        pts.push(v);
        labels.push(c);
    }
    (pts, labels)
}

/// Quantise a raw embedding through the production path.
#[must_use]
pub fn quant(v: &[f32]) -> QuantVec {
    QuantVec::quantise(&Embedding(v.to_vec()))
}

/// A keyword space trained on a fresh corpus of the given shape.
#[must_use]
pub fn keywords(k: usize, dim: usize, seed: u64) -> Keywords {
    let (vecs, _) = corpus(k * 20, dim, k, seed);
    Keywords::train(&vecs, k, dim, seed)
}

fn normalise(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in &mut *v {
            *x /= n;
        }
    }
}
