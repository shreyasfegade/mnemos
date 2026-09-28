//! The keyword space shared by the DSSE baselines.
//!
//! Sophos and Diana are *keyword* schemes; Mnemos is a *vector* store. Comparing them requires
//! fixing a correspondence, and the one used everywhere in this project — in the Python attack
//! lab, in `THREAT-MODEL.md` §3.4, and here — is:
//!
//! > **keyword ≡ semantic partition.**
//!
//! An entry's keyword is the k-means partition its embedding falls into. That is the honest
//! mapping for three reasons:
//!
//! 1. It is what an attacker actually cares about. Finding F2 is about recovering *what an
//!    entry is about*, and a partition is a topic.
//! 2. It gives the baselines the same granularity Mnemos has, so neither side is handicapped
//!    by a mismatched keyword space.
//! 3. It is the same partitioning function, from [`mnemos_index::spherical_kmeans`], so
//!    differences in the results cannot be attributed to different clusterings.
//!
//! A per-token keyword space (one keyword per word of text) would be more conventional for
//! DSSE, and would leak *more* — every entry would touch dozens of keywords instead of one.
//! Choosing the coarser space is the conservative choice, consistent with the fairness rule
//! stated in [`crate::sophos`].

use mnemos_core::types::{BucketId, QuantVec};

/// A fixed keyword space: `k` semantic partitions with a deterministic assignment rule.
#[derive(Clone)]
pub struct Keywords {
    centroids: Vec<QuantVec>,
}

impl Keywords {
    /// Train a keyword space over a sample.
    ///
    /// # Panics
    /// Never; `spherical_kmeans` always returns exactly `k` centroids.
    #[must_use]
    pub fn train(sample: &[Vec<f32>], k: usize, dim: usize, seed: u64) -> Self {
        Self {
            centroids: mnemos_index::spherical_kmeans(sample, k.max(1), dim, seed),
        }
    }

    /// Adopt an existing centroid table, so a baseline and an engine can be given **the same**
    /// partitioning rather than two independently trained ones.
    #[must_use]
    pub fn from_centroids(centroids: Vec<QuantVec>) -> Self {
        Self { centroids }
    }

    /// Number of keywords.
    #[must_use]
    pub fn len(&self) -> usize {
        self.centroids.len()
    }

    /// Whether the space is empty. Present because clippy asks for it next to `len`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    /// The centroid table, for handing the identical partitioning to another component.
    #[must_use]
    pub fn centroids(&self) -> &[QuantVec] {
        &self.centroids
    }

    /// The keyword a vector belongs to.
    ///
    /// Identical rule to `EncryptedIvf::assign` — nearest centroid by cosine, lowest index
    /// wins a tie — so an entry lands in the same partition in every scheme under comparison.
    #[must_use]
    pub fn assign(&self, v: &QuantVec) -> BucketId {
        let mut best = 0u32;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, c) in self.centroids.iter().enumerate() {
            let s = v.cosine(c);
            if s > best_sim {
                best_sim = s;
                best = i as u32;
            }
        }
        BucketId(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{corpus, quant};

    #[test]
    fn assignment_matches_the_encrypted_index() {
        // The comparison is void if the two disagree about which partition an entry is in.
        use mnemos_core::traits::Index;

        let (vecs, _) = corpus(150, 16, 5, 21);
        let kw = Keywords::train(&vecs, 8, 16, 21);

        let dir = tempfile::tempdir().unwrap();
        let mut params = mnemos_core::Params::discover().unwrap();
        params.index.buckets = 8;
        params.index.probe_width = 2;
        params.index.decoy_buckets = 0;
        params.embedding.dim = 16;
        let ivf = mnemos_index::EncryptedIvf::create(
            &dir.path().join("i.slab"),
            &params,
            kw.centroids().to_vec(),
        )
        .unwrap();

        for v in &vecs {
            let q = quant(v);
            assert_eq!(kw.assign(&q), ivf.assign(&q));
        }
    }
}
