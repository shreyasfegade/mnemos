//! Plaintext IVF — the utility ceiling and the latency floor.
//!
//! No encryption, no padding, no epochs. Every performance number in the paper is quoted
//! relative to this, and every recall number is measured against its exact-NN results.
//!
//! Keeping it in-tree rather than shelling out to FAISS means the comparison uses the same
//! quantisation, the same partitioning, and the same code path for everything except the
//! security machinery — so "20× slower" is a statement about the security machinery, not about
//! two different libraries. Concretely it shares `mnemos_index::spherical_kmeans` and
//! `QuantVec`, so the *only* differences left are padding, AEAD, and the epoch schedule.
//!
//! ## What it is allowed to do that Mnemos is not
//!
//! All of it deliberately, and all of it observable in the numbers:
//!
//! - **Skip empty partitions.** Mnemos reads `bucket_capacity` slots whether occupied or not.
//! - **Early-exit** once `top_k` candidates are settled.
//! - **Variable probe width** — stop when the next centroid is far.
//! - **Write in place**, with no batching and no epoch delay.
//!
//! Each is a leakage channel closed elsewhere in the tree, so the gap between this baseline's
//! latency and Mnemos's *is* the price of the leakage profile, itemised.

use std::collections::HashMap;

use mnemos_core::error::{Error, Result};
use mnemos_core::traits::{SearchableMemory, Stat};
use mnemos_core::types::{BucketId, Epoch, Hit, Memory, MemoryId, QuantVec};

/// One stored row. Plaintext throughout — that is the point.
struct Row {
    id: MemoryId,
    text: String,
    meta: String,
    vector: QuantVec,
}

/// Unencrypted IVF over int8-quantised embeddings.
pub struct PlaintextIvf {
    centroids: Vec<QuantVec>,
    /// Partition contents. A `Vec` per partition, grown on demand — the shape of the structure
    /// is itself the occupancy leak that `mnemos-index::bucket` pads away.
    partitions: Vec<Vec<usize>>,
    rows: Vec<Row>,
    by_id: HashMap<MemoryId, usize>,
    probe_width: usize,
    top_k: usize,
    epoch: Epoch,
}

impl PlaintextIvf {
    /// Build from a training sample.
    ///
    /// Uses the same partitioner as `mnemos-index`, seeded identically, so the two indexes
    /// agree on partition boundaries and the recall comparison isolates padding rather than
    /// clustering luck.
    ///
    /// # Errors
    /// [`Error::Params`] if `buckets` is zero.
    pub fn build(
        sample: &[Vec<f32>],
        buckets: usize,
        dim: usize,
        probe_width: usize,
        top_k: usize,
        seed: u64,
    ) -> Result<Self> {
        if buckets == 0 {
            return Err(Error::Params(
                "plaintext IVF needs at least one partition".into(),
            ));
        }
        let centroids = mnemos_index::spherical_kmeans(sample, buckets, dim, seed);
        Ok(Self {
            centroids,
            partitions: vec![Vec::new(); buckets],
            rows: Vec::new(),
            by_id: HashMap::new(),
            probe_width: probe_width.max(1).min(buckets),
            top_k,
            epoch: Epoch(0),
        })
    }

    /// An empty index over an already-trained centroid table.
    ///
    /// [`Self::build`] runs k-means, which on a 10,000-by-768 sample costs minutes. A
    /// write benchmark has to start from an empty structure repeatedly --- otherwise it
    /// accumulates every row it ever inserted, and at criterion's iteration counts that is
    /// millions of rows and gigabytes of resident memory, which stops measuring insertion cost
    /// and starts measuring the allocator under pressure. Splitting training from allocation
    /// lets the benchmark reset cheaply.
    ///
    /// # Errors
    /// [`Error::Params`] if `centroids` is empty.
    pub fn from_centroids(
        centroids: Vec<QuantVec>,
        probe_width: usize,
        top_k: usize,
    ) -> Result<Self> {
        if centroids.is_empty() {
            return Err(Error::Params(
                "plaintext IVF needs at least one partition".into(),
            ));
        }
        let buckets = centroids.len();
        Ok(Self {
            centroids,
            partitions: vec![Vec::new(); buckets],
            rows: Vec::new(),
            by_id: HashMap::new(),
            probe_width: probe_width.max(1).min(buckets),
            top_k,
            epoch: Epoch(0),
        })
    }

    /// The trained centroid table, for handing to [`Self::from_centroids`].
    #[must_use]
    pub fn centroids(&self) -> &[QuantVec] {
        &self.centroids
    }

    /// Partition a vector belongs to. Identical rule to `EncryptedIvf::assign`.
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

    /// Exact nearest neighbour by brute force. **The recall ground truth** for every utility
    /// measurement in the paper.
    ///
    /// # Errors
    /// Never fails; returns `Result` to match the shape of the encrypted paths so a caller can
    /// swap implementations without restructuring.
    pub fn exact_search(&self, q: &QuantVec, top_k: usize) -> Result<Vec<Hit>> {
        let mut scored: Vec<(usize, f32)> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, r)| (i, q.cosine(&r.vector)))
            .collect();
        sort_by_score(&mut scored);
        Ok(scored
            .into_iter()
            .take(top_k)
            .map(|(i, s)| self.hit(i, s))
            .collect())
    }

    /// The ids an exact search returns, for recall computation.
    ///
    /// # Errors
    /// Never fails.
    pub fn exact_ids(&self, q: &QuantVec, top_k: usize) -> Result<Vec<MemoryId>> {
        Ok(self
            .exact_search(q, top_k)?
            .into_iter()
            .map(|h| h.memory.id)
            .collect())
    }

    /// Number of partitions read for a query. **Variable**, unlike Mnemos — reported so the
    /// benchmark can state the I/O gap rather than leave it implicit.
    #[must_use]
    pub fn probes_for(&self, q: &QuantVec) -> Vec<BucketId> {
        let mut scored: Vec<(u32, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i as u32, q.cosine(c)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored
            .into_iter()
            .take(self.probe_width)
            // The optimisation Mnemos is forbidden: an empty partition costs nothing to skip,
            // and skipping it is exactly what makes I/O volume reveal occupancy.
            .filter(|(b, _)| !self.partitions[*b as usize].is_empty())
            .map(|(b, _)| BucketId(b))
            .collect()
    }

    fn hit(&self, i: usize, score: f32) -> Hit {
        let r = &self.rows[i];
        Hit {
            memory: Memory {
                text: r.text.clone(),
                meta: r.meta.clone(),
                id: r.id,
            },
            score,
        }
    }
}

impl SearchableMemory for PlaintextIvf {
    fn put(&mut self, text: &str, meta: &str, embedding: &QuantVec) -> Result<MemoryId> {
        let id = MemoryId::random();
        let b = self.assign(embedding);
        let idx = self.rows.len();
        self.rows.push(Row {
            id,
            text: text.to_owned(),
            meta: meta.to_owned(),
            vector: embedding.clone(),
        });
        // Appended in arrival order. That ordering is readable straight off the structure and
        // is the artifact `docs/adr/0003` removes from Mnemos.
        self.partitions[b.0 as usize].push(idx);
        self.by_id.insert(id, idx);
        Ok(id)
    }

    fn search(&self, query: &QuantVec) -> Result<Vec<Hit>> {
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for b in self.probes_for(query) {
            for idx in &self.partitions[b.0 as usize] {
                scored.push((*idx, query.cosine(&self.rows[*idx].vector)));
            }
        }
        sort_by_score(&mut scored);
        Ok(scored
            .into_iter()
            .take(self.top_k)
            .map(|(i, s)| self.hit(i, s))
            .collect())
    }

    fn update(
        &mut self,
        id: MemoryId,
        text: &str,
        meta: &str,
        embedding: &QuantVec,
    ) -> Result<MemoryId> {
        // In place, keeping the id — the cheap thing Mnemos refuses to do (ADR-0004), because
        // a same-slot rewrite is the correlation signal a live adversary most wants.
        let idx = *self.by_id.get(&id).ok_or(Error::NotFound)?;
        let old_bucket = self.assign(&self.rows[idx].vector);
        let new_bucket = self.assign(embedding);
        text.clone_into(&mut self.rows[idx].text);
        meta.clone_into(&mut self.rows[idx].meta);
        self.rows[idx].vector = embedding.clone();
        if old_bucket != new_bucket {
            self.partitions[old_bucket.0 as usize].retain(|i| *i != idx);
            self.partitions[new_bucket.0 as usize].push(idx);
        }
        Ok(id)
    }

    fn delete(&mut self, id: MemoryId) -> Result<()> {
        let idx = self.by_id.remove(&id).ok_or(Error::NotFound)?;
        let b = self.assign(&self.rows[idx].vector);
        self.partitions[b.0 as usize].retain(|i| *i != idx);
        Ok(())
    }

    fn flush(&mut self) -> Result<Epoch> {
        // No batching: writes were already durable. The epoch counter exists only so the trait
        // is satisfiable, and its uselessness here is the comparison.
        self.epoch = self.epoch.next();
        Ok(self.epoch)
    }

    fn stat(&self) -> Stat {
        Stat {
            live_entries: self.by_id.len() as u64,
            epoch: self.epoch,
            capacity: u64::MAX,
            unlocked: true,
        }
    }

    fn unlock(&mut self) -> Result<()> {
        Ok(())
    }

    fn lock(&mut self) {}
}

/// Descending by score, ties broken by index so the order is total and reproducible.
fn sort_by_score(v: &mut [(usize, f32)]) {
    v.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{corpus, quant};

    #[test]
    fn approximate_search_tracks_exact_search() {
        let (vecs, _) = corpus(400, 32, 8, 5);
        let mut ivf = PlaintextIvf::build(&vecs, 8, 32, 4, 10, 5).unwrap();
        for v in &vecs {
            ivf.put("t", "{}", &quant(v)).unwrap();
        }

        let q = quant(&vecs[0]);
        let approx: Vec<_> = ivf
            .search(&q)
            .unwrap()
            .into_iter()
            .map(|h| h.memory.id)
            .collect();
        let exact = ivf.exact_ids(&q, 10).unwrap();
        let overlap = approx.iter().filter(|i| exact.contains(i)).count();
        assert!(overlap >= 7, "recall@10 was {overlap}/10 against exact NN");
    }

    #[test]
    fn occupancy_is_visible_in_the_probe_set() {
        // The negative control for the whole padding argument: this baseline's read volume
        // DOES depend on occupancy. If it did not, "Mnemos pads and the baseline does not"
        // would be an unmeasurable claim.
        let (vecs, _) = corpus(120, 16, 6, 2);
        let mut ivf = PlaintextIvf::build(&vecs, 32, 16, 8, 10, 2).unwrap();
        let q = quant(&vecs[0]);
        let empty = ivf.probes_for(&q).len();
        for v in &vecs {
            ivf.put("t", "{}", &quant(v)).unwrap();
        }
        let full = ivf.probes_for(&q).len();
        assert!(
            full > empty,
            "probe count did not grow with occupancy ({empty} -> {full}); the baseline is not \
             exhibiting the leak it exists to exhibit"
        );
    }

    #[test]
    fn update_keeps_the_id() {
        // Contrast with Mnemos, where update() returns a NEW id by design (ADR-0004). The
        // difference is a documented API cost paid for a security property.
        let (vecs, _) = corpus(20, 8, 3, 1);
        let mut ivf = PlaintextIvf::build(&vecs, 4, 8, 2, 5, 1).unwrap();
        let id = ivf.put("a", "{}", &quant(&vecs[0])).unwrap();
        let same = ivf.update(id, "b", "{}", &quant(&vecs[1])).unwrap();
        assert_eq!(id, same);
    }
}
