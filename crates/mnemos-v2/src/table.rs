//! The partition table and probe groups, sealed and written once at creation, plus the keyed
//! permutation from logical partitions to physical regions.
//!
//! # Why a permutation and not a rotation
//!
//! If the partition table is public (trained on public text), an observer can embed its own
//! documents, find their nearest centroid, and so learn which partition holds which topic
//! without inducing a single write. Rotating the centroids and every stored vector by a secret
//! orthogonal matrix does not prevent this: a rotation preserves inner products, so the nearest
//! centroid of a rotated vector is the rotation of its nearest centroid and the partition index
//! is unchanged. What the observer must not learn is where partition `i` lives on disk, so the
//! secret is a permutation of physical positions.

use std::path::Path;

use mnemos_core::error::{Error, Result};
use mnemos_core::types::QuantVec;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::seal::{open_vec, seal_vec};

const MAGIC: &[u8; 8] = b"MNEMOS2T";

/// File name inside the store directory.
pub const NAME: &str = "table.enc";

/// Logical partition table.
#[derive(Clone)]
pub struct Table {
    /// Centroid of each logical partition.
    pub centroids: Vec<QuantVec>,
    /// Probe group of each logical partition.
    pub groups: Vec<u32>,
}

impl Table {
    /// Seal and write.
    ///
    /// # Errors
    /// I/O or crypto failures.
    pub fn write(&self, dir: &Path, key: &[u8; 32]) -> Result<u64> {
        let dim = self.centroids.first().map_or(0, |c| c.codes.len());
        let mut body = Vec::new();
        body.extend_from_slice(&(self.centroids.len() as u32).to_le_bytes());
        body.extend_from_slice(&(dim as u32).to_le_bytes());
        for c in &self.centroids {
            body.extend_from_slice(&c.encode());
        }
        for g in &self.groups {
            body.extend_from_slice(&g.to_le_bytes());
        }
        let sealed = seal_vec(key, body, MAGIC)?;
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&sealed);
        std::fs::write(dir.join(NAME), &out)?;
        Ok(out.len() as u64)
    }

    /// Read and open.
    ///
    /// # Errors
    /// I/O, crypto or format failures.
    pub fn read(dir: &Path, key: &[u8; 32]) -> Result<Self> {
        let raw = std::fs::read(dir.join(NAME))?;
        if raw.len() < 8 || &raw[..8] != MAGIC {
            return Err(Error::Format("bad table magic".into()));
        }
        let body = open_vec(key, &raw[8..], MAGIC)?;
        let bad = || Error::Format("malformed table".into());
        let b = u32::from_le_bytes(
            body.get(0..4)
                .ok_or_else(bad)?
                .try_into()
                .map_err(|_| bad())?,
        ) as usize;
        let dim = u32::from_le_bytes(
            body.get(4..8)
                .ok_or_else(bad)?
                .try_into()
                .map_err(|_| bad())?,
        ) as usize;
        let w = QuantVec::encoded_len(dim);
        let mut o = 8;
        let mut centroids = Vec::with_capacity(b);
        for _ in 0..b {
            centroids.push(QuantVec::decode(body.get(o..o + w).ok_or_else(bad)?).ok_or_else(bad)?);
            o += w;
        }
        let mut groups = Vec::with_capacity(b);
        for _ in 0..b {
            groups.push(u32::from_le_bytes(
                body.get(o..o + 4)
                    .ok_or_else(bad)?
                    .try_into()
                    .map_err(|_| bad())?,
            ));
            o += 4;
        }
        Ok(Self { centroids, groups })
    }

    /// Nearest logical partition, ties broken by lower index.
    #[must_use]
    pub fn assign(&self, v: &QuantVec) -> u32 {
        self.ranked(v).first().map_or(0, |(i, _)| *i)
    }

    /// All logical partitions by descending similarity, ties by index.
    #[must_use]
    pub fn ranked(&self, v: &QuantVec) -> Vec<(u32, f32)> {
        let mut s: Vec<(u32, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i as u32, v.cosine(c)))
            .collect();
        s.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        s
    }
}

/// Balanced grouping of partitions by centroid similarity, deterministic.
///
/// The lowest-index ungrouped partition seeds a group, which takes its `g - 1` most similar
/// ungrouped partitions. Requires `g` to divide `B`.
#[must_use]
pub fn build_groups(centroids: &[QuantVec], g: usize) -> Vec<u32> {
    let b = centroids.len();
    let mut groups = vec![u32::MAX; b];
    if g == 0 || b % g != 0 {
        return (0..b as u32).collect();
    }
    let mut next = 0u32;
    for seed in 0..b {
        if groups[seed] != u32::MAX {
            continue;
        }
        groups[seed] = next;
        let mut cand: Vec<(usize, f32)> = (0..b)
            .filter(|j| groups[*j] == u32::MAX)
            .map(|j| (j, centroids[seed].cosine(&centroids[j])))
            .collect();
        cand.sort_by(|a, c| {
            c.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&c.0))
        });
        for (j, _) in cand.into_iter().take(g - 1) {
            groups[j] = next;
        }
        next += 1;
    }
    groups
}

/// Keyed permutation: `region_of[logical]`. Identity when `identity` is set.
#[must_use]
pub fn permutation(key: &[u8; 32], buckets: u32, identity: bool) -> Vec<u32> {
    let mut p: Vec<u32> = (0..buckets).collect();
    if !identity {
        let mut rng = ChaCha20Rng::from_seed(*key);
        p.shuffle(&mut rng);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnemos_core::types::Embedding;

    fn qv(i: usize, dim: usize) -> QuantVec {
        let mut v = vec![0.01f32; dim];
        v[i % dim] = 1.0;
        QuantVec::quantise(&Embedding(v))
    }

    #[test]
    fn groups_are_balanced() {
        let cents: Vec<QuantVec> = (0..16).map(|i| qv(i, 8)).collect();
        let g = build_groups(&cents, 4);
        for id in 0..4u32 {
            assert_eq!(g.iter().filter(|x| **x == id).count(), 4);
        }
    }

    #[test]
    fn permutation_is_keyed_and_bijective() {
        let a = permutation(&[1u8; 32], 64, false);
        let b = permutation(&[2u8; 32], 64, false);
        assert_ne!(a, b);
        let mut s = a.clone();
        s.sort_unstable();
        assert_eq!(s, (0..64).collect::<Vec<_>>());
        assert_eq!(
            permutation(&[1u8; 32], 64, true),
            (0..64).collect::<Vec<_>>()
        );
    }

    #[test]
    fn roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cents: Vec<QuantVec> = (0..8).map(|i| qv(i, 8)).collect();
        let t = Table {
            groups: build_groups(&cents, 2),
            centroids: cents,
        };
        t.write(dir.path(), &[5u8; 32]).unwrap();
        let back = Table::read(dir.path(), &[5u8; 32]).unwrap();
        assert_eq!(back.groups, t.groups);
        assert_eq!(back.centroids.len(), 8);
        assert_eq!(back.centroids[3].encode(), t.centroids[3].encode());
    }
}
