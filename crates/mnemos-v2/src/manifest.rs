//! The sealed, fixed-size manifest, rewritten at every flush whether or not anything changed.
//!
//! It holds the only mutable metadata: the flush counter, the maintenance cursor, the occupancy
//! bitmap, and the pending tombstones. The partition table never changes and lives in
//! `table.enc`, written once, so the per-flush manifest stays small.

use std::path::Path;

use mnemos_core::error::{Error, Result};

use crate::seal::{open_vec, seal_vec};

const MAGIC: &[u8; 8] = b"MNEMOS2M";
const VERSION: u32 = 1;

/// File name inside the store directory.
pub const NAME: &str = "manifest.enc";

/// Mutable store state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestV2 {
    /// Flushes completed.
    pub epoch: u64,
    /// Next region the maintenance rewrite visits.
    pub cursor: u32,
    /// Live slots, `region * C + slot`.
    pub occupancy: Vec<bool>,
    /// Tags of records deleted but not yet purged.
    pub tombstones: Vec<[u8; 16]>,
}

impl ManifestV2 {
    /// Empty state for a store with `slots` slots.
    #[must_use]
    pub fn new(slots: usize) -> Self {
        Self {
            epoch: 0,
            cursor: 0,
            occupancy: vec![false; slots],
            tombstones: Vec::new(),
        }
    }

    /// Plaintext body length: a function of the geometry alone.
    #[must_use]
    pub const fn body_len(slots: usize, tombstone_capacity: usize) -> usize {
        4 + 8 + 4 + slots.div_ceil(8) + 4 + tombstone_capacity * 16
    }

    /// Serialise, pad, seal, write atomically. Returns bytes written.
    ///
    /// # Errors
    /// I/O or crypto failures, or [`Error::Format`] if tombstones exceed capacity.
    pub fn write(&self, dir: &Path, key: &[u8; 32], cap: usize, durable: bool) -> Result<u64> {
        if self.tombstones.len() > cap {
            return Err(Error::Format("tombstones exceed manifest capacity".into()));
        }
        let mut body = Vec::with_capacity(Self::body_len(self.occupancy.len(), cap));
        body.extend_from_slice(&VERSION.to_le_bytes());
        body.extend_from_slice(&self.epoch.to_le_bytes());
        body.extend_from_slice(&self.cursor.to_le_bytes());
        let mut bits = vec![0u8; self.occupancy.len().div_ceil(8)];
        for (i, live) in self.occupancy.iter().enumerate() {
            if *live {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        body.extend_from_slice(&bits);
        body.extend_from_slice(&(self.tombstones.len() as u32).to_le_bytes());
        for t in &self.tombstones {
            body.extend_from_slice(t);
        }
        body.resize(Self::body_len(self.occupancy.len(), cap), 0);

        let sealed = seal_vec(key, body, MAGIC)?;
        let mut out = Vec::with_capacity(8 + sealed.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&sealed);

        let tmp = dir.join("manifest.tmp");
        {
            use std::io::Write as _;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&out)?;
            if durable {
                f.sync_all()?;
            }
        }
        std::fs::rename(&tmp, dir.join(NAME))?;
        Ok(out.len() as u64)
    }

    /// Read and open.
    ///
    /// # Errors
    /// I/O, crypto, or format failures.
    pub fn read(dir: &Path, key: &[u8; 32], slots: usize, cap: usize) -> Result<Self> {
        let raw = std::fs::read(dir.join(NAME))?;
        if raw.len() < 8 || &raw[..8] != MAGIC {
            return Err(Error::Format("bad manifest magic".into()));
        }
        let body = open_vec(key, &raw[8..], MAGIC)?;
        if body.len() != Self::body_len(slots, cap) {
            return Err(Error::Format("manifest size does not match params".into()));
        }
        let bad = || Error::Format("malformed manifest".into());
        let u32_at = |o: usize| -> Result<u32> {
            Ok(u32::from_le_bytes(
                body.get(o..o + 4)
                    .ok_or_else(bad)?
                    .try_into()
                    .map_err(|_| bad())?,
            ))
        };
        if u32_at(0)? != VERSION {
            return Err(Error::Format("manifest version".into()));
        }
        let epoch = u64::from_le_bytes(body[4..12].try_into().map_err(|_| bad())?);
        let cursor = u32_at(12)?;
        let bitlen = slots.div_ceil(8);
        let bits = &body[16..16 + bitlen];
        let occupancy = (0..slots)
            .map(|i| bits[i / 8] & (1 << (i % 8)) != 0)
            .collect();
        let n = u32_at(16 + bitlen)? as usize;
        if n > cap {
            return Err(bad());
        }
        let base = 20 + bitlen;
        let tombstones = (0..n)
            .map(|i| {
                let mut t = [0u8; 16];
                t.copy_from_slice(&body[base + i * 16..base + (i + 1) * 16]);
                t
            })
            .collect();
        Ok(Self {
            epoch,
            cursor,
            occupancy,
            tombstones,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_constant_size() {
        let dir = tempfile::tempdir().unwrap();
        let key = [3u8; 32];
        let mut m = ManifestV2::new(40);
        m.write(dir.path(), &key, 8, false).unwrap();
        let empty_len = std::fs::metadata(dir.path().join(NAME)).unwrap().len();
        m.epoch = 9;
        m.cursor = 3;
        m.occupancy[5] = true;
        m.occupancy[39] = true;
        m.tombstones = vec![[1u8; 16], [2u8; 16], [3u8; 16]];
        m.write(dir.path(), &key, 8, false).unwrap();
        assert_eq!(
            std::fs::metadata(dir.path().join(NAME)).unwrap().len(),
            empty_len
        );
        assert_eq!(ManifestV2::read(dir.path(), &key, 40, 8).unwrap(), m);
        assert!(ManifestV2::read(dir.path(), &[4u8; 32], 40, 8).is_err());
    }
}
