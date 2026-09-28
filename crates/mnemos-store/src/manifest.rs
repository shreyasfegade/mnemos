//! The manifest — small, encrypted, fixed-size, and **rewritten every epoch whether or not
//! anything changed**.
//!
//! An unconditional rewrite means the file's mtime and its write events carry no information
//! about whether real work happened. A conditional write would turn "the manifest changed" into
//! "something real occurred this epoch", which is a free activity oracle for a watching
//! adversary.
//!
//! # What it holds, and why each field is secret
//!
//! | Field | Why encrypted |
//! |---|---|
//! | `live_entries` | Corpus size. Declared bounded, not public. |
//! | `occupancy` bitmap | Which slots are live — the secret the slab deliberately does not store |
//! | `epoch`, `compaction_cursor` | Activity level over time |
//! | `wrapped_master_key` | The only key material on disk, and useless without the TPM that wrapped it |
//! | `index_occupancy` bitmap | Which *index* slots are live — same argument as the slab bitmap |
//! | `centroids` | The partition table. Reveals the corpus's semantic structure directly. |
//!
//! The occupancy bitmap is the interesting one. `mnemos-store::slab` keeps tags in the clear
//! (they are PRF outputs, unlinkable without the key) but **cannot** store occupancy, because
//! a live/dummy flag would let an adversary count the corpus by inspection. Parking it here,
//! encrypted and fixed-size, is what lets Robin Hood probing work at all while keeping
//! occupancy secret.
//!
//! # Why the centroid table lives here
//!
//! It has to be persisted: recomputing k-means at every `open` costs tens of seconds at
//! production geometry, and — worse — a *different* table would silently reroute every entry's
//! partition, so a store would stop finding its own contents.
//!
//! It has to be **encrypted**: a plaintext centroid table is a description of what the corpus
//! is about, which is most of what an adversary wants and none of what the leakage profile
//! permits.
//!
//! It goes in the manifest rather than in its own file because the store's file *count* and
//! file *names* are themselves observable, and `crates/mnemos-engine/tests/baseline_contrast.rs`
//! asserts the exact footprint. Adding `centroids.enc` would have been a new row in the
//! leakage profile for no benefit — the manifest is already fixed-size, already encrypted, and
//! already rewritten unconditionally every epoch.
//!
//! # Fixed size
//!
//! The serialised body is padded to a constant before sealing, so the manifest's length never
//! varies with `live_entries` or with how much of the bitmap is set.

use std::path::Path;

use mnemos_core::error::{Error, Result};
use mnemos_core::types::Epoch;
use mnemos_crypto::aead::{open_metadata, seal_metadata};
use mnemos_crypto::kdf::DerivedKey;

const MAGIC: &[u8; 8] = b"MNEMOSMF";
const FORMAT_VERSION: u32 = 2;
/// Fixed nonce position and length; the nonce is stored in the clear ahead of the ciphertext.
const NONCE_LEN: usize = 12;

/// Store-level state.
#[derive(Clone, Debug)]
pub struct Manifest {
    /// Format version.
    pub format_version: u32,
    /// Current epoch.
    pub epoch: Epoch,
    /// Declared capacity.
    pub n_max: u64,
    /// Live entries. Secret — this is why the manifest is encrypted rather than plain metadata.
    pub live_entries: u64,
    /// Generation counter for the centroid table, bumped on repartition.
    pub centroid_generation: u32,
    /// Where compaction resumes next epoch.
    pub compaction_cursor: u64,
    /// Which slab slots are live. Secret; see the module docs.
    pub occupancy: Vec<bool>,
    /// Which index slots are live, flattened `bucket * capacity + slot`.
    ///
    /// Without this, reopening a store resets every partition to "empty" and the next flush
    /// overwrites live entries from slot zero — silent data loss that no functional test
    /// catches, because writes keep succeeding and only *older* entries disappear.
    pub index_occupancy: Vec<bool>,
    /// Encoded centroid table, `buckets * QuantVec::encoded_len(dim)` bytes.
    pub centroids: Vec<u8>,
    /// Master key wrapped by the hardware key store.
    pub wrapped_master_key: Vec<u8>,
}

impl Manifest {
    /// A fresh manifest for a newly created store.
    #[must_use]
    pub fn new(
        n_max: u64,
        index_slots: usize,
        centroids: Vec<u8>,
        wrapped_master_key: Vec<u8>,
    ) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            epoch: Epoch(0),
            n_max,
            live_entries: 0,
            centroid_generation: 0,
            compaction_cursor: 0,
            occupancy: vec![false; n_max as usize],
            index_occupancy: vec![false; index_slots],
            centroids,
            wrapped_master_key,
        }
    }

    /// Serialise, pad to a constant length, seal, and replace atomically.
    ///
    /// Called **every epoch, unconditionally.**
    ///
    /// # Errors
    /// I/O or crypto failures.
    pub fn write(&self, path: &Path, key: &DerivedKey) -> Result<()> {
        let body = self.encode();

        // Pad to a constant so the sealed length never varies with content. The bitmap
        // dominates and is already fixed-length, but the wrapped key is provider-dependent, so
        // we pad to a generous ceiling rather than assume.
        let padded_len =
            Self::padded_body_len(self.n_max, self.index_occupancy.len(), self.centroids.len());
        if body.len() > padded_len {
            return Err(Error::Format("manifest body exceeds its fixed size".into()));
        }
        let mut padded = vec![0u8; padded_len];
        padded[..body.len()].copy_from_slice(&body);

        // Fresh random nonce per write. An epoch-derived nonce repeated whenever the epoch
        // counter did: `create` and the first flush both write epoch 0, and `open` restarts the
        // clock at 0, so the same (key, nonce) pair sealed different bitmaps. Under GCM that
        // exposes the XOR of the plaintexts and the authentication key. A random 96-bit nonce
        // also keeps the epoch counter out of the clear header.
        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(&mnemos_crypto::aead::dummy_record(NONCE_LEN));
        let sealed = seal_metadata(key, &padded, &nonce)?;

        let mut out = Vec::with_capacity(8 + NONCE_LEN + sealed.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&sealed);

        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &out)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Read and decrypt.
    ///
    /// # Errors
    /// I/O, [`Error::Format`], or [`Error::Crypto`] on authentication failure.
    pub fn read(path: &Path, key: &DerivedKey) -> Result<Self> {
        let raw = std::fs::read(path)?;
        if raw.len() < 8 + NONCE_LEN {
            return Err(Error::Format("manifest too short".into()));
        }
        if &raw[..8] != MAGIC {
            return Err(Error::Format("bad manifest magic".into()));
        }
        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(&raw[8..8 + NONCE_LEN]);
        let body = open_metadata(key, &raw[8 + NONCE_LEN..], &nonce)?;
        Self::decode(&body)
    }

    /// Sealed manifest size for a given geometry — constant, content-independent.
    ///
    /// Every term is a function of `params`, never of what is stored: two stores with the same
    /// parameters produce byte-identical manifest *lengths* whatever their contents.
    #[must_use]
    pub fn padded_body_len(n_max: u64, index_slots: usize, centroid_bytes: usize) -> usize {
        // fixed scalars + both bitmaps + the centroid table + room for any provider's key
        64 + (n_max as usize).div_ceil(8) + index_slots.div_ceil(8) + centroid_bytes + 1024
    }

    fn encode(&self) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&self.format_version.to_le_bytes());
        v.extend_from_slice(&self.epoch.0.to_le_bytes());
        v.extend_from_slice(&self.n_max.to_le_bytes());
        v.extend_from_slice(&self.live_entries.to_le_bytes());
        v.extend_from_slice(&self.centroid_generation.to_le_bytes());
        v.extend_from_slice(&self.compaction_cursor.to_le_bytes());

        // Bitmap, LSB-first within each byte.
        let mut bits = vec![0u8; self.occupancy.len().div_ceil(8)];
        for (i, live) in self.occupancy.iter().enumerate() {
            if *live {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        v.extend_from_slice(&(bits.len() as u32).to_le_bytes());
        v.extend_from_slice(&bits);

        let mut ibits = vec![0u8; self.index_occupancy.len().div_ceil(8)];
        for (i, live) in self.index_occupancy.iter().enumerate() {
            if *live {
                ibits[i / 8] |= 1 << (i % 8);
            }
        }
        v.extend_from_slice(&(self.index_occupancy.len() as u32).to_le_bytes());
        v.extend_from_slice(&(ibits.len() as u32).to_le_bytes());
        v.extend_from_slice(&ibits);

        v.extend_from_slice(&(self.centroids.len() as u32).to_le_bytes());
        v.extend_from_slice(&self.centroids);

        v.extend_from_slice(&(self.wrapped_master_key.len() as u32).to_le_bytes());
        v.extend_from_slice(&self.wrapped_master_key);
        v
    }

    fn decode(b: &[u8]) -> Result<Self> {
        let bad = || Error::Format("malformed manifest body".into());
        let mut o = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            let s = b.get(o..o + n).ok_or_else(bad)?;
            o += n;
            Ok(s)
        };

        let format_version = u32::from_le_bytes(take(4)?.try_into().map_err(|_| bad())?);
        if format_version != FORMAT_VERSION {
            return Err(Error::Format(format!(
                "manifest format version {format_version}; expected {FORMAT_VERSION}"
            )));
        }
        let epoch = Epoch(u64::from_le_bytes(take(8)?.try_into().map_err(|_| bad())?));
        let n_max = u64::from_le_bytes(take(8)?.try_into().map_err(|_| bad())?);
        let live_entries = u64::from_le_bytes(take(8)?.try_into().map_err(|_| bad())?);
        let centroid_generation = u32::from_le_bytes(take(4)?.try_into().map_err(|_| bad())?);
        let compaction_cursor = u64::from_le_bytes(take(8)?.try_into().map_err(|_| bad())?);

        let nbits = u32::from_le_bytes(take(4)?.try_into().map_err(|_| bad())?) as usize;
        let bits = take(nbits)?.to_vec();
        let mut occupancy = vec![false; n_max as usize];
        for (i, slot) in occupancy.iter_mut().enumerate() {
            *slot = bits
                .get(i / 8)
                .is_some_and(|byte| byte & (1 << (i % 8)) != 0);
        }

        let index_slots = u32::from_le_bytes(take(4)?.try_into().map_err(|_| bad())?) as usize;
        let index_bitmap_len = u32::from_le_bytes(take(4)?.try_into().map_err(|_| bad())?) as usize;
        let index_bits = take(index_bitmap_len)?.to_vec();
        let mut index_occupancy = vec![false; index_slots];
        for (i, slot) in index_occupancy.iter_mut().enumerate() {
            *slot = index_bits
                .get(i / 8)
                .is_some_and(|byte| byte & (1 << (i % 8)) != 0);
        }

        let clen = u32::from_le_bytes(take(4)?.try_into().map_err(|_| bad())?) as usize;
        let centroids = take(clen)?.to_vec();

        let klen = u32::from_le_bytes(take(4)?.try_into().map_err(|_| bad())?) as usize;
        let wrapped_master_key = take(klen)?.to_vec();

        Ok(Self {
            format_version,
            epoch,
            n_max,
            live_entries,
            centroid_generation,
            compaction_cursor,
            occupancy,
            index_occupancy,
            centroids,
            wrapped_master_key,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> DerivedKey {
        DerivedKey([11u8; 32])
    }

    #[test]
    fn the_nonce_is_fresh_even_when_the_epoch_repeats() {
        // `create` and the first flush both write epoch 0, and reopening restarts the epoch
        // clock, so a nonce derived from the epoch would repeat under the same key.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.enc");
        let m = Manifest::new(64, 32, vec![7u8; 96], vec![1, 2, 3, 4]);

        m.write(&path, &key()).unwrap();
        let first = std::fs::read(&path).unwrap();
        m.write(&path, &key()).unwrap();
        let second = std::fs::read(&path).unwrap();

        assert_ne!(
            first[8..8 + NONCE_LEN],
            second[8..8 + NONCE_LEN],
            "manifest nonce repeated"
        );
        assert_eq!(first.len(), second.len());
        assert_eq!(Manifest::read(&path, &key()).unwrap().epoch, Epoch(0));
    }

    #[test]
    fn roundtrip_preserves_every_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.enc");

        let mut m = Manifest::new(64, 32, vec![7u8; 96], vec![1, 2, 3, 4]);
        m.epoch = Epoch(7);
        m.live_entries = 21;
        m.compaction_cursor = 13;
        m.centroid_generation = 2;
        m.occupancy[3] = true;
        m.occupancy[40] = true;
        m.index_occupancy[5] = true;

        m.write(&path, &key()).unwrap();
        let back = Manifest::read(&path, &key()).unwrap();

        assert_eq!(back.epoch, Epoch(7));
        assert_eq!(back.live_entries, 21);
        assert_eq!(back.compaction_cursor, 13);
        assert_eq!(back.centroid_generation, 2);
        assert_eq!(back.wrapped_master_key, vec![1, 2, 3, 4]);
        assert!(back.occupancy[3] && back.occupancy[40]);
        assert!(!back.occupancy[4]);
        assert!(
            back.index_occupancy[5],
            "index occupancy did not survive a round trip"
        );
        assert!(!back.index_occupancy[6]);
        assert_eq!(
            back.centroids,
            vec![7u8; 96],
            "centroid table did not survive a round trip"
        );
    }

    #[test]
    fn size_is_constant_regardless_of_content() {
        // An adversary must not be able to read corpus size off the manifest's length.
        let dir = tempfile::tempdir().unwrap();
        let empty_path = dir.path().join("a.enc");
        let full_path = dir.path().join("b.enc");

        let empty = Manifest::new(256, 64, vec![3u8; 128], vec![9; 64]);
        let mut full = Manifest::new(256, 64, vec![3u8; 128], vec![9; 64]);
        full.live_entries = 200;
        for i in 0..200 {
            full.occupancy[i] = true;
        }
        for i in 0..64 {
            full.index_occupancy[i] = true;
        }

        empty.write(&empty_path, &key()).unwrap();
        full.write(&full_path, &key()).unwrap();

        assert_eq!(
            std::fs::metadata(&empty_path).unwrap().len(),
            std::fs::metadata(&full_path).unwrap().len(),
            "manifest length must not track corpus size"
        );
    }

    #[test]
    fn wrong_key_fails_to_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.enc");
        Manifest::new(32, 8, vec![0u8; 16], vec![1])
            .write(&path, &key())
            .unwrap();
        assert!(Manifest::read(&path, &DerivedKey([99u8; 32])).is_err());
    }

    #[test]
    fn tampering_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.enc");
        Manifest::new(32, 8, vec![0u8; 16], vec![1])
            .write(&path, &key())
            .unwrap();

        let mut raw = std::fs::read(&path).unwrap();
        let n = raw.len();
        raw[n - 1] ^= 0xFF;
        std::fs::write(&path, &raw).unwrap();

        assert!(
            Manifest::read(&path, &key()).is_err(),
            "AEAD must reject a modified manifest"
        );
    }
}
