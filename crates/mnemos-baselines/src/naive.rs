//! Encrypt-then-search: AEAD blobs, decrypt everything, scan in the clear.
//!
//! The strawman everyone reaches for first, and it is genuinely worth measuring twice:
//!
//! 1. **It is the security floor.** A1 and A2 should demolish it, which validates that the
//!    attack harness works before those same attacks are pointed at anything subtle. An attack
//!    that fails against Mnemos means nothing unless it demonstrably succeeds somewhere.
//! 2. **It is the honest latency comparison people actually care about.** "Why not just encrypt
//!    the whole store and decrypt on read?" has a numeric answer, and it is O(n) per query.
//!
//! v1 of the spec called this the Stage 0 baseline. The vet's reframing promoted the
//! *forward-private* baselines to that role, because breaking a strawman proves nothing about
//! forward privacy — but this one still earns its place for the two reasons above.
//!
//! ## What it leaks that Mnemos does not
//!
//! Everything structural, and none of it requires a key:
//!
//! - **Entry sizes**, because records are sealed at their natural length rather than padded to
//!   a block. That alone is enough for A1: text length is a strong fingerprint over a known
//!   corpus.
//! - **Insertion order**, from append position.
//! - **Update history**, because an update appends a new record and tombstones the old one, so
//!   the file records how many times each slot was touched.
//!
//! The first is the interesting one. It is the cheapest possible fix — pad to a block — and
//! the fact that a "properly encrypted" store still falls to A1 without it is worth one figure.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use mnemos_core::error::{Error, Result};
use mnemos_core::traits::{SearchableMemory, Stat};
use mnemos_core::types::{Epoch, Hit, Memory, MemoryId, QuantVec};
use mnemos_crypto::aead::{open_content, seal_content};
use mnemos_crypto::kdf::KeyHierarchy;
use mnemos_crypto::prf;

const MAGIC: &[u8; 8] = b"MNEMNAIV";

/// AEAD-per-entry with a full decrypt-and-scan search.
pub struct NaiveEts {
    path: PathBuf,
    file: File,
    /// `(id, offset, length)` per record, in append order. Held in memory; also recoverable
    /// from the file, since the length prefix is plaintext — which is the point.
    entries: Vec<(MemoryId, u64, u32)>,
    tombstoned: Vec<bool>,
    keys: Option<KeyHierarchy>,
    master: [u8; 32],
    top_k: usize,
    epoch: Epoch,
}

impl NaiveEts {
    /// Create a store. Starts **unlocked**, matching the engine's `create`.
    ///
    /// The master key is held in the struct rather than in a keystore: this baseline exists to
    /// be broken, and giving it hardware custody would only obscure which property is doing
    /// the work in the comparison.
    ///
    /// # Errors
    /// I/O failures.
    pub fn create(path: &Path, master: [u8; 32], top_k: usize) -> Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)?;
        file.write_all(MAGIC)?;
        file.flush()?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            entries: Vec::new(),
            tombstoned: Vec::new(),
            keys: Some(KeyHierarchy::derive(&master)?),
            master,
            top_k,
            epoch: Epoch(0),
        })
    }

    /// Record lengths as they appear on disk, **without any key**.
    ///
    /// This is attack A1's entire input. Exposed as a method so the harness reads it the way an
    /// adversary would rather than through private fields — the same discipline as
    /// [`crate::sophos::SophosFp::snapshot_observables`].
    ///
    /// # Errors
    /// I/O or [`Error::Format`] failures.
    pub fn snapshot_lengths(path: &Path) -> Result<Vec<u32>> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(Error::Format("bad naive-EtS magic".into()));
        }
        let mut out = Vec::new();
        loop {
            let mut len = [0u8; 4];
            match f.read_exact(&mut len) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let n = u32::from_le_bytes(len);
            out.push(n);
            f.seek(SeekFrom::Current(i64::from(n)))?;
        }
        Ok(out)
    }

    fn keys(&self) -> Result<&KeyHierarchy> {
        self.keys.as_ref().ok_or(Error::Locked)
    }
}

impl SearchableMemory for NaiveEts {
    fn put(&mut self, text: &str, meta: &str, embedding: &QuantVec) -> Result<MemoryId> {
        let id = MemoryId::random();
        let keys = self.keys()?;
        let tag = prf::tag_for(&keys.tag, id, 0);

        // The vector is stored alongside the text because search has to score it, and there is
        // nowhere else to put it without building an index — which is what makes this the
        // naive scheme.
        let mut plain = Vec::new();
        plain.extend_from_slice(&(text.len() as u32).to_le_bytes());
        plain.extend_from_slice(text.as_bytes());
        plain.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        plain.extend_from_slice(meta.as_bytes());
        plain.extend_from_slice(&embedding.encode());

        // block_size = 1: sealed at natural length, NOT padded. That is the leak, and it is one
        // parameter away from being closed — which is precisely why it is worth measuring.
        let ct = seal_content(&keys.content, &plain, tag, 1)?;

        let off = self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&(ct.len() as u32).to_le_bytes())?;
        self.file.write_all(&ct)?;
        self.file.flush()?;

        self.entries.push((id, off + 4, ct.len() as u32));
        self.tombstoned.push(false);
        Ok(id)
    }

    fn search(&self, query: &QuantVec) -> Result<Vec<Hit>> {
        let keys = self.keys()?;
        let mut f = self.file.try_clone()?;
        let mut scored: Vec<(usize, Hit)> = Vec::new();

        // O(n) decryptions per query. Measure it; do not apologise for it.
        for (i, (id, off, len)) in self.entries.iter().enumerate() {
            if self.tombstoned[i] {
                continue;
            }
            let mut ct = vec![0u8; *len as usize];
            f.seek(SeekFrom::Start(*off))?;
            f.read_exact(&mut ct)?;

            let tag = prf::tag_for(&keys.tag, *id, 0);
            let mut out = vec![0u8; *len as usize];
            let Ok(n) = open_content(&keys.content, &ct, tag, &mut out) else {
                continue;
            };
            let Some((text, meta, vector)) = split_record(&out[..n]) else {
                continue;
            };
            let score = query.cosine(&vector);
            scored.push((
                i,
                Hit {
                    memory: Memory {
                        text,
                        meta,
                        id: *id,
                    },
                    score,
                },
            ));
        }

        scored.sort_by(|a, b| {
            b.1.score
                .partial_cmp(&a.1.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        Ok(scored
            .into_iter()
            .take(self.top_k)
            .map(|(_, h)| h)
            .collect())
    }

    fn update(
        &mut self,
        id: MemoryId,
        text: &str,
        meta: &str,
        embedding: &QuantVec,
    ) -> Result<MemoryId> {
        self.delete(id)?;
        self.put(text, meta, embedding)
    }

    fn delete(&mut self, id: MemoryId) -> Result<()> {
        let i = self
            .entries
            .iter()
            .position(|(e, _, _)| *e == id)
            .ok_or(Error::NotFound)?;
        // A tombstone, not an overwrite. The ciphertext stays on disk, so this baseline has no
        // backward privacy at all — a snapshot after a delete still contains the record.
        self.tombstoned[i] = true;
        Ok(())
    }

    fn flush(&mut self) -> Result<Epoch> {
        self.file.flush()?;
        self.epoch = self.epoch.next();
        Ok(self.epoch)
    }

    fn stat(&self) -> Stat {
        Stat {
            live_entries: self.tombstoned.iter().filter(|t| !**t).count() as u64,
            epoch: self.epoch,
            capacity: u64::MAX,
            unlocked: self.keys.is_some(),
        }
    }

    fn unlock(&mut self) -> Result<()> {
        self.keys = Some(KeyHierarchy::derive(&self.master)?);
        Ok(())
    }

    fn lock(&mut self) {
        self.keys = None;
    }
}

impl std::fmt::Debug for NaiveEts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NaiveEts(path={:?}, {} records)",
            self.path,
            self.entries.len()
        )
    }
}

fn split_record(b: &[u8]) -> Option<(String, String, QuantVec)> {
    if b.len() < 4 {
        return None;
    }
    let tl = u32::from_le_bytes(b[..4].try_into().ok()?) as usize;
    let rest = b.get(4..)?;
    let text = String::from_utf8_lossy(rest.get(..tl)?).into_owned();
    let rest = rest.get(tl..)?;
    let ml = u32::from_le_bytes(rest.get(..4)?.try_into().ok()?) as usize;
    let rest = rest.get(4..)?;
    let meta = String::from_utf8_lossy(rest.get(..ml)?).into_owned();
    let vector = QuantVec::decode(rest.get(ml..)?)?;
    Some((text, meta, vector))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{corpus, quant};

    fn store(dir: &tempfile::TempDir, top_k: usize) -> NaiveEts {
        NaiveEts::create(&dir.path().join("naive.bin"), [3u8; 32], top_k).unwrap()
    }

    #[test]
    fn roundtrips_and_ranks() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = store(&dir, 3);
        let (vecs, _) = corpus(30, 16, 3, 4);
        for (i, v) in vecs.iter().enumerate() {
            s.put(&format!("entry {i}"), "{}", &quant(v)).unwrap();
        }
        let hits = s.search(&quant(&vecs[0])).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(
            hits[0].memory.text, "entry 0",
            "the exact match should rank first"
        );
    }

    #[test]
    fn record_lengths_leak_text_length_without_a_key() {
        // Attack A1's entire input, and the reason this baseline is not merely a strawman:
        // "properly encrypted" plus "unpadded" is still broken.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("naive.bin");
        let mut s = NaiveEts::create(&path, [9u8; 32], 5).unwrap();
        let (vecs, _) = corpus(3, 8, 1, 2);
        s.put("short", "{}", &quant(&vecs[0])).unwrap();
        s.put(&"x".repeat(500), "{}", &quant(&vecs[1])).unwrap();
        s.put(&"y".repeat(2000), "{}", &quant(&vecs[2])).unwrap();

        let lens = NaiveEts::snapshot_lengths(&path).unwrap();
        assert_eq!(lens.len(), 3);
        assert!(
            lens[0] < lens[1] && lens[1] < lens[2],
            "record lengths did not track plaintext lengths ({lens:?}); A1 would have nothing \
             to work with and the baseline would be proving the wrong thing"
        );
    }

    #[test]
    fn a_deleted_record_is_still_on_disk() {
        // No backward privacy whatsoever. Stated as a test so the contrast with Mnemos's
        // immediate overwrite is a measured fact rather than a claim in prose.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("naive.bin");
        let mut s = NaiveEts::create(&path, [1u8; 32], 5).unwrap();
        let (vecs, _) = corpus(1, 8, 1, 1);
        let id = s.put("secret", "{}", &quant(&vecs[0])).unwrap();
        let before = NaiveEts::snapshot_lengths(&path).unwrap();
        s.delete(id).unwrap();
        let after = NaiveEts::snapshot_lengths(&path).unwrap();
        assert_eq!(
            before, after,
            "delete changed the file; this baseline is not supposed to"
        );
    }

    #[test]
    fn locked_store_refuses_to_search() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = store(&dir, 5);
        let (vecs, _) = corpus(2, 8, 1, 1);
        s.put("a", "{}", &quant(&vecs[0])).unwrap();
        s.lock();
        assert!(matches!(s.search(&quant(&vecs[0])), Err(Error::Locked)));
        s.unlock().unwrap();
        assert_eq!(s.search(&quant(&vecs[0])).unwrap().len(), 1);
    }
}
