//! Diana-style forward-private DSSE — attack A3's second victim.
//!
//! Bost–Minaud–Ohrimenko, CCS 2017. GGM-tree puncturable PRF instead of Sophos's trapdoor
//! permutation: symmetric-only and much faster, which is why Mnemos borrows the same primitive
//! (`mnemos-crypto::pprf`).
//!
//! **The leak is different in mechanism and, in a local deployment, strictly larger.** Diana
//! still has to persist a per-keyword counter — it is the leaf index the next write goes to,
//! and without it the store cannot resume after a restart. So Diana leaks *everything Sophos
//! leaks*. On top of that, its search token is a GGM **constrained key** whose node count is
//! `popcount(counter) + 1`; any local implementation caches that key rather than rebuilding an
//! `O(log c)`-node cover on every search, and the cached key's size is a second channel.
//!
//! ### Why "Diana instead of Sophos" is not a fix
//!
//! This is the point of carrying both baselines. A reader's natural first response to Sophos
//! alone is "use Diana, it has no explicit counter". It does have one — the counter moved from
//! being the token to being the leaf index, and stayed on disk either way.
//!
//! What Diana genuinely improves is **cost**: symmetric primitives instead of an RSA trapdoor
//! permutation. It buys forward privacy more cheaply. It buys **no** snapshot resistance,
//! because forward privacy and snapshot resistance are independent axes — which is the paper's
//! thesis, and this module is the cleanest demonstration of it in the tree: same security
//! notion, different construction, identical exposure at time tau.
//!
//! Neither baseline is *broken* by this. Both are being used outside the model they were proved
//! in; see the setting-shift note in [`crate::sophos`].
//!
//! Mnemos's defence is not to avoid GGM trees — it uses one — but to **pad every chain's
//! serialised state to a constant length** (`mnemos_crypto::pprf::TokenChain::serialise`). Same
//! primitive, one additional property, different outcome. That contrast is worth making
//! explicitly in the paper: it shows the fix is cheap and local once you know to look for it.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use mnemos_core::error::{Error, Result};
use mnemos_core::traits::{SearchableMemory, Stat};
use mnemos_core::types::{BucketId, Epoch, Hit, Memory, MemoryId, QuantVec};
use mnemos_crypto::aead::{open_content, seal_content};
use mnemos_crypto::kdf::KeyHierarchy;
use mnemos_crypto::pprf::{eval_from_cover, eval_leaf, prefix_cover, tree_root, NodeKey};
use mnemos_crypto::prf::mac16;

use crate::keyword::Keywords;
use crate::sophos::SnapshotView;

const IDX_MAGIC: &[u8; 8] = b"MNEMDIAN";
const STATE_MAGIC: &[u8; 8] = b"MNEMDIST";

/// Tree depth. Shallower than Mnemos's 32 so the per-keyword state stays small in the
/// baseline; the leak's *shape* is identical at any depth.
const DEPTH: u8 = 16;

/// Forward-private DSSE baseline over a GGM puncturable PRF.
pub struct DianaFp {
    dir: PathBuf,
    index: File,
    /// Per-keyword GGM tree roots. Client-side material: the client can address any leaf.
    roots: HashMap<[u8; 16], [u8; 16]>,
    /// Per-keyword write counters — **the leaf index the next write lands on**.
    ///
    /// This is Sophos's `ST_c` under another name. It has to be persisted for the store to
    /// resume, so a tau-arrival adversary reads it, and Diana's tree does nothing about that.
    counters: HashMap<[u8; 16], u64>,
    /// Cached constrained search keys, covering `[0, counter]` per keyword.
    ///
    /// Persisted *unpadded*, and its node count is `popcount(counter) + 1`. Mnemos pads the
    /// equivalent structure to `TokenChain::serialised_len`; this baseline deliberately writes
    /// only the live prefix, because writing the padded form would make it Mnemos.
    search_keys: HashMap<[u8; 16], Vec<NodeKey>>,
    records: Vec<([u8; 16], u64, u32)>,
    by_label: HashMap<[u8; 16], (u64, u32)>,
    payload: HashMap<[u8; 16], (MemoryId, BucketId)>,
    keywords: Keywords,
    keys: Option<KeyHierarchy>,
    master: [u8; 32],
    top_k: usize,
    epoch: Epoch,
}

impl DianaFp {
    /// Create a new baseline store.
    ///
    /// # Errors
    /// I/O or key-derivation failures.
    pub fn create(dir: &Path, master: [u8; 32], keywords: Keywords, top_k: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let mut index = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(dir.join("diana.idx"))?;
        index.write_all(IDX_MAGIC)?;
        index.flush()?;

        let keys = KeyHierarchy::derive(&master)?;
        let mut roots = HashMap::new();
        let mut counters = HashMap::new();
        let mut search_keys = HashMap::new();
        for w in 0..keywords.len() {
            let bucket = BucketId(w as u32);
            let token = keyword_token(&keys.tag.0, bucket);
            let root = tree_root(&keys.tag, bucket)?;
            search_keys.insert(token, prefix_cover(&root, DEPTH, 0));
            roots.insert(token, root);
            counters.insert(token, 0u64);
        }

        let s = Self {
            dir: dir.to_path_buf(),
            index,
            roots,
            counters,
            search_keys,
            records: Vec::new(),
            by_label: HashMap::new(),
            payload: HashMap::new(),
            keywords,
            keys: Some(keys),
            master,
            top_k,
            epoch: Epoch(0),
        };
        s.persist_state()?;
        Ok(s)
    }

    /// Write the per-keyword state to disk, **unpadded**.
    ///
    /// Two things go out: the counter (identical in kind to Sophos's) and the cached
    /// constrained key, whose node count is the second channel. Mnemos writes
    /// `TokenChain::serialised_len(depth)` bytes for the latter unconditionally; this baseline
    /// writes the live prefix, and that single difference is the whole of the mitigation.
    fn persist_state(&self) -> Result<()> {
        let mut out = Vec::new();
        out.extend_from_slice(STATE_MAGIC);
        out.extend_from_slice(&(self.counters.len() as u32).to_le_bytes());

        let mut kv: Vec<_> = self.counters.iter().collect();
        kv.sort_by_key(|(k, _)| **k);
        for (token, counter) in kv {
            let nodes = self.search_keys.get(token).map_or(0, Vec::len);
            out.extend_from_slice(token);
            out.extend_from_slice(&counter.to_le_bytes());
            out.extend_from_slice(&(nodes as u32).to_le_bytes());
            // The key material itself. Written at its true length — that length IS the leak,
            // and padding it here would be implementing Mnemos inside the baseline.
            if let Some(cover) = self.search_keys.get(token) {
                for n in cover {
                    out.extend_from_slice(&n.key);
                    out.push(n.depth);
                    out.extend_from_slice(&n.index.to_le_bytes());
                }
            }
        }
        std::fs::write(self.dir.join("diana.state"), &out)?;
        Ok(())
    }

    /// The A3 interface: per-keyword update counts, cached-constrained-key sizes, and
    /// insertion order from physical layout.
    ///
    /// `update_counts` carries the **true counter**, because Diana persists it just as Sophos
    /// does — see the module docs. `residual_sizes` carries the cached constrained key's node
    /// count, which is the additional, weaker channel: `popcount(counter) + 1`.
    ///
    /// Reporting both separately keeps the comparison honest. Collapsing them would hide that
    /// the counter channel is shared and only the second one is Diana-specific.
    ///
    /// # Errors
    /// I/O or [`Error::Format`] failures.
    pub fn snapshot_observables(dir: &Path) -> Result<SnapshotView> {
        let state = std::fs::read(dir.join("diana.state"))?;
        if state.len() < 12 || &state[..8] != STATE_MAGIC {
            return Err(Error::Format("bad diana state magic".into()));
        }
        let n = u32::from_le_bytes(state[8..12].try_into().map_err(|_| Error::Crypto)?) as usize;

        let mut residual_sizes = Vec::with_capacity(n);
        let mut update_counts = Vec::with_capacity(n);
        let mut off = 12;
        for _ in 0..n {
            let Some(hdr) = state.get(off..off + 28) else {
                return Err(Error::Format("diana state truncated".into()));
            };
            let handle = u32::from_le_bytes(hdr[..4].try_into().map_err(|_| Error::Crypto)?);
            let counter = u64::from_le_bytes(hdr[16..24].try_into().map_err(|_| Error::Crypto)?);
            let nodes = u32::from_le_bytes(hdr[24..28].try_into().map_err(|_| Error::Crypto)?);
            update_counts.push((handle, counter));
            residual_sizes.push((handle, nodes));
            off += 28 + nodes as usize * 25;
        }

        let mut f = File::open(dir.join("diana.idx"))?;
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != IDX_MAGIC {
            return Err(Error::Format("bad diana index magic".into()));
        }
        let mut insertion_order = Vec::new();
        loop {
            let mut hdr = [0u8; 20];
            match f.read_exact(&mut hdr) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            insertion_order.push(u32::from_le_bytes(
                hdr[..4].try_into().map_err(|_| Error::Crypto)?,
            ));
            let len = u32::from_le_bytes(hdr[16..20].try_into().map_err(|_| Error::Crypto)?);
            f.seek(SeekFrom::Current(i64::from(len)))?;
        }

        let entry_count = insertion_order.len() as u64;
        Ok(SnapshotView {
            update_counts,
            insertion_order,
            residual_sizes,
            entry_count,
        })
    }

    fn keys(&self) -> Result<&KeyHierarchy> {
        self.keys.as_ref().ok_or(Error::Locked)
    }
}

impl SearchableMemory for DianaFp {
    fn put(&mut self, text: &str, meta: &str, embedding: &QuantVec) -> Result<MemoryId> {
        let id = MemoryId::random();
        let w = self.keywords.assign(embedding);

        let token = {
            let keys = self.keys()?;
            keyword_token(&keys.tag.0, w)
        };
        let count = {
            let c = self.counters.entry(token).or_insert(0);
            *c += 1;
            *c
        };

        if count >= (1u64 << DEPTH) {
            return Err(Error::CapacityExceeded {
                live: count,
                n_max: 1 << DEPTH,
            });
        }
        let label = {
            let keys = self.keys()?;
            let root = self.roots.get(&token).ok_or(Error::NotFound)?;
            // Leaf `count`, evaluated from the root the client holds. The previously issued
            // constrained key covers [0, count-1] and therefore cannot derive this label —
            // that is forward privacy, and it is why the cover is refreshed after the write.
            let st = eval_leaf(root, DEPTH, count);
            mac16(&keys.tag.0, &[b"diana/label".as_slice(), &st].concat())
        };

        let ct = {
            let keys = self.keys()?;
            let mut plain = Vec::new();
            plain.extend_from_slice(&id.0);
            plain.extend_from_slice(&(text.len() as u32).to_le_bytes());
            plain.extend_from_slice(text.as_bytes());
            plain.extend_from_slice(&(meta.len() as u32).to_le_bytes());
            plain.extend_from_slice(meta.as_bytes());
            plain.extend_from_slice(&embedding.encode());

            seal_content(&keys.content, &plain, mnemos_core::types::Tag(label), 1)?
        };

        let off = self.index.seek(SeekFrom::End(0))?;
        self.index.write_all(&label)?;
        self.index.write_all(&(ct.len() as u32).to_le_bytes())?;
        self.index.write_all(&ct)?;
        self.index.flush()?;

        self.records.push((label, off + 20, ct.len() as u32));
        self.by_label.insert(label, (off + 20, ct.len() as u32));
        self.payload.insert(label, (id, w));

        if let Some(root) = self.roots.get(&token) {
            self.search_keys
                .insert(token, prefix_cover(root, DEPTH, count));
        }
        Ok(id)
    }

    fn search(&self, query: &QuantVec) -> Result<Vec<Hit>> {
        let keys = self.keys()?;
        let w = self.keywords.assign(query);
        let token = keyword_token(&keys.tag.0, w);
        let latest = self.counters.get(&token).copied().unwrap_or(0);
        let cover = self.search_keys.get(&token).ok_or(Error::NotFound)?;

        let mut hits: Vec<Hit> = Vec::new();
        let mut f = self.index.try_clone()?;
        for c in 1..=latest {
            // Derived from the CONSTRAINED key, not the root: the server's view. A bug that let
            // search reach past the constraint shows up here as a failing test rather than as a
            // silent loss of forward privacy.
            let Some(st) = eval_from_cover(cover, DEPTH, c) else {
                continue;
            };
            let label = mac16(&keys.tag.0, &[b"diana/label".as_slice(), &st].concat());
            let Some((off, len)) = self.by_label.get(&label) else {
                continue;
            };

            let mut ct = vec![0u8; *len as usize];
            f.seek(SeekFrom::Start(*off))?;
            f.read_exact(&mut ct)?;
            let mut out = vec![0u8; *len as usize];
            if let Ok(n) =
                open_content(&keys.content, &ct, mnemos_core::types::Tag(label), &mut out)
            {
                if let Some((id, text, meta, v)) = split_record(&out[..n]) {
                    let score = query.cosine(&v);
                    hits.push(Hit {
                        memory: Memory { text, meta, id },
                        score,
                    });
                }
            }
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(self.top_k);
        Ok(hits)
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
        let label = *self
            .payload
            .iter()
            .find(|(_, (i, _))| *i == id)
            .map(|(l, _)| l)
            .ok_or(Error::NotFound)?;
        self.payload.remove(&label);
        Ok(())
    }

    fn flush(&mut self) -> Result<Epoch> {
        self.index.flush()?;
        self.persist_state()?;
        self.epoch = self.epoch.next();
        Ok(self.epoch)
    }

    fn stat(&self) -> Stat {
        Stat {
            live_entries: self.payload.len() as u64,
            epoch: self.epoch,
            capacity: 1 << DEPTH,
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

fn keyword_token(k_tag: &[u8; 32], w: BucketId) -> [u8; 16] {
    mac16(
        k_tag,
        &[b"diana/kw".as_slice(), &w.0.to_le_bytes()].concat(),
    )
}

fn split_record(b: &[u8]) -> Option<(MemoryId, String, String, QuantVec)> {
    let id = MemoryId(b.get(..16)?.try_into().ok()?);
    let rest = b.get(16..)?;
    let tl = u32::from_le_bytes(rest.get(..4)?.try_into().ok()?) as usize;
    let rest = rest.get(4..)?;
    let text = String::from_utf8_lossy(rest.get(..tl)?).into_owned();
    let rest = rest.get(tl..)?;
    let ml = u32::from_le_bytes(rest.get(..4)?.try_into().ok()?) as usize;
    let rest = rest.get(4..)?;
    let meta = String::from_utf8_lossy(rest.get(..ml)?).into_owned();
    let v = QuantVec::decode(rest.get(ml..)?)?;
    Some((id, text, meta, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{corpus, keywords, quant};

    fn store(dir: &Path) -> DianaFp {
        DianaFp::create(dir, [11u8; 32], keywords(8, 16, 6), 5).unwrap()
    }

    #[test]
    fn stores_and_retrieves() {
        let d = tempfile::tempdir().unwrap();
        let mut s = store(d.path());
        let (vecs, _) = corpus(40, 16, 4, 6);
        for (i, v) in vecs.iter().enumerate() {
            s.put(&format!("e{i}"), "{}", &quant(v)).unwrap();
        }
        let hits = s.search(&quant(&vecs[0])).unwrap();
        assert!(!hits.is_empty(), "a keyword search returned nothing");
        assert_eq!(
            hits[0].memory.text, "e0",
            "the exact match should rank first"
        );
    }

    #[test]
    fn search_survives_many_writes() {
        // Regression. An earlier version punctured the chain on every flush using the *global*
        // epoch as the leaf index, which silently made every write past the flush count
        // underivable — search kept returning results, just fewer of them, which is exactly the
        // kind of failure a happy-path test misses.
        let d = tempfile::tempdir().unwrap();
        let mut s = store(d.path());
        let (vecs, _) = corpus(200, 16, 4, 6);
        for (i, v) in vecs.iter().enumerate() {
            s.put(&format!("e{i}"), "{}", &quant(v)).unwrap();
            if i % 4 == 3 {
                s.flush().unwrap();
            }
        }
        let hits = s.search(&quant(&vecs[0])).unwrap();
        assert_eq!(
            hits[0].memory.text, "e0",
            "the exact match was lost after 200 writes"
        );
    }

    #[test]
    fn a_constrained_key_cannot_derive_the_next_write() {
        // Forward privacy at the scheme level rather than the primitive level: the token that
        // was current before a write must not recognise the record that write produces.
        let d = tempfile::tempdir().unwrap();
        let mut s = store(d.path());
        let (vecs, _) = corpus(30, 16, 4, 6);
        s.put("first", "{}", &quant(&vecs[0])).unwrap();

        let w = s.keywords.assign(&quant(&vecs[0]));
        let token = keyword_token(&s.keys().unwrap().tag.0, w);
        let stale = s.search_keys.get(&token).cloned().unwrap();
        let next = s.counters[&token] + 1;

        assert!(
            eval_from_cover(&stale, DEPTH, next).is_none(),
            "the pre-write constrained key derived the post-write leaf — forward privacy is void"
        );
    }

    #[test]
    fn snapshot_leaks_the_counter_exactly_as_sophos_does() {
        // The corrected claim. Diana's GGM tree replaces Sophos's trapdoor permutation, but the
        // per-keyword counter stays on disk either way — it is the leaf index the next write
        // lands on, and the store cannot resume without it.
        //
        // This is the module's whole point: forward privacy and snapshot resistance are
        // independent axes, and switching construction moves neither.
        let d = tempfile::tempdir().unwrap();
        let mut s = store(d.path());
        let (vecs, _) = corpus(80, 16, 4, 6);
        for v in &vecs {
            s.put("x", "{}", &quant(v)).unwrap();
        }
        s.flush().unwrap();

        let view = DianaFp::snapshot_observables(d.path()).unwrap();
        assert!(
            view.has_per_keyword_signal(),
            "no per-keyword signal on disk"
        );
        assert_eq!(view.entry_count, 80);
        let total: u64 = view.update_counts.iter().map(|(_, c)| c).sum();
        assert_eq!(total, 80, "counters do not account for every write");
        assert!(
            view.update_counts
                .iter()
                .map(|(_, c)| c)
                .collect::<std::collections::HashSet<_>>()
                .len()
                > 1,
            "all counters equal — the corpus is not exercising the partitioning"
        );
    }

    #[test]
    fn the_cached_search_key_is_a_second_channel() {
        // Diana-specific, and weaker than the counter: the cached constrained key's node count
        // is popcount(counter) + 1, so it takes at most DEPTH + 1 values however large the
        // counter grows. Asserted so the comparison reports it as an ADDITIONAL channel rather
        // than as an alternative to the counter.
        let d = tempfile::tempdir().unwrap();
        let mut s = store(d.path());
        let (vecs, _) = corpus(300, 16, 6, 6);
        for v in &vecs {
            s.put("x", "{}", &quant(v)).unwrap();
        }
        s.flush().unwrap();

        let view = DianaFp::snapshot_observables(d.path()).unwrap();
        assert!(!view.residual_sizes.is_empty());
        for (handle, nodes) in &view.residual_sizes {
            let counter = view
                .update_counts
                .iter()
                .find(|(h, _)| h == handle)
                .map_or(0, |(_, c)| *c);
            assert_eq!(
                u64::from(*nodes),
                u64::from(counter.count_ones()) + 1,
                "cached key size did not track popcount(counter)"
            );
        }
        let distinct: std::collections::HashSet<_> =
            view.residual_sizes.iter().map(|(_, r)| *r).collect();
        assert!(
            distinct.len() <= usize::from(DEPTH) + 1,
            "cached-key size took more values than the tree depth permits"
        );
    }

    #[test]
    fn persisted_state_length_varies_because_this_baseline_does_not_pad() {
        // The negative control for `pprf::serialised_length_is_independent_of_update_count`.
        // Mnemos pads; this baseline must not, or the comparison has no contrast in it.
        let d = tempfile::tempdir().unwrap();
        let mut s = store(d.path());
        let (vecs, _) = corpus(400, 16, 6, 6);

        let mut sizes = std::collections::HashSet::new();
        for (i, v) in vecs.iter().enumerate() {
            s.put("x", "{}", &quant(v)).unwrap();
            if i % 50 == 49 {
                s.flush().unwrap();
                sizes.insert(
                    std::fs::metadata(d.path().join("diana.state"))
                        .unwrap()
                        .len(),
                );
            }
        }
        assert!(
            sizes.len() > 1,
            "the unpadded state file never changed size; Diana is not exhibiting the leak \
             Mnemos's padding exists to close"
        );
    }
}
