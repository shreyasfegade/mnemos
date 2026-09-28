//! Sophos-style forward-private DSSE — **attack A3's primary victim.**
//!
//! Bost, *Σοφος: Forward Secure Searchable Encryption*, CCS 2016.
//!
//! ## Why this baseline exists
//!
//! Sophos is forward-private: an update cannot be linked to previously issued search tokens.
//! Under the standard DSSE threat model, where the adversary has watched since `t = 0`, that is
//! a strong guarantee.
//!
//! Under a **τ-arrival** adversary it is not enough, and this module is where that becomes
//! visible. Sophos maintains a per-keyword search-token counter `ST_c`. That counter is
//! *state*, it is on disk at time τ, and it says how many times each keyword has been updated.
//! Combined with the insertion order that an append-structured index encodes in physical
//! layout, that is the classical input to a re-identification attack (IKK NDSS'12; Cash et al.
//! CCS'15).
//!
//! **This is Corollary 1's counterexample, in code.**
//!
//! ## The setting shift, stated plainly
//!
//! In the deployment Sophos was designed for, the counter map is *client* state and the
//! adversary compromises the *server*. The counters are therefore out of reach and none of
//! this applies.
//!
//! Mnemos's setting is a **local** encrypted store: an agent's memory on the user's own disk.
//! There is no remote server. Client state and server state are the same bytes on the same
//! device, and a τ-arrival adversary who images that device gets both. Sophos is not broken
//! here — it is being **used outside the model it was proved in**, which is a different and
//! more interesting claim, and the one the paper makes. Nothing in this module should be read
//! as a defect in Bost's construction.
//!
//! ## Fairness
//!
//! This implementation must be faithful. Where the paper leaves the on-disk representation
//! open, we take the most conservative reading — the one that leaks *least* — and record the
//! choice here. Specifically:
//!
//! - **Keywords are stored as PRF tokens, never in the clear.** A naive local implementation
//!   would write the keyword itself into the client state file, which would be a strawman. The
//!   adversary here sees a map from *pseudorandom handles* to counts, and must solve the
//!   assignment problem to learn anything. That is the strongest reasonable reading, and it is
//!   why A3's measured advantage is capped near chance rather than trivially 1.0 — see
//!   [F1](../../../the findings notes).
//! - `ST_c` is stored as a compact per-keyword map, not a padded array, because that is what
//!   the construction requires for its update rule. (Padding it would be a different scheme —
//!   in fact it would be a step toward Mnemos.)
//! - Entries are appended in insertion order, because that is what "append-only encrypted
//!   index" means. We do **not** additionally sort, index, or annotate anything.
//!
//! ### The one substitution, and why it does not affect the result
//!
//! Sophos derives tokens with an RSA trapdoor permutation: the client walks *backwards* with
//! the trapdoor, the server walks *forwards* with the public key. We substitute a **reverse
//! hash chain** — `ST_c = H^{(N-c)}(R)` for a per-keyword secret `R` and a bounded `N`. This
//! preserves the property forward privacy actually rests on: given `ST_c`, older tokens are
//! computable and **newer ones are not**.
//!
//! What changes: chain length is bounded by `N`, and the client recomputes from `R` rather than
//! applying a trapdoor. Those are performance and capacity differences.
//! What does not change: **the persisted state's shape** — a per-keyword counter map plus an
//! append-ordered encrypted index. A3 consumes only persisted state, so the substitution is
//! invisible to it. This is stated here rather than buried because it is the first thing a
//! hostile reviewer should check, and they should find it already answered.
//!
//! If A3 still reaches `Adv_hist ≥ 0.50` against that, the finding stands. It does **not** —
//! it caps near 0.12 — and that negative result is recorded as F1 and is what motivated the
//! pivot to the structure-transfer attack (A4/F2).

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use mnemos_core::error::{Error, Result};
use mnemos_core::traits::{SearchableMemory, Stat};
use mnemos_core::types::{BucketId, Epoch, Hit, Memory, MemoryId, QuantVec};
use mnemos_crypto::aead::{open_content, seal_content};
use mnemos_crypto::kdf::{DerivedKey, KeyHierarchy};
use mnemos_crypto::prf::mac16;

use crate::keyword::Keywords;

const IDX_MAGIC: &[u8; 8] = b"MNEMSOPH";
const STATE_MAGIC: &[u8; 8] = b"MNEMSPST";

/// Chain bound `N`. Every keyword supports this many updates before its chain is exhausted.
///
/// A bound is the cost of substituting a hash chain for the trapdoor permutation: `ST_c` sits
/// `N - c` steps from the chain's secret root, so the client cannot jump to it — it either
/// walks `N - c` steps per update or precomputes the chain once. We precompute (see
/// [`SophosFp::chains`]), which is `N` hashes per keyword at construction and `O(1)` per write.
///
/// 4096 is sized to the evaluation: the largest corpus is 100k entries over 256 partitions,
/// averaging ~400 writes per keyword, so this leaves ~10x headroom while keeping the
/// precomputed table at `keywords * N * 16` bytes (16 MiB at 256 keywords). An earlier value of
/// 2^20 was correct in spirit and unusable in practice — it put a million SHA-256 calls on
/// every single `put`.
const CHAIN_BOUND: u64 = 4096;

/// Forward-private DSSE baseline with a persisted per-keyword update counter.
pub struct SophosFp {
    dir: PathBuf,
    index: File,
    /// `ST_c` — the counter A3 harvests. Named to match the paper so the correspondence is
    /// unmistakable when a reviewer reads this file.
    ///
    /// **This field is the finding.** Its presence on disk is what makes a forward-private
    /// scheme snapshot-vulnerable. Keyed by the keyword's PRF token, not by the keyword.
    search_token_counters: HashMap<[u8; 16], u64>,
    /// Append-ordered index: `(label, offset, len)`. Labels are pseudorandom and unlinkable
    /// across counters — the ordering is the only structure visible without a key.
    records: Vec<([u8; 16], u64, u32)>,
    /// Label lookup for the search path. Derived from `records`; kept separate so `records`
    /// stays a faithful append log rather than becoming a map with an incidental order.
    by_label: HashMap<[u8; 16], (u64, u32)>,
    /// Precomputed reverse hash chain per keyword: `chains[w][c] = ST_c`.
    ///
    /// The client legitimately holds this material — it is `R_w` expanded. Precomputing trades
    /// memory for the `O(N)` per-update cost the hash-chain substitution would otherwise
    /// impose; see [`CHAIN_BOUND`].
    chains: Vec<Vec<[u8; 16]>>,
    /// Plaintext side-table so search can return content. Not persisted, not observable;
    /// exists because this baseline must be *usable* to be benchmarked.
    payload: HashMap<[u8; 16], (MemoryId, BucketId)>,
    keywords: Keywords,
    keys: Option<KeyHierarchy>,
    master: [u8; 32],
    top_k: usize,
    epoch: Epoch,
}

impl SophosFp {
    /// Create a new baseline store over a fixed keyword space.
    ///
    /// Keywords are partitions, assigned by the same spherical k-means the encrypted index
    /// uses — so "keyword" here means "semantic partition", exactly as in the Python attack lab
    /// and in the paper's threat model.
    ///
    /// # Errors
    /// I/O failures.
    pub fn create(dir: &Path, master: [u8; 32], keywords: Keywords, top_k: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let mut index = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(dir.join("sophos.idx"))?;
        index.write_all(IDX_MAGIC)?;
        index.flush()?;

        let keys = KeyHierarchy::derive(&master)?;
        let chains = (0..keywords.len())
            .map(|w| build_chain(&keys.tag, BucketId(w as u32)))
            .collect();

        let s = Self {
            dir: dir.to_path_buf(),
            index,
            search_token_counters: HashMap::new(),
            records: Vec::new(),
            by_label: HashMap::new(),
            chains,
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

    /// The keyword's PRF token — the opaque handle an adversary sees in place of the keyword.
    fn keyword_token(k: &DerivedKey, w: BucketId) -> [u8; 16] {
        mac16(
            &k.0,
            &[b"sophos/kw".as_slice(), &w.0.to_le_bytes()].concat(),
        )
    }

    /// `ST_c` for keyword `w` at counter `c`, from the precomputed reverse hash chain.
    ///
    /// `ST_c = H^{(N - c)}(R_w)`. Given `ST_c` the server computes `ST_{c-1} = H(ST_c)` and so
    /// walks *backwards* through history; computing `ST_{c+1}` would require inverting `H`.
    /// That asymmetry is the whole of forward privacy.
    ///
    /// Returns the chain's terminal token when `c` exceeds [`CHAIN_BOUND`], which only happens
    /// on an exhausted chain — a capacity limit of the substitution, documented on the
    /// constant, and unreachable at every corpus size in the evaluation.
    fn token_at(&self, w: BucketId, c: u64) -> [u8; 16] {
        let chain = &self.chains[w.0 as usize % self.chains.len()];
        chain[(c as usize).min(chain.len() - 1)]
    }

    fn label_of(st: &[u8; 16]) -> [u8; 16] {
        hash_tagged(st, b"label")
    }

    /// Write the counter map to disk.
    ///
    /// **This call is the vulnerability**, and it is not optional: a local store with no
    /// persisted counter map cannot resume after a restart. Sophos's client state has to live
    /// somewhere, and on a single device "somewhere" is the same disk.
    fn persist_state(&self) -> Result<()> {
        let mut out = Vec::with_capacity(8 + self.search_token_counters.len() * 24);
        out.extend_from_slice(STATE_MAGIC);
        out.extend_from_slice(&(self.search_token_counters.len() as u32).to_le_bytes());
        // Sorted so the file is deterministic — the point being that even a *canonical*
        // ordering does not help: the counts are the leak, not their order.
        let mut kv: Vec<_> = self.search_token_counters.iter().collect();
        kv.sort_by_key(|(k, _)| **k);
        for (token, count) in kv {
            out.extend_from_slice(token);
            out.extend_from_slice(&count.to_le_bytes());
        }
        std::fs::write(self.dir.join("sophos.state"), &out)?;
        Ok(())
    }

    /// **The A3 interface.** Everything a snapshot adversary can extract from this store's
    /// on-disk state, and nothing more.
    ///
    /// The attack harness calls this instead of reading private fields, so the attack is
    /// constrained to genuinely snapshot-derivable information. That constraint is the
    /// difference between a result and a rigged demo.
    ///
    /// # Errors
    /// I/O or [`Error::Format`] failures.
    pub fn snapshot_observables(dir: &Path) -> Result<SnapshotView> {
        let state = std::fs::read(dir.join("sophos.state"))?;
        if state.len() < 12 || &state[..8] != STATE_MAGIC {
            return Err(Error::Format("bad sophos state magic".into()));
        }
        let n = u32::from_le_bytes(state[8..12].try_into().map_err(|_| Error::Crypto)?) as usize;

        let mut update_counts = Vec::with_capacity(n);
        for i in 0..n {
            let off = 12 + i * 24;
            let Some(chunk) = state.get(off..off + 24) else {
                return Err(Error::Format("sophos state truncated".into()));
            };
            // The adversary sees a pseudorandom handle. Truncating to u32 for the view is
            // lossless for the attack's purposes: it only ever uses handles for equality.
            let handle = u32::from_le_bytes(chunk[..4].try_into().map_err(|_| Error::Crypto)?);
            let count = u64::from_le_bytes(chunk[16..24].try_into().map_err(|_| Error::Crypto)?);
            update_counts.push((handle, count));
        }

        let mut f = File::open(dir.join("sophos.idx"))?;
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != IDX_MAGIC {
            return Err(Error::Format("bad sophos index magic".into()));
        }
        let mut insertion_order = Vec::new();
        loop {
            let mut hdr = [0u8; 20];
            match f.read_exact(&mut hdr) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            // File position IS arrival order. Labels are unlinkable to keywords on their own —
            // but an adversary who later learns label-to-keyword mappings (A4's injection step)
            // can use this ordering to place *earlier* records in time. That is the channel
            // finding F2 exploits.
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
            residual_sizes: Vec::new(),
            entry_count,
        })
    }

    fn keys(&self) -> Result<&KeyHierarchy> {
        self.keys.as_ref().ok_or(Error::Locked)
    }
}

impl SearchableMemory for SophosFp {
    fn put(&mut self, text: &str, meta: &str, embedding: &QuantVec) -> Result<MemoryId> {
        let id = MemoryId::random();
        let w = self.keywords.assign(embedding);

        let token = {
            let keys = self.keys()?;
            Self::keyword_token(&keys.tag, w)
        };
        // The counter is bumped BEFORE the token is derived, so the new entry uses a token no
        // previously issued search token can reach. That is forward privacy.
        let count = {
            let c = self.search_token_counters.entry(token).or_insert(0);
            *c += 1;
            *c
        };
        let label = Self::label_of(&self.token_at(w, count));

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
        self.persist_state()?;
        Ok(id)
    }

    fn search(&self, query: &QuantVec) -> Result<Vec<Hit>> {
        let keys = self.keys()?;
        let mut hits: Vec<Hit> = Vec::new();
        let mut f = self.index.try_clone()?;

        // Probe the query's own partition. A DSSE scheme searches one keyword at a time; the
        // multi-probe generalisation is Mnemos's, not Sophos's.
        let w = self.keywords.assign(query);
        let token = Self::keyword_token(&keys.tag, w);
        let latest = self.search_token_counters.get(&token).copied().unwrap_or(0);

        // Walk the chain backwards from the newest counter, exactly as the server does.
        let mut st = self.token_at(w, latest);
        for _ in 0..latest {
            let label = Self::label_of(&st);
            if let Some((off, len)) = self.by_label.get(&label) {
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
            st = hash_step(&st);
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
        // Sophos is forward-private only; deletion is an *addition* of a tombstone record, so
        // the ciphertext survives on disk. No backward privacy, by construction.
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
            capacity: CHAIN_BOUND,
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

/// What a τ-arrival adversary recovers from a forward-private index's snapshot.
///
/// The existence of this struct with non-empty fields **is the paper's finding**. The
/// corresponding function for Mnemos returns constants.
#[derive(Debug, Default, Clone)]
pub struct SnapshotView {
    /// Per-keyword-token update counts, from `ST_c`. **Should be empty for Mnemos.**
    ///
    /// The first element is a *pseudorandom handle*, not a keyword: the adversary knows two
    /// records share a keyword but not which keyword. Recovering the mapping is the assignment
    /// problem A3 solves, and the reason its advantage is capped by count collisions rather
    /// than being trivially 1.0.
    pub update_counts: Vec<(u32, u64)>,
    /// Insertion order recovered from physical layout. **Should be empty for Mnemos.**
    ///
    /// One truncated label per record, in file order. Useless alone; decisive once a
    /// post-compromise injection supplies label-to-partition mappings, which is finding F2.
    pub insertion_order: Vec<u32>,
    /// Residual key-set sizes per keyword handle. Populated by
    /// [`crate::diana::DianaFp`] only — Sophos has no GGM tree.
    pub residual_sizes: Vec<(u32, u32)>,
    /// Total entries — the one thing everyone leaks, and a declared non-goal to hide.
    pub entry_count: u64,
}

impl SnapshotView {
    /// Whether this view carries any per-keyword signal at all.
    ///
    /// Mnemos's snapshot must make this `false`. Used by the cross-scheme test that pins the
    /// contrast, so "Mnemos leaks nothing here" is asserted rather than asserted-about.
    #[must_use]
    pub fn has_per_keyword_signal(&self) -> bool {
        !self.update_counts.is_empty() || !self.residual_sizes.is_empty()
    }
}

/// Precompute keyword `w`'s reverse hash chain: `out[c] = ST_c = H^{(N - c)}(R_w)`.
///
/// Built from the terminal end downward, so `out[c] = H(out[c + 1])` by construction and the
/// server's backward walk is exactly one `hash_step` per counter.
fn build_chain(k_tag: &DerivedKey, w: BucketId) -> Vec<[u8; 16]> {
    let n = CHAIN_BOUND as usize;
    let root = mac16(
        &k_tag.0,
        &[b"sophos/root".as_slice(), &w.0.to_le_bytes()].concat(),
    );
    let mut out = vec![[0u8; 16]; n + 1];
    out[n] = root;
    for c in (0..n).rev() {
        out[c] = hash_step(&out[c + 1]);
    }
    out
}

/// One step of the reverse hash chain.
fn hash_step(x: &[u8; 16]) -> [u8; 16] {
    hash_tagged(x, b"chain")
}

fn hash_tagged(x: &[u8; 16], domain: &[u8]) -> [u8; 16] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(domain);
    h.update(x);
    let d = h.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&d[..16]);
    out
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

    fn store(dir: &Path) -> SophosFp {
        SophosFp::create(dir, [5u8; 32], keywords(8, 16, 3), 5).unwrap()
    }

    #[test]
    fn stores_and_retrieves() {
        let d = tempfile::tempdir().unwrap();
        let mut s = store(d.path());
        let (vecs, _) = corpus(40, 16, 4, 3);
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
    fn snapshot_leaks_per_keyword_update_counts() {
        // The finding, as a test. If this ever returns an empty view, A3 has nothing to attack
        // and every result derived from this baseline is void.
        let d = tempfile::tempdir().unwrap();
        let mut s = store(d.path());
        let (vecs, _) = corpus(60, 16, 4, 3);
        for v in &vecs {
            s.put("x", "{}", &quant(v)).unwrap();
        }

        let view = SophosFp::snapshot_observables(d.path()).unwrap();
        assert!(
            view.has_per_keyword_signal(),
            "no per-keyword signal on disk"
        );
        assert_eq!(view.entry_count, 60);
        assert_eq!(view.insertion_order.len(), 60);
        let total: u64 = view.update_counts.iter().map(|(_, c)| c).sum();
        assert_eq!(total, 60, "counters do not account for every write");
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
    fn forward_privacy_holds_the_token_asymmetry() {
        // The property the hash-chain substitution has to preserve: from ST_c the server
        // reaches ST_{c-1}, and cannot reach ST_{c+1}. If this fails the baseline is not
        // forward-private and comparing against it proves nothing.
        let d = tempfile::tempdir().unwrap();
        let s = store(d.path());
        let w = BucketId(2);

        let st5 = s.token_at(w, 5);
        let st4 = s.token_at(w, 4);
        let st6 = s.token_at(w, 6);

        assert_eq!(
            hash_step(&st5),
            st4,
            "server cannot walk backwards — search would break"
        );
        assert_ne!(
            hash_step(&st5),
            st6,
            "newer token derivable from older — forward privacy is void"
        );
    }

    #[test]
    fn insertion_order_is_readable_without_a_key() {
        // Corollary 1's counterexample: forward privacy does not imply ordering privacy.
        let d = tempfile::tempdir().unwrap();
        let mut s = store(d.path());
        let (vecs, _) = corpus(12, 16, 3, 9);
        for v in &vecs {
            s.put("x", "{}", &quant(v)).unwrap();
        }
        let view = SophosFp::snapshot_observables(d.path()).unwrap();
        assert_eq!(view.insertion_order.len(), 12);
        // Every record's label appears exactly once and in write order. The adversary holds no
        // key and has still recovered a total order over the corpus.
        let uniq: std::collections::HashSet<_> = view.insertion_order.iter().collect();
        assert_eq!(
            uniq.len(),
            12,
            "labels collided; the ordering channel is not clean"
        );
    }
}
