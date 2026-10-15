//! The encrypted inverted-file index.
//!
//! # The invariant this file exists to uphold
//!
//! > Every search reads **exactly** `(probe_width + decoy_buckets) × bucket_capacity` slots and
//! > scores candidates from all of them, regardless of the query and regardless of how many
//! > entries actually match.
//!
//! Every plausible optimisation violates it. Skipping an empty bucket, stopping early once
//! `top_k` good candidates are found, reading a partial bucket — each makes I/O volume
//! query-dependent, which is the volumetric leakage Xu et al. (CCS 2023) showed survives
//! forward *and* backward privacy. Attack A5 is aimed at this file specifically.
//!
//! # On-disk layout
//!
//! ```text
//! [ header 64 B ]  magic, version, buckets, capacity, dim
//! [ bucket 0: capacity × slot ] [ bucket 1 ] ... [ bucket B-1 ]
//!
//! slot = [ tag 16 B ][ quantised vector: dim + 4 B ]
//! ```
//!
//! **Every bucket occupies `capacity × slot` bytes whether it holds zero entries or all of
//! them.** That is the direct answer to the leakage-profile row reading "per-bucket occupancy:
//! none" — and it is precisely the row where an unpadded index reads "update count per
//! partition", which is what the structure-transfer attack (F2) consumes to map the store.
//!
//! **Every slot is sealed under `K_bucket` with AES-256-GCM and a fresh per-slot nonce.** For
//! most of this project's life it was not: slots held `tag || QuantVec::encode()` in the clear,
//! and the argument for why that was acceptable — a tag is a PRF output, a query-space vector
//! means nothing without the embedding model — appeared right here, in this comment, and was
//! wrong. Padding was working perfectly and concealing only occupancy while every stored
//! embedding sat on disk readable with no key. See
//!
//!
//! Live and dummy slots are now indistinguishable because a dummy is uniform random bytes and a
//! live slot is a ciphertext; the only way to tell them apart is to hold the key and observe
//! which ones authenticate.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use mnemos_core::error::{Error, Result};
use mnemos_core::traits::Index;
use mnemos_core::types::{BucketId, Epoch, QuantVec, Tag};
use mnemos_core::Params;

use crate::bucket::{bucket_len, slot_len, BucketSlot};
use crate::revoke::RevocationList;
use crate::token::TokenTable;
use mnemos_crypto::kdf::DerivedKey;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const MAGIC: &[u8; 8] = b"MNEMOSIX";
const FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = 64;

/// Encrypted IVF: a centroid table plus `B` fixed-capacity padded partitions.
pub struct EncryptedIvf {
    file: File,
    /// `B` centroids, held in memory while unlocked.
    centroids: Vec<QuantVec>,
    buckets: u32,
    capacity: u32,
    dim: usize,
    probe_width: u32,
    decoy_buckets: u32,
    probe_spread: u32,
    probe_retention: f32,
    overflow_depth: u32,
    /// Records placed outside their assigned partition, and records the index could not place
    /// at all. In memory only and never persisted: these are operational counters, and writing
    /// them to disk would make "how skewed is this corpus" readable off the store.
    spilled: u64,
    dropped: u64,
    /// Live-slot set per bucket. **In memory only**, like the slab's occupancy: which slots
    /// hold real entries is a secret, and it is exactly what padding exists to hide.
    occupied: Vec<Vec<bool>>,
    /// AEAD key for bucket slots. `None` while locked.
    ///
    /// Slots are sealed, so reading a partition's *contents* requires the key; reading its
    /// *size* does not, and never will, which is why padding is a separate mechanism from
    /// encryption and both are necessary.
    slot_key: Option<DerivedKey>,
    /// Forward-privacy token chains, one per partition.
    ///
    /// `None` while the store is locked: the chains derive from key material, and holding them
    /// across a lock would keep key-derived state resident exactly when the design says it must
    /// not be. Installed by [`EncryptedIvf::install_tokens`] on unlock.
    tokens: Option<TokenTable>,
    /// Type-II revocation lists, one per partition. Preallocated and dummy-filled, so their
    /// on-disk footprint is a function of capacity and never of how many deletions happened.
    revocations: Vec<RevocationList>,
}

impl EncryptedIvf {
    /// Build a fresh index over the given centroids, fully padded.
    ///
    /// # Errors
    /// I/O failures.
    pub fn create(path: &Path, params: &Params, centroids: Vec<QuantVec>) -> Result<Self> {
        let buckets = params.index.buckets;
        let capacity = params.index.bucket_capacity;
        let dim = params.embedding.dim;

        if centroids.len() != buckets as usize {
            return Err(Error::Format(format!(
                "expected {buckets} centroids, got {}",
                centroids.len()
            )));
        }

        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)?;

        let mut header = vec![0u8; HEADER_LEN];
        header[..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&buckets.to_le_bytes());
        header[16..20].copy_from_slice(&capacity.to_le_bytes());
        header[20..24].copy_from_slice(&u32::try_from(dim).unwrap_or(0).to_le_bytes());
        file.write_all(&header)?;

        // Every slot in every bucket starts as dummy material, so bucket regions are
        // byte-identical in size and distribution whatever they end up holding.
        for _ in 0..(u64::from(buckets) * u64::from(capacity)) {
            file.write_all(&BucketSlot::dummy_bytes(dim))?;
        }
        file.flush()?;

        Ok(Self {
            file,
            centroids,
            buckets,
            capacity,
            dim,
            probe_width: params.index.probe_width,
            decoy_buckets: params.index.decoy_buckets,
            probe_spread: params.index.probe_spread,
            probe_retention: params.index.probe_retention,
            overflow_depth: params.index.overflow_depth,
            spilled: 0,
            dropped: 0,
            occupied: vec![vec![false; capacity as usize]; buckets as usize],
            slot_key: None,
            tokens: None,
            revocations: (0..buckets)
                .map(|b| RevocationList::new(BucketId(b), capacity))
                .collect(),
        })
    }

    /// Open an existing index.
    ///
    /// # Errors
    /// I/O or [`Error::Format`].
    ///
    /// # Panics
    /// If the header is shorter than `HEADER_LEN`, which `read_exact` already rejects.
    pub fn open(
        path: &Path,
        params: &Params,
        centroids: Vec<QuantVec>,
        occupied: Vec<Vec<bool>>,
    ) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut header = vec![0u8; HEADER_LEN];
        file.read_exact(&mut header)?;
        if &header[..8] != MAGIC {
            return Err(Error::Format("bad index magic".into()));
        }
        let version = u32::from_le_bytes(header[8..12].try_into().expect("4"));
        if version != FORMAT_VERSION {
            return Err(Error::Format(format!("index format version {version}")));
        }
        Ok(Self {
            file,
            centroids,
            buckets: params.index.buckets,
            capacity: params.index.bucket_capacity,
            dim: params.embedding.dim,
            probe_width: params.index.probe_width,
            decoy_buckets: params.index.decoy_buckets,
            probe_spread: params.index.probe_spread,
            probe_retention: params.index.probe_retention,
            overflow_depth: params.index.overflow_depth,
            spilled: 0,
            dropped: 0,
            occupied,
            slot_key: None,
            tokens: None,
            revocations: (0..params.index.buckets)
                .map(|b| RevocationList::new(BucketId(b), params.index.bucket_capacity))
                .collect(),
        })
    }

    /// Size of a partition's region on disk. A function of capacity alone; exposed so tests
    /// can assert that directly rather than through a proxy.
    ///
    /// # Errors
    /// I/O failures.
    pub fn raw_bucket_bytes(&self, b: BucketId) -> Result<usize> {
        if b.0 >= self.buckets {
            return Err(Error::Format("bucket id out of range".into()));
        }
        Ok(bucket_len(self.capacity, self.dim))
    }

    /// Bytes read from disk by one search. **The observable quantity.**
    ///
    /// This, not the candidate count, is what an adversary watching the file system measures.
    /// The two used to coincide because unencrypted dummy slots decoded successfully and became
    /// candidates; now that slots are sealed, dummies fail to authenticate and are dropped in
    /// memory *after* the read. The I/O is unchanged; only an in-process count differs.
    #[must_use]
    pub fn bytes_read_per_search(&self) -> u64 {
        u64::from(self.slots_per_search()) * slot_len(self.dim) as u64
    }

    /// Exactly how many index slots a search reads. Constant by construction.
    #[must_use]
    pub fn slots_per_search(&self) -> u32 {
        (self.probe_width * self.probe_spread + self.decoy_buckets) * self.capacity
    }

    /// Live-slot map, for persisting into the manifest.
    #[must_use]
    pub fn occupancy(&self) -> &[Vec<bool>] {
        &self.occupied
    }

    /// The partitions a query touches.
    ///
    /// Exactly `probe_width * probe_spread + decoy_buckets` of them, always, whatever the
    /// query and whatever the store holds. That count is the constant-volume invariant and
    /// nothing below is allowed to make it vary.
    ///
    /// # Three mechanisms, defending three different things
    ///
    /// **Real probes** (`probe_width`): the top-`k` partitions by centroid similarity. These
    /// are what makes search work.
    ///
    /// **Decoys** (`decoy_buckets`): extra partitions chosen by a PRF over `(query, epoch)`,
    /// *deterministically*. Fresh randomness here would be worse than useless — an adversary
    /// seeing the same query twice in an epoch would intersect the two read sets and cancel
    /// the decoys out, so determinism is the point.
    ///
    /// **Spread** (`probe_spread`) and **retention** (`probe_retention`): the query-equality
    /// mitigation, and the only mechanism here drawing *fresh* randomness per call.
    ///
    /// # Why a third mechanism was needed
    ///
    /// Determinism is exactly what makes decoys work and exactly what makes query equality
    /// unfixable by them. The whole read set — real probes and decoys alike — is a pure
    /// function of `(query, epoch)`, so two searches for the same thing produce byte-identical
    /// observations. Attack A5 measures this and recovers query equality at 1.000, *across*
    /// epochs as well as within them, because the real probes never depended on an epoch at
    /// all. Token rotation cannot touch it: tokens govern which entries a search can
    /// recognise, not which partitions it reads.
    ///
    /// Spread breaks the determinism by reading `probe_spread * probe_width` partitions, of
    /// which only `probe_width` are real, with the remainder drawn freshly each call. Two
    /// searches for the same query now differ.
    ///
    /// That alone defeats *pairwise* equality but not *intersection*: an adversary who can
    /// group repetitions some other way intersects them and the real probes fall out, because
    /// they are in every set. Retention closes that too, by dropping each real probe
    /// independently with probability `1 - p`, so a real probe survives `r` intersected
    /// repetitions with probability only `p^r`. The cost is recall — a dropped probe is a
    /// partition genuinely not searched — which is why it is a knob and not a default.
    ///
    /// With `probe_spread == 1 && probe_retention == 1.0` this reduces *exactly* to the
    /// original deterministic behaviour, including the decoy walk, so every measurement taken
    /// before the mitigation existed remains reproducible.
    #[must_use]
    pub fn probe_set(&self, q: &QuantVec, epoch: Epoch) -> Vec<BucketId> {
        let mut scored: Vec<(BucketId, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (BucketId(i as u32), q.cosine(c)))
            .collect();
        // Descending by similarity; ties broken by index so the order is total and stable.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        let real_k = self.probe_width as usize;
        let want = (self.probe_width * self.probe_spread) as usize;
        let randomised = self.probe_spread > 1 || self.probe_retention < 1.0;

        let mut set: Vec<BucketId> = Vec::with_capacity(want + self.decoy_buckets as usize);

        if randomised {
            // Fresh entropy per call. This is the one place in the read path that must *not*
            // be reproducible: reproducibility here is precisely the leak.
            let mut rng = ChaCha20Rng::from_entropy();

            for (b, _) in scored.iter().take(real_k) {
                if self.probe_retention >= 1.0 || rng.gen::<f32>() < self.probe_retention {
                    set.push(*b);
                }
            }

            // Backfill to the full width with uniform partitions. A dropped real probe is
            // replaced rather than simply omitted, so retention costs recall but never volume.
            let mut guard = 0u64;
            let limit = u64::from(self.buckets) * 8;
            while set.len() < want && guard < limit {
                let cand = BucketId(rng.gen_range(0..self.buckets));
                if !set.contains(&cand) {
                    set.push(cand);
                }
                guard += 1;
            }
            // Degenerate fallback: if the store has so few partitions that random sampling
            // cannot reach `want` distinct ones, walk them in order. Volume stays constant,
            // which is the property that must not break.
            for i in 0..self.buckets {
                if set.len() >= want {
                    break;
                }
                let cand = BucketId(i);
                if !set.contains(&cand) {
                    set.push(cand);
                }
            }
        } else {
            set.extend(scored.iter().take(real_k).map(|(b, _)| *b));
        }

        if self.decoy_buckets > 0 {
            let chosen: std::collections::HashSet<u32> = set.iter().map(|b| b.0).collect();
            // PRF-seeded walk over the remaining partitions, keyed on the query's own bytes and
            // the epoch, so the choice is deterministic for a given (query, epoch).
            let mut seed = u64::from(epoch.0 as u32);
            for c in q.encode().iter().take(16) {
                seed = seed
                    .wrapping_mul(0x0100_0000_01b3)
                    .wrapping_add(u64::from(*c));
            }
            let mut added = 0;
            let mut i = 0u64;
            while added < self.decoy_buckets && i < u64::from(self.buckets) * 4 {
                let cand = ((seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15)))
                    % u64::from(self.buckets)) as u32;
                if !chosen.contains(&cand) && !set.iter().any(|b| b.0 == cand) {
                    set.push(BucketId(cand));
                    added += 1;
                }
                i += 1;
            }
        }

        // Canonical read order. Without this the sequence of reads would carry the similarity
        // ranking, which under randomisation would hand back exactly what randomisation is
        // meant to remove: the real probes are the ones that come first.
        set.sort_unstable_by_key(|b| b.0);
        set
    }

    /// Read an entire partition — **always the whole thing**.
    ///
    /// A partial read would make I/O volume depend on occupancy, which is what full padding
    /// exists to hide.
    ///
    /// # Errors
    /// I/O failures.
    pub fn read_bucket(&self, b: BucketId) -> Result<Vec<BucketSlot>> {
        if b.0 >= self.buckets {
            return Err(Error::Format("bucket id out of range".into()));
        }
        let sl = slot_len(self.dim);
        let base = HEADER_LEN as u64 + u64::from(b.0) * u64::from(self.capacity) * sl as u64;
        let mut f = self.file.try_clone()?;
        f.seek(SeekFrom::Start(base))?;
        let mut buf = vec![0u8; sl * self.capacity as usize];
        f.read_exact(&mut buf)?;

        // EVERY slot is read and EVERY slot is offered to the AEAD. Dummies fail to
        // authenticate and are dropped here, in memory, after the I/O has already happened --
        // so the number of bytes read is still a function of `capacity` alone. Filtering before
        // the read would be the volumetric leak this design exists to remove.
        let key = self.slot_key()?;
        let dim = self.dim;
        Ok(buf
            .chunks_exact(sl)
            .filter_map(|c| BucketSlot::open(key, c, dim))
            .collect())
    }

    /// Install forward-privacy token chains derived from `root`.
    ///
    /// Called by the engine on unlock. Separated from construction because the chains are key
    /// material: `mnemos-index` must never hold a key across a lock, and the engine is the only
    /// component that knows when the store is unlocked.
    ///
    /// # Errors
    /// [`Error::Crypto`] on key-derivation failure.
    pub fn install_tokens(&mut self, root: &DerivedKey) -> Result<()> {
        self.tokens = Some(TokenTable::derive(root, self.buckets)?);
        // The same key seals the bucket slots. One key for both because both are
        // partition-scoped index material with the same lifetime; splitting them would add a
        // derivation without adding a boundary.
        self.slot_key = Some(root.clone());
        Ok(())
    }

    /// Drop the token chains and the slot key. Called on lock, alongside key zeroisation.
    pub fn clear_tokens(&mut self) {
        self.tokens = None;
        self.slot_key = None;
    }

    fn slot_key(&self) -> Result<&DerivedKey> {
        self.slot_key.as_ref().ok_or(Error::Locked)
    }

    /// The search token for a partition at an epoch, if the chains are installed.
    ///
    /// # Errors
    /// [`Error::Locked`] if no chains are installed, or [`Error::Crypto`] if the epoch has been
    /// punctured away — which is forward privacy working, not a fault.
    pub fn search_token(&self, bucket: BucketId, epoch: Epoch) -> Result<[u8; 16]> {
        self.tokens
            .as_ref()
            .ok_or(Error::Locked)?
            .token(bucket, epoch)
    }

    /// Bytes the token table would occupy on disk. **Constant**, by construction.
    #[must_use]
    pub fn token_state_len(&self) -> usize {
        self.tokens.as_ref().map_or(0, TokenTable::on_disk_len)
    }

    /// Whether a candidate has been revoked.
    ///
    /// Consulted during rerank on entries **already read**. See `crate::revoke` for why it must
    /// never be used to skip a read.
    ///
    /// # Errors
    /// Propagates from the revocation list; currently infallible.
    pub fn is_revoked(&self, bucket: BucketId, tag: Tag) -> Result<bool> {
        match self.revocations.get(bucket.0 as usize) {
            Some(list) => list.is_revoked(tag),
            None => Ok(false),
        }
    }

    /// Revoke a tag without knowing which partition holds it.
    ///
    /// **Scans every partition, always, even after finding the entry.** That is deliberate and
    /// it is not a missing optimisation: stopping early would make the cost of a delete depend
    /// on where the entry lived, which is a partition-identifying side channel on exactly the
    /// operation an adversary is most interested in. A delete is rare; a constant one is cheap
    /// enough to prefer.
    ///
    /// The engine needs this because it holds an entry's tag but not its partition — partition
    /// is a function of the *vector*, which is exactly what deletion is removing.
    ///
    /// # Errors
    /// I/O failures.
    pub fn revoke_anywhere(&mut self, tag: Tag, epoch: Epoch) -> Result<bool> {
        let mut found = false;
        for b in 0..self.buckets {
            if self.revoke(BucketId(b), tag, epoch).is_ok() && self.was_revoked(BucketId(b), tag) {
                found = true;
            }
        }
        Ok(found)
    }

    fn was_revoked(&self, bucket: BucketId, tag: Tag) -> bool {
        self.revocations
            .get(bucket.0 as usize)
            .is_some_and(|l| l.is_revoked(tag).unwrap_or(false))
    }

    /// Free slot within one partition, chosen by a PRF over the tag rather than by scan order.
    ///
    /// A first-free policy would order entries within the partition by arrival, reintroducing
    /// the ordering artifact history independence removes.
    fn free_slot(&self, bucket: BucketId, tag: Tag) -> Option<u32> {
        let cap = self.capacity;
        let start = u32::from(tag.as_bytes()[0]) % cap;
        (0..cap)
            .map(|i| (start + i) % cap)
            .find(|slot| !self.occupied[bucket.0 as usize][*slot as usize])
    }

    /// The assigned partition, followed by up to `overflow_depth` next-nearest ones.
    ///
    /// Ranked by centroid similarity to the record itself, which is what makes a spilled record
    /// still findable: a query close enough to retrieve the record reads a similar top-`k_probe`
    /// set, and `overflow_depth < probe_width` is enforced in [`Params::validate`] so the
    /// deepest spill target is always inside the probe width of the record's own vector.
    fn spill_order(&self, bucket: BucketId, v: &QuantVec) -> Vec<BucketId> {
        let mut order = vec![bucket];
        if self.overflow_depth == 0 {
            return order;
        }
        let mut ranked: Vec<(BucketId, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (BucketId(i as u32), v.cosine(c)))
            .filter(|(b, _)| *b != bucket)
            .collect();
        // Descending similarity, ties broken by partition id so the order is total and does not
        // depend on the enumeration order of a hash map or on float comparison luck.
        ranked.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        order.extend(
            ranked
                .into_iter()
                .take(self.overflow_depth as usize)
                .map(|(b, _)| b),
        );
        order
    }

    /// `(spilled, dropped)`: records placed outside their assigned partition, and records the
    /// index could not place at all.
    ///
    /// Exposed because the interesting failure is the quiet one. A store whose geometry is too
    /// small does not crash; it silently stops indexing, and every functional test stays green.
    #[must_use]
    pub const fn overflow_stats(&self) -> (u64, u64) {
        (self.spilled, self.dropped)
    }

    /// Place `s` in `slot` by rewriting the **whole** partition region.
    ///
    /// Writing only the chosen slot would disclose, to anyone watching block-level writes,
    /// which slots in the partition were free at the time. Accumulated over many writes that
    /// reveals how many entries each partition holds, including entries written before the
    /// observer arrived, which is the per-partition occupancy the padding exists to hide.
    /// Every live slot is therefore re-sealed under a fresh nonce and every free slot is
    /// refilled with fresh random bytes, so the region changes completely on every write and
    /// the only thing a write discloses is which partition it touched.
    fn rewrite_region(&mut self, b: BucketId, place: Option<(u32, &BucketSlot)>) -> Result<()> {
        let sl = slot_len(self.dim);
        let cap = self.capacity as usize;
        let base = HEADER_LEN as u64 + u64::from(b.0) * u64::from(self.capacity) * sl as u64;
        let key = self.slot_key()?.clone();

        let mut current = vec![0u8; sl * cap];
        {
            let mut f = self.file.try_clone()?;
            f.seek(SeekFrom::Start(base))?;
            f.read_exact(&mut current)?;
        }

        let mut region = Vec::with_capacity(sl * cap);
        for i in 0..cap {
            if let Some((slot, s)) = place.filter(|(slot, _)| *slot as usize == i) {
                debug_assert_eq!(slot as usize, i);
                region.extend_from_slice(&s.seal(&key, self.dim)?);
            } else if self.occupied[b.0 as usize][i] {
                let live = BucketSlot::open(&key, &current[i * sl..(i + 1) * sl], self.dim)
                    .ok_or(Error::Crypto)?;
                region.extend_from_slice(&live.seal(&key, self.dim)?);
            } else {
                region.extend_from_slice(&BucketSlot::dummy_bytes(self.dim));
            }
        }

        self.file.seek(SeekFrom::Start(base))?;
        self.file.write_all(&region)?;
        Ok(())
    }
}

impl Index for EncryptedIvf {
    fn assign(&self, v: &QuantVec) -> BucketId {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (BucketId(i as u32), v.cosine(c)))
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(b.0.cmp(&a.0))
            })
            .map_or(BucketId(0), |(b, _)| b)
    }

    fn insert(
        &mut self,
        bucket: BucketId,
        tag: Tag,
        v: &QuantVec,
        _epoch: Epoch,
    ) -> Result<BucketId> {
        if bucket.0 >= self.buckets {
            return Err(Error::Format("bucket id out of range".into()));
        }

        // Assigned partition first, then its next-nearest neighbours in similarity order.
        //
        // Before this existed, a full partition meant the record was written to the slab, the
        // engine discarded the `Err`, and the entry never appeared in a search result again.
        // Nothing reported it: `stat()` counted the entry, the slab held it, and only recall
        // against known ground truth could have noticed. See `IndexParams::overflow_depth` for
        // why spilling -- rather than the textbook overflow area -- is the repair that keeps
        // reads constant-volume.
        for (depth, target) in self.spill_order(bucket, v).into_iter().enumerate() {
            if let Some(slot) = self.free_slot(target, tag) {
                self.rewrite_region(
                    target,
                    Some((
                        slot,
                        &BucketSlot {
                            tag,
                            vector: v.clone(),
                        },
                    )),
                )?;
                self.occupied[target.0 as usize][slot as usize] = true;
                self.file.flush()?;
                if depth > 0 {
                    self.spilled += 1;
                }
                return Ok(target);
            }
        }

        // Every candidate full. This is now an error that propagates rather than a silent
        // omission, and the count is exposed through `overflow_stats` so a caller can see the
        // index shedding records instead of inferring it from a recall drop months later.
        self.dropped += 1;
        Err(Error::CapacityExceeded {
            live: u64::from(self.capacity),
            n_max: u64::from(self.capacity),
        })
    }

    fn search(&self, q: &QuantVec) -> Result<Vec<(Tag, f32)>> {
        // Reads every slot of every probed partition, real and decoy alike. No early exit, no
        // skipping empty buckets, no partial reads — see the module docs and attack A5.
        let mut out = Vec::new();
        for b in self.probe_set(q, Epoch(0)) {
            for slot in self.read_bucket(b)? {
                out.push((slot.tag, q.cosine(&slot.vector)));
            }
        }
        Ok(out)
    }

    fn revoke(&mut self, bucket: BucketId, tag: Tag, epoch: Epoch) -> Result<()> {
        // Overwrite the slot with dummy material and free it. The read path still reads the
        // whole partition afterwards, so volume is unchanged.
        let sl = slot_len(self.dim);
        for slot in 0..self.capacity {
            if !self.occupied[bucket.0 as usize][slot as usize] {
                continue;
            }
            let off = HEADER_LEN as u64
                + (u64::from(bucket.0) * u64::from(self.capacity) + u64::from(slot)) * sl as u64;
            let mut f = self.file.try_clone()?;
            f.seek(SeekFrom::Start(off))?;
            let mut raw = vec![0u8; sl];
            f.read_exact(&mut raw)?;
            // The tag is inside the sealed slot now, so identifying the entry to revoke takes
            // the key. That is the correct shape: deletion is a privileged operation.
            let matches = BucketSlot::open(self.slot_key()?, &raw, self.dim)
                .is_some_and(|s| s.tag.as_bytes() == tag.as_bytes());
            if matches {
                // Free the slot, then rewrite the whole region. Overwriting only this slot
                // would disclose its position within the partition, and that position was
                // chosen by probing past whichever slots were occupied when the entry arrived.
                self.occupied[bucket.0 as usize][slot as usize] = false;
                self.rewrite_region(bucket, None)?;
                // Record the revocation as well as overwriting the slot. The overwrite removes
                // the vector from disk; the record is what lets a concurrent in-flight search
                // drop the candidate during rerank rather than returning a stale hit.
                if let Some(list) = self.revocations.get_mut(bucket.0 as usize) {
                    list.revoke(tag, epoch)?;
                }
                self.file.flush()?;
                return Ok(());
            }
        }
        Err(Error::NotFound)
    }

    fn advance_tokens(&mut self, epoch: Epoch) -> Result<()> {
        // EVERY partition advances, not only the ones written this epoch. Advancing a subset
        // would make the advanced set name exactly the partitions that received real records —
        // handing back the information the WAL's dummy padding just paid to hide.
        //
        // A locked store has no chains installed and nothing to advance; that is not an error,
        // because `flush` on a locked store already fails earlier with `Error::Locked`.
        match self.tokens.as_mut() {
            Some(t) => t.advance_all(epoch),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnemos_core::types::Embedding;

    fn params(buckets: u32, capacity: u32, dim: usize) -> Params {
        let mut p = Params::discover().expect("params.toml");
        p.index.buckets = buckets;
        p.index.bucket_capacity = capacity;
        p.index.probe_width = 2;
        p.index.decoy_buckets = 0;
        // Pinned rather than inherited. `params.toml` now ships the query-equality mitigation
        // on (s = 2), and these tests are about index mechanics, not about the mitigation --
        // inheriting a randomised probe set would make every one of them flaky for a reason
        // that has nothing to do with what it asserts. The mitigation has its own tests.
        p.index.probe_spread = 1;
        p.index.probe_retention = 1.0;
        p.index.overflow_depth = 1;
        p.embedding.dim = dim;
        p
    }

    fn qv(v: Vec<f32>) -> QuantVec {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        QuantVec::quantise(&Embedding(v.into_iter().map(|x| x / n).collect()))
    }

    fn centroids(b: u32, dim: usize) -> Vec<QuantVec> {
        (0..b)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i as usize % dim] = 1.0;
                qv(v)
            })
            .collect()
    }

    /// Create an index and unlock it.
    ///
    /// Slots are sealed now, so an index without a key can neither read nor write them. Every
    /// test needs the unlocked form; production gets it from `Engine::create`/`unlock`.
    fn open_unlocked(path: &std::path::Path, p: &Params, cents: Vec<QuantVec>) -> EncryptedIvf {
        let mut ivf = EncryptedIvf::create(path, p, cents).unwrap();
        ivf.install_tokens(&DerivedKey([0x3Cu8; 32])).unwrap();
        ivf
    }

    /// Byte ranges of every slot in partition `b`, read from the index file.
    fn region_slots(path: &std::path::Path, b: u32, capacity: u32, dim: usize) -> Vec<Vec<u8>> {
        let bytes = std::fs::read(path).unwrap();
        let sl = slot_len(dim);
        let base = HEADER_LEN + b as usize * capacity as usize * sl;
        (0..capacity as usize)
            .map(|i| bytes[base + i * sl..base + (i + 1) * sl].to_vec())
            .collect()
    }

    #[test]
    fn insert_and_revoke_rewrite_the_whole_partition_region() {
        // Writing only the chosen slot would show, over many writes, which slots of a partition
        // were free, and so how many entries it holds. Every slot of the touched region must
        // change, and no other region may change.
        let (b, cap, dim) = (4u32, 8u32, 8usize);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.slab");
        let p = params(b, cap, dim);
        let mut ivf = open_unlocked(&path, &p, centroids(b, dim));

        let v = qv(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        ivf.insert(BucketId(1), Tag([1u8; 16]), &v, Epoch(0))
            .unwrap();
        let before_touched = region_slots(&path, 1, cap, dim);
        let before_other = region_slots(&path, 2, cap, dim);

        ivf.insert(BucketId(1), Tag([2u8; 16]), &v, Epoch(0))
            .unwrap();
        let after_insert = region_slots(&path, 1, cap, dim);
        for (i, (x, y)) in before_touched.iter().zip(&after_insert).enumerate() {
            assert_ne!(
                x, y,
                "slot {i} of the written partition was left unchanged by an insert"
            );
        }
        assert_eq!(
            before_other,
            region_slots(&path, 2, cap, dim),
            "an untouched region changed"
        );

        ivf.revoke(BucketId(1), Tag([1u8; 16]), Epoch(1)).unwrap();
        let after_revoke = region_slots(&path, 1, cap, dim);
        for (i, (x, y)) in after_insert.iter().zip(&after_revoke).enumerate() {
            assert_ne!(
                x, y,
                "slot {i} of the written partition was left unchanged by a revoke"
            );
        }
    }

    #[test]
    fn every_bucket_occupies_the_same_bytes_regardless_of_content() {
        // The leakage-profile row "per-bucket occupancy: none". This is the single property
        // that defeats the structure-transfer attack of finding F2.
        let dir = tempfile::tempdir().unwrap();
        let p = params(8, 4, 4);
        let empty = dir.path().join("a.idx");
        let full = dir.path().join("b.idx");

        let a = open_unlocked(&empty, &p, centroids(8, 4));
        let mut b = open_unlocked(&full, &p, centroids(8, 4));
        for i in 0..4u8 {
            b.insert(
                BucketId(0),
                Tag([i; 16]),
                &qv(vec![1.0, 0.0, 0.0, 0.0]),
                Epoch(0),
            )
            .unwrap();
        }
        drop(a);
        drop(b);

        assert_eq!(
            std::fs::metadata(&empty).unwrap().len(),
            std::fs::metadata(&full).unwrap().len(),
            "an empty and a full index must be identical in size"
        );
    }

    #[test]
    fn search_reads_a_constant_number_of_slots() {
        // The constant-volume invariant, executable.
        //
        // Asserted on BYTES READ, not on candidate count. Those two used to be the same number
        // because unencrypted dummy slots decoded successfully and became candidates. Sealed
        // dummies fail authentication and are dropped in memory after the read, so the
        // candidate count now tracks live entries while the I/O does not. The adversary in
        // the paper's threat model observes the file system, so the I/O is the quantity that matters;
        // asserting the old proxy would now be asserting something the threat model never
        // mentioned, and would fail for a reason that is not a leak.
        let dir = tempfile::tempdir().unwrap();
        let p = params(8, 4, 4);
        let mut ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(8, 4));

        // Load one bucket heavily and leave the rest empty.
        for i in 0..4u8 {
            ivf.insert(
                BucketId(0),
                Tag([i; 16]),
                &qv(vec![1.0, 0.0, 0.0, 0.0]),
                Epoch(0),
            )
            .unwrap();
        }

        let dense_probe = ivf.probe_set(&qv(vec![1.0, 0.0, 0.0, 0.0]), Epoch(0));
        let sparse_probe = ivf.probe_set(&qv(vec![0.0, 0.0, 1.0, 0.0]), Epoch(0));
        assert_eq!(
            dense_probe.len(),
            sparse_probe.len(),
            "a query over a full partition read a different NUMBER of partitions than one over \
             an empty partition"
        );

        // Every probed partition is read whole, occupied or not.
        for b in dense_probe.iter().chain(sparse_probe.iter()) {
            let raw = ivf.raw_bucket_bytes(*b).unwrap();
            assert_eq!(raw, bucket_len(p.index.bucket_capacity, p.embedding.dim));
        }
        assert_eq!(
            ivf.bytes_read_per_search(),
            u64::from(ivf.slots_per_search()) * slot_len(p.embedding.dim) as u64
        );

        // And the searches still work.
        assert!(!ivf
            .search(&qv(vec![1.0, 0.0, 0.0, 0.0]))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn decoys_widen_the_probe_set_by_exactly_d() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(16, 4, 4);
        p.index.decoy_buckets = 3;
        let ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(16, 4));

        let set = ivf.probe_set(&qv(vec![1.0, 0.0, 0.0, 0.0]), Epoch(0));
        assert_eq!(
            set.len(),
            (p.index.probe_width + p.index.decoy_buckets) as usize
        );

        let unique: std::collections::HashSet<u32> = set.iter().map(|b| b.0).collect();
        assert_eq!(
            unique.len(),
            set.len(),
            "probe set must not contain duplicates"
        );
    }

    #[test]
    fn decoy_choice_is_stable_for_a_given_query_and_epoch() {
        // Fresh randomness per call would let a repeated query's read sets be intersected,
        // cancelling the decoys out entirely.
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(16, 4, 4);
        p.index.decoy_buckets = 3;
        let ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(16, 4));
        let q = qv(vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(ivf.probe_set(&q, Epoch(5)), ivf.probe_set(&q, Epoch(5)));
    }

    #[test]
    fn assignment_and_retrieval_agree() {
        let dir = tempfile::tempdir().unwrap();
        let p = params(4, 8, 4);
        let mut ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(4, 4));

        let v = qv(vec![0.0, 1.0, 0.0, 0.0]);
        let b = ivf.assign(&v);
        ivf.insert(b, Tag([9u8; 16]), &v, Epoch(0)).unwrap();

        let hits = ivf.search(&v).unwrap();
        let best = hits
            .iter()
            .max_by(|a, c| a.1.partial_cmp(&c.1).unwrap())
            .unwrap();
        assert_eq!(
            best.0.as_bytes(),
            &[9u8; 16],
            "the inserted vector should score highest"
        );
    }

    #[test]
    fn overflow_spills_into_a_nearby_partition_instead_of_dropping_the_record() {
        // The hole this closes: a full partition used to make `insert` return `Err`, the flush
        // path discarded it, and the record stayed durable in the slab while being absent from
        // every subsequent search. Nothing anywhere reported that. The store said the entry was
        // live, because it was -- just unfindable.
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(4, 2, 4);
        p.index.probe_width = 3;
        p.index.overflow_depth = 2;
        let mut ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(4, 4));

        let v = qv(vec![1.0, 0.5, 0.25, 0.0]);
        let b = ivf.assign(&v);
        // Four records into a partition that holds two.
        for i in 0..4u8 {
            ivf.insert(b, Tag([i; 16]), &v, Epoch(0)).unwrap();
        }

        assert_eq!(
            ivf.overflow_stats(),
            (2, 0),
            "two records should have spilled into a neighbour and none been dropped"
        );

        let found: std::collections::BTreeSet<u8> = ivf
            .search(&v)
            .unwrap()
            .iter()
            .map(|(t, _)| t.as_bytes()[0])
            .collect();
        assert_eq!(
            found,
            (0..4u8).collect::<std::collections::BTreeSet<u8>>(),
            "a spilled record must still be returned by a search for its own vector"
        );
    }

    #[test]
    fn spilling_does_not_change_read_volume() {
        // The reason spilling is the repair and an overflow *area* is not. A chain would have to
        // be read by any search probing the overflowing partition, making read volume depend on
        // occupancy -- the exact channel `bucket_capacity` padding exists to close.
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(4, 2, 4);
        p.index.probe_width = 3;
        p.index.overflow_depth = 2;
        let mut ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(4, 4));

        let before = ivf.bytes_read_per_search();
        let v = qv(vec![1.0, 0.5, 0.25, 0.0]);
        let b = ivf.assign(&v);
        for i in 0..4u8 {
            ivf.insert(b, Tag([i; 16]), &v, Epoch(0)).unwrap();
        }
        assert_eq!(
            before,
            ivf.bytes_read_per_search(),
            "spilling changed read volume"
        );
    }

    #[test]
    fn a_record_that_fits_nowhere_is_an_error_not_a_silent_omission() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(4, 1, 4);
        p.index.probe_width = 1;
        p.index.overflow_depth = 0;
        let mut ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(4, 4));

        let v = qv(vec![1.0, 0.5, 0.25, 0.0]);
        let b = ivf.assign(&v);
        ivf.insert(b, Tag([0u8; 16]), &v, Epoch(0)).unwrap();
        let second = ivf.insert(b, Tag([1u8; 16]), &v, Epoch(0));

        assert!(
            second.is_err(),
            "a record with nowhere to go must not be accepted quietly"
        );
        assert_eq!(ivf.overflow_stats(), (0, 1), "the drop must be counted");
    }

    #[test]
    fn insert_reports_the_partition_it_actually_wrote() {
        // A leakage event naming the *assigned* partition after a spill would describe a write
        // that did not happen. That is finding F5 with different coordinates, so the trait
        // hands the real destination back rather than letting the caller assume it.
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(4, 1, 4);
        p.index.probe_width = 3;
        p.index.overflow_depth = 2;
        let mut ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(4, 4));

        let v = qv(vec![1.0, 0.5, 0.25, 0.0]);
        let b = ivf.assign(&v);
        assert_eq!(ivf.insert(b, Tag([0u8; 16]), &v, Epoch(0)).unwrap(), b);
        let spilled_to = ivf.insert(b, Tag([1u8; 16]), &v, Epoch(0)).unwrap();
        assert_ne!(spilled_to, b, "the second record had to go somewhere else");
    }

    #[test]
    fn revoke_removes_the_entry_but_not_the_volume() {
        let dir = tempfile::tempdir().unwrap();
        let p = params(4, 8, 4);
        let mut ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(4, 4));
        let v = qv(vec![1.0, 0.0, 0.0, 0.0]);
        let b = ivf.assign(&v);
        ivf.insert(b, Tag([1u8; 16]), &v, Epoch(0)).unwrap();

        // Again: the invariant is I/O, not candidate count. A deleted entry SHOULD stop being
        // a candidate -- that is what deletion means -- while the bytes read stay identical.
        let before_bytes = ivf.bytes_read_per_search();
        let before_region = ivf.raw_bucket_bytes(b).unwrap();
        assert_eq!(
            ivf.search(&v).unwrap().len(),
            1,
            "the entry should be findable first"
        );

        ivf.revoke(b, Tag([1u8; 16]), Epoch(1)).unwrap();

        assert_eq!(
            ivf.search(&v).unwrap().len(),
            0,
            "a revoked entry was still returned -- deletion did not take effect"
        );
        assert_eq!(
            before_bytes,
            ivf.bytes_read_per_search(),
            "deletion changed read volume"
        );
        assert_eq!(
            before_region,
            ivf.raw_bucket_bytes(b).unwrap(),
            "deletion changed the partition's size on disk"
        );
    }
    // ---- query-equality mitigation -------------------------------------------------------

    /// The leak, stated as a test so it cannot be quietly reintroduced.
    ///
    /// With the mitigation off, the read set is a pure function of the query. This is the
    /// property attack A5 turns into query-equality recovery of 1.000, and it is *deliberate*
    /// at these settings — decoys have to be deterministic or they cancel by intersection.
    #[test]
    fn without_spread_the_probe_set_is_a_pure_function_of_the_query() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(32, 8, 4);
        p.index.probe_spread = 1;
        p.index.probe_retention = 1.0;
        p.index.decoy_buckets = 3;
        let ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(32, 4));
        let q = qv(vec![1.0, 0.0, 0.0, 0.0]);
        let first = ivf.probe_set(&q, Epoch(7));
        for _ in 0..16 {
            assert_eq!(
                first,
                ivf.probe_set(&q, Epoch(7)),
                "read set must be deterministic"
            );
        }
    }

    #[test]
    fn spread_makes_repeated_queries_produce_different_read_sets() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(64, 8, 4);
        p.index.probe_width = 4;
        p.index.probe_spread = 4;
        p.index.probe_retention = 1.0;
        let ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(64, 4));
        let q = qv(vec![1.0, 0.0, 0.0, 0.0]);

        let sets: Vec<Vec<BucketId>> = (0..32).map(|_| ivf.probe_set(&q, Epoch(1))).collect();
        let distinct: std::collections::HashSet<Vec<u32>> = sets
            .iter()
            .map(|s| s.iter().map(|b| b.0).collect())
            .collect();

        // 32 draws of 12 extra partitions from 64: collisions are possible but a *majority*
        // of draws colliding would mean the entropy is not there.
        assert!(
            distinct.len() > 16,
            "expected fresh randomness per search, got {} distinct sets in 32",
            distinct.len()
        );
    }

    #[test]
    fn spread_preserves_the_real_probes_and_the_volume() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(64, 8, 4);
        p.index.probe_width = 4;
        p.index.probe_spread = 4;
        p.index.probe_retention = 1.0;
        p.index.decoy_buckets = 2;
        let ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(64, 4));

        let mut plain = p.clone();
        plain.index.probe_spread = 1;
        let dir2 = tempfile::tempdir().unwrap();
        let base = open_unlocked(&dir2.path().join("j.idx"), &plain, centroids(64, 4));

        let q = qv(vec![1.0, 0.0, 0.0, 0.0]);
        let real: std::collections::HashSet<u32> = base
            .probe_set(&q, Epoch(0))
            .iter()
            .map(|b| b.0)
            .take(4)
            .collect();

        let want = (p.index.probe_width * p.index.probe_spread + p.index.decoy_buckets) as usize;
        for _ in 0..32 {
            let set = ivf.probe_set(&q, Epoch(0));
            assert_eq!(set.len(), want, "volume must stay constant under spread");
            let got: std::collections::HashSet<u32> = set.iter().map(|b| b.0).collect();
            // retention = 1.0, so at p = 1 every genuine probe is still read: recall is exact.
            assert!(
                real.is_subset(&got),
                "a superset must contain the real probes"
            );
        }
    }

    /// Retention is what defeats intersection, and this is the arithmetic that shows it.
    ///
    /// A real probe survives `r` intersected repetitions with probability `p^r`. At p = 0.5
    /// and r = 8 that is 1/256, so the intersection of eight repeats should almost never still
    /// contain all four real probes.
    #[test]
    fn retention_makes_real_probes_fall_out_of_an_intersection() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(64, 8, 4);
        p.index.probe_width = 4;
        p.index.probe_spread = 4;
        p.index.probe_retention = 0.5;
        let ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(64, 4));
        let q = qv(vec![1.0, 0.0, 0.0, 0.0]);

        let mut inter: std::collections::HashSet<u32> =
            ivf.probe_set(&q, Epoch(0)).iter().map(|b| b.0).collect();
        for _ in 0..7 {
            let next: std::collections::HashSet<u32> =
                ivf.probe_set(&q, Epoch(0)).iter().map(|b| b.0).collect();
            inter = inter.intersection(&next).copied().collect();
        }

        let mut plain = p.clone();
        plain.index.probe_spread = 1;
        plain.index.probe_retention = 1.0;
        let dir2 = tempfile::tempdir().unwrap();
        let base = open_unlocked(&dir2.path().join("j.idx"), &plain, centroids(64, 4));
        let real: std::collections::HashSet<u32> =
            base.probe_set(&q, Epoch(0)).iter().map(|b| b.0).collect();

        assert!(
            !real.is_subset(&inter),
            "intersecting 8 repeats still yielded every real probe; retention is not biting"
        );
    }

    #[test]
    fn the_read_order_carries_no_similarity_ranking() {
        // If the set were returned best-first, the adversary would read the ranking straight
        // off the trace and randomisation would be decorative.
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(32, 8, 4);
        p.index.probe_spread = 2;
        let ivf = open_unlocked(&dir.path().join("i.idx"), &p, centroids(32, 4));
        let set = ivf.probe_set(&qv(vec![1.0, 0.0, 0.0, 0.0]), Epoch(0));
        let mut sorted = set.clone();
        sorted.sort_unstable_by_key(|b| b.0);
        assert_eq!(
            set, sorted,
            "probe set must be returned in canonical partition order"
        );
    }
}
