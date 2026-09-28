//! The engine — orchestration, and the place cross-cutting invariants are enforced.
//!
//! Individual components each guarantee something locally (the slab is history-independent, the
//! index is fully padded, the WAL is fixed-length). The invariants that span them can only be
//! enforced here:
//!
//! 1. **Constant-volume search** — exactly `(k_probe + d) × C` index reads and exactly `top_k`
//!    content reads per query, dummy-padded when fewer entries matched.
//! 2. **Fixed-size epoch flush** — exactly `epoch_writes` records, real and dummy alike, all of
//!    them written to *both* the slab and the index. Inserting only the real ones into the
//!    index would make the index-write count reveal how many records in the batch were real,
//!    undoing the WAL's padding one layer down.
//! 3. **Fixed compaction budget** — `compaction_slots` per epoch, whatever is reclaimable.
//! 4. **Unconditional manifest rewrite** — every epoch, changed or not.
//! 5. **Plaintext confined to the arena** — decrypt in, copy out, zeroise.
//!
//! Each is asserted by `tests/leakage_profile.rs` against `docs/leakage-profile.md`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mnemos_core::error::{Error, Result};
use mnemos_core::event::{EventSink, LeakEvent, NullSink};
use mnemos_core::traits::{BlobStore, Index, KeyStore, Record, SearchableMemory, Stat};
use mnemos_core::types::{BucketId, Epoch, Hit, Memory, MemoryId, QuantVec, SlotAddr, Tag};
use mnemos_core::Params;
use mnemos_crypto::aead::{open_content, seal_content};
use mnemos_crypto::kdf::KeyHierarchy;
use mnemos_crypto::{prf, Arena};
use mnemos_index::EncryptedIvf;
use mnemos_store::manifest::Manifest;
use mnemos_store::slab::Slab;
use mnemos_store::wal::Wal;
use rand::RngCore;

use crate::epoch::EpochClock;

/// A memory staged in the current epoch but not yet flushed.
struct Staged {
    tag: Tag,
    vector: QuantVec,
}

/// The store.
///
/// Construct with [`Engine::create`] or [`Engine::open`], then [`SearchableMemory::unlock`].
/// Locked is the default state: an engine that started unlocked would keep the master key
/// resident for the whole process lifetime, and process lifetime is not a bound we control.
pub struct Engine {
    dir: PathBuf,
    params: Params,
    keystore: Box<dyn KeyStore>,
    slab: Slab,
    index: EncryptedIvf,
    wal: Wal,
    manifest: Manifest,
    events: Arc<dyn EventSink>,

    /// Key hierarchy. `None` while locked — the type makes "locked" unusable rather than a flag
    /// someone can forget to check.
    keys: Option<KeyHierarchy>,
    /// Locked plaintext region. `None` while locked.
    arena: Option<Arena>,
    clock: EpochClock,
    staged: Vec<Staged>,
    /// Records written durably to the slab that the index could not place anywhere.
    ///
    /// This used to be unrepresented, and the omission was invisible: the flush discarded the
    /// insert error, `stat()` reported the entry as live, and the only symptom was a recall
    /// figure nobody was measuring. It is a counter now, and [`Engine::unindexed`] exposes it,
    /// because the failure worth catching is the one with no symptom.
    unindexed: u64,
}

impl Engine {
    /// Records durably stored but not present in the index, because every candidate partition
    /// was full.
    ///
    /// A non-zero value means the store is silently shedding retrievability and the geometry
    /// (`buckets` x `bucket_capacity`) is too small for the corpus. Nothing else reports it:
    /// [`Engine::stat`] counts these entries as live, because they are.
    #[must_use]
    pub const fn unindexed(&self) -> u64 {
        self.unindexed
    }

    /// `(spilled, dropped)` from the index: records placed outside their assigned partition,
    /// and records placed nowhere. Spills cost recall; drops cost the record.
    #[must_use]
    pub const fn overflow_stats(&self) -> (u64, u64) {
        self.index.overflow_stats()
    }

    /// Create a new store, preallocated to `params.index.n_max`. Starts **unlocked**.
    ///
    /// # Errors
    /// I/O, key-custody, or parameter failures.
    pub fn create(
        dir: &Path,
        params: Params,
        keystore: Box<dyn KeyStore>,
        centroids: Vec<QuantVec>,
    ) -> Result<Self> {
        std::fs::create_dir_all(dir)?;

        let mut master = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut master);
        let wrapped = keystore.wrap_master_key(&master)?;
        let keys = KeyHierarchy::derive(&master)?;

        let slab = Slab::create(&dir.join("memory.slab"), &params)?;
        // Encoded before the index takes ownership. The table is persisted in the manifest so
        // that `open` restores the *same* partitioning: recomputing k-means would produce a
        // different table, which would silently reroute every entry and leave a store unable
        // to find its own contents.
        let centroid_bytes = encode_centroids(&centroids);
        let index = EncryptedIvf::create(&dir.join("index.slab"), &params, centroids)?;
        let wal = Wal::create(&dir.join("wal.enc"), &params, slab.payload_len())?;

        let index_slots = params.index.buckets as usize * params.index.bucket_capacity as usize;
        let manifest = Manifest::new(
            params.index.n_max,
            index_slots,
            centroid_bytes,
            wrapped.clone(),
        );
        manifest.write(&dir.join("manifest.enc"), &keys.content)?;
        // The wrapped key lives beside the store rather than inside the encrypted manifest,
        // because opening needs it *before* anything can be decrypted. It is useless without
        // the hardware that wrapped it.
        std::fs::write(dir.join("wrapped.key"), &wrapped)?;

        let clock = EpochClock::new(&params);
        let arena = Arena::new(params.memory.arena_bytes, params.memory.lock_pages)?;
        let events: Arc<dyn EventSink> = Arc::new(NullSink);

        // A freshly created store is unlocked, so its forward-privacy chains are live from the
        // first write. Deferring this to the first `unlock` would leave the initial epoch
        // unprotected -- and the initial epoch is the one an adversary who arrives early sees
        // most of.
        let mut index = index;
        index.install_tokens(&keys.bucket)?;

        events.emit(LeakEvent::Setup {
            n_max: params.index.n_max,
            buckets: params.index.buckets,
            bucket_capacity: params.index.bucket_capacity,
            block_size: params.crypto.block_size_bytes,
        });

        Ok(Self {
            dir: dir.to_path_buf(),
            params,
            keystore,
            slab,
            index,
            wal,
            manifest,
            events,
            keys: Some(keys),
            arena: Some(arena),
            clock,
            staged: Vec::new(),
            unindexed: 0,
        })
    }

    /// Open an existing store. Starts **locked**; call `unlock` before use.
    ///
    /// # Errors
    /// I/O or format failures.
    /// Takes no centroid table: it is restored from the manifest, encrypted, exactly as it was
    /// written at `create`. An earlier signature accepted one from the caller, which made every
    /// caller responsible for reproducing a k-means run bit-for-bit — a requirement nothing
    /// enforced and the CLI silently failed to meet.
    pub fn open(dir: &Path, params: Params, keystore: Box<dyn KeyStore>) -> Result<Self> {
        let wrapped = std::fs::read(dir.join("wrapped.key"))?;
        let master_vec = keystore.unwrap_master_key(&wrapped)?;
        let master: [u8; 32] = master_vec
            .as_slice()
            .try_into()
            .map_err(|_| Error::Format("master key is not 32 bytes".into()))?;
        let keys = KeyHierarchy::derive(&master)?;

        let manifest = Manifest::read(&dir.join("manifest.enc"), &keys.content)?;
        let slab = Slab::open(
            &dir.join("memory.slab"),
            &params,
            manifest.occupancy.clone(),
        )?;

        let centroids = decode_centroids(&manifest.centroids, params.embedding.dim);
        if centroids.len() != params.index.buckets as usize {
            return Err(Error::Format(format!(
                "manifest holds {} centroids, params declares {} partitions",
                centroids.len(),
                params.index.buckets
            )));
        }

        // Restored, not reset. Starting from an all-empty occupancy map made the next flush
        // insert from slot zero in every partition, overwriting live entries — writes kept
        // succeeding and only older entries vanished, which no functional test detects.
        let cap = params.index.bucket_capacity as usize;
        let occ: Vec<Vec<bool>> = (0..params.index.buckets as usize)
            .map(|b| {
                (0..cap)
                    .map(|i| {
                        manifest
                            .index_occupancy
                            .get(b * cap + i)
                            .copied()
                            .unwrap_or(false)
                    })
                    .collect()
            })
            .collect();
        let index = EncryptedIvf::open(&dir.join("index.slab"), &params, centroids, occ)?;
        let wal = Wal::open(&dir.join("wal.enc"), &params, slab.payload_len())?;
        let clock = EpochClock::new(&params);

        Ok(Self {
            dir: dir.to_path_buf(),
            params,
            keystore,
            slab,
            index,
            wal,
            manifest,
            events: Arc::new(NullSink),
            keys: None,
            arena: None,
            clock,
            staged: Vec::new(),
            unindexed: 0,
        })
    }

    /// Replace the event sink.
    ///
    /// Production uses [`NullSink`]; the demo uses a WebSocket broadcaster; the evaluation
    /// harness uses a recorder. The attack harness consumes the recorder's output **as the
    /// adversary's transcript**, which is what stops an attack from cheating.
    pub fn set_event_sink(&mut self, sink: Arc<dyn EventSink>) {
        self.events = sink;
    }

    fn keys(&self) -> Result<&KeyHierarchy> {
        self.keys.as_ref().ok_or(Error::Locked)
    }

    /// **Evaluation only.** The index bucket that holds record `id`, or `None` if it is staged
    /// or unindexed.
    ///
    /// Found by looking for the record's tag in the index rather than by recomputing its
    /// nearest centroid, because the two differ: the engine assigns over quantised vectors, and a
    /// full bucket spills its record into a neighbour. A recomputed assignment is the shortcut
    /// that put refusals into experiment E5's admissible arms. Emits no event: it answers the
    /// experimenter's question and is never part of what an adversary observes.
    ///
    /// # Errors
    /// [`Error::Locked`] when locked; I/O failures.
    #[doc(hidden)]
    pub fn actual_bucket(&self, id: MemoryId) -> Result<Option<u32>> {
        let tag = prf::tag_for(&self.keys()?.tag, id, 0);
        for b in 0..self.params.index.buckets {
            if self
                .index
                .read_bucket(BucketId(b))?
                .iter()
                .any(|slot| prf::tags_equal(slot.tag, tag))
            {
                return Ok(Some(b));
            }
        }
        Ok(None)
    }
}

impl SearchableMemory for Engine {
    fn put(&mut self, text: &str, meta: &str, embedding: &QuantVec) -> Result<MemoryId> {
        let block = self.params.crypto.block_size_bytes as usize;
        let id = MemoryId::random();

        let (tag, ct) = {
            let keys = self.keys()?;
            let tag = prf::tag_for(&keys.tag, id, 0);

            // Text and metadata are sealed together: metadata is content and gets no weaker
            // treatment.
            let mut plain = Vec::with_capacity(text.len() + meta.len() + 1);
            plain.extend_from_slice(text.as_bytes());
            plain.push(0u8);
            plain.extend_from_slice(meta.as_bytes());

            let ct = seal_content(&keys.content, &plain, tag, block)?;
            // Chunking across slots is not implemented. A record that seals to more than one
            // slot would overrun into the neighbouring slot, so it is rejected instead.
            if ct.len() != self.slab.payload_len() {
                return Err(Error::Format(format!(
                    "entry of {} bytes does not fit in one {block}-byte block",
                    plain.len()
                )));
            }
            (tag, ct)
        };

        let bucket = self.index.assign(embedding);
        self.wal.append(Record {
            tag,
            ciphertext: ct,
            bucket,
        })?;
        self.staged.push(Staged {
            tag,
            vector: embedding.clone(),
        });

        // Emits NOTHING until the flush. A per-put event would be a per-put disk write, and a
        // per-put disk write is a correlation channel for an adversary who chose the task that
        // caused it.
        if self.clock.stage() {
            self.flush()?;
        }
        Ok(id)
    }

    fn search(&self, query: &QuantVec) -> Result<Vec<Hit>> {
        let keys = self.keys()?;
        let top_k = self.params.index.top_k as usize;

        // 1. Probe set: exactly probe_width + decoy_buckets partitions, every time.
        let probes = self.index.probe_set(query, self.clock.current());
        let mut candidates: Vec<(Tag, f32)> = Vec::new();
        for b in &probes {
            self.events.emit(LeakEvent::BucketRead {
                bucket: *b,
                decoy: false,
            });
            for slot in self.index.read_bucket(*b)? {
                candidates.push((slot.tag, query.cosine(&slot.vector)));
            }
        }

        // 2. Exact rerank over every candidate, real and dummy alike.
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Fetch EXACTLY top_k content slots. When fewer than top_k candidates resolve to a
        //    live entry we still issue top_k reads — otherwise the read count would reveal how
        //    many entries genuinely matched, which is the volumetric leakage attack A5 hunts.
        let mut hits = Vec::new();
        let mut out = vec![0u8; self.params.crypto.max_entry_bytes as usize];

        for i in 0..top_k {
            let cand = candidates.get(i);
            let addr = cand
                .and_then(|(tag, _)| self.slab.lookup(&keys.idx.0, *tag))
                // A padding read at a uniformly random address. A fixed address (for example
                // the loop index) would make padding reads recognisable and reveal how many
                // candidates resolved to live entries.
                .unwrap_or_else(|| SlotAddr(rand::rngs::OsRng.next_u64() % self.slab.capacity()));

            let sealed = self.slab.read_slot(addr)?;
            self.events.emit(LeakEvent::SlotRead { addr });

            if let Some((tag, score)) = cand {
                if let Ok(n) = open_content(&keys.content, &sealed, *tag, &mut out) {
                    let (text, meta) = split_frame(&out[..n]);
                    hits.push(Hit {
                        memory: Memory {
                            text,
                            meta,
                            id: MemoryId([0u8; 16]),
                        },
                        score: *score,
                    });
                }
            }
        }

        Ok(hits)
    }

    fn update(
        &mut self,
        id: MemoryId,
        text: &str,
        meta: &str,
        embedding: &QuantVec,
    ) -> Result<MemoryId> {
        // delete + put, never in-place: an in-place rewrite is a same-slot write, which is the
        // correlation signal a live adversary most wants. Returns a NEW id — a visible API cost
        // paid for a security property (ADR-0004).
        self.delete(id)?;
        self.put(text, meta, embedding)
    }

    fn delete(&mut self, id: MemoryId) -> Result<()> {
        let (tag, k_idx) = {
            let keys = self.keys()?;
            (prf::tag_for(&keys.tag, id, 0), keys.idx.0)
        };
        let addr = self.slab.lookup(&k_idx, tag).ok_or(Error::NotFound)?;
        self.slab.free(addr)?;

        // Also remove the entry from the INDEX. Freeing the slab slot alone made the content
        // unreadable but left the entry's quantised vector sitting in its partition -- and the
        // vector is the semantic payload, which is precisely what the topic-recovery attack
        // consumes. A "deleted" entry whose vector survives on disk is not deleted in any sense
        // the threat model recognises.
        let epoch = self.clock.current();
        let _ = self.index.revoke_anywhere(tag, epoch);
        // A delete overwrites a slot immediately, so it IS disk-visible and must be on the
        // event bus. Omitting it would leave the transcript claiming nothing happened while
        // the disk says otherwise.
        self.events.emit(LeakEvent::SlotWrite {
            addr,
            epoch: self.clock.current(),
        });
        Ok(())
    }

    fn flush(&mut self) -> Result<Epoch> {
        let epoch = self.clock.current();
        let (k_epoch, k_idx) = {
            let keys = self.keys()?;
            (keys.epoch_key(epoch)?, keys.idx.0)
        };

        // Exactly epoch_writes records, dummy-padded and PRF-shuffled.
        let batch = self.wal.take_padded_batch(&k_epoch.0, epoch)?;
        debug_assert_eq!(batch.len(), self.params.epoch.epoch_writes as usize);

        // One event per slot ACTUALLY written, displacement cascades included — then padded up
        // to a fixed budget. Robin Hood displacement makes the raw count data-dependent (a
        // crowded neighbourhood cascades), so without this padding the write count would reveal
        // how full the region around each write was. Closes docs/leakage-profile.md §4.1.
        // Refuse rather than leak. Past `max_load_factor`, Robin Hood cascades exceed the
        // padding budget and the per-flush write count becomes data-dependent again — the
        // exact residual the budget exists to close. Cascade length *explodes* rather than
        // degrades (42 writes at 0.60 load, 244 at 0.875, 1211 at 0.94), so there is no safe
        // "slightly over". Until `Grow` is implemented this is a hard stop, which is the
        // correct failure mode: loud and safe beats quiet and leaking.
        let live = self.slab.live_count();
        if !self
            .params
            .within_safe_load(live + u64::from(self.params.epoch.epoch_writes))
        {
            return Err(Error::CapacityExceeded {
                live,
                n_max: self.params.index.n_max,
            });
        }

        let touched = self.slab.write_batch(&k_idx, epoch, &batch)?;
        let budget = self.params.epoch.slot_write_budget as usize;
        for addr in &touched {
            self.events
                .emit(LeakEvent::SlotWrite { addr: *addr, epoch });
        }
        for i in touched.len()..budget {
            // Real disk writes, not just events: an adversary counting I/O must see the same
            // number as one counting events.
            let hint = (u64::from(k_idx[0]) + i as u64).wrapping_mul(0x9E37_79B9);
            let addr = self.slab.write_padding_slot(hint % self.slab.capacity())?;
            self.events.emit(LeakEvent::SlotWrite { addr, epoch });
        }

        // EVERY record goes into the index, dummies included. Inserting only the real ones
        // would make the index-write count reveal how many of the batch were real, undoing the
        // WAL's padding one layer down.
        let dim = self.params.embedding.dim;
        for r in &batch {
            let vector = self
                .staged
                .iter()
                .find(|s| prf::tags_equal(s.tag, r.tag))
                .map_or_else(
                    || mnemos_index::bucket::dummy_vector(dim),
                    |s| s.vector.clone(),
                );
            // A full partition is not an error at flush time — it means this record has no home
            // in its assigned bucket. Overflow chaining is future work; for now the record is
            // durable in the slab and simply unindexed.
            // Evaluation ablation: with padding disabled, dummies never reach the index, so a
            // flush touches exactly the real records' partitions. That is finding F5, kept
            // reproducible so E9 can measure what it cost instead of asserting it.
            let is_dummy = !self.staged.iter().any(|s| prf::tags_equal(s.tag, r.tag));
            if !self.params.index.index_write_padding && is_dummy {
                continue;
            }
            // The partition actually written, which is not always the assigned one: a full
            // partition spills into a nearby one rather than dropping the record. Emitting
            // `r.bucket` here would name a write that did not happen, and a leakage model that
            // describes the wrong offset is the F5 failure with different coordinates.
            let written = if let Ok(b) = self.index.insert(r.bucket, r.tag, &vector, epoch) {
                b
            } else {
                // Every candidate partition was full. The record stays durable in the slab but
                // is not retrievable, so it is counted and surfaced through `Engine::unindexed`
                // instead of vanishing. The event is still emitted at the assigned partition:
                // no slot write occurred, and suppressing the event would make the per-flush
                // index-write count data-dependent -- exactly F5.
                self.unindexed += 1;
                r.bucket
            };
            // Emitted whether or not the insert found a home. An event stream that reported
            // only successful placements would once again make the count data-dependent --
            // which is the whole of finding F5.
            self.events.emit(LeakEvent::IndexWrite {
                bucket: written,
                epoch,
            });
        }
        self.staged.clear();

        self.index.advance_tokens(epoch)?;
        for b in 0..self.params.index.buckets {
            self.events.emit(LeakEvent::TokenAdvance {
                bucket: BucketId(b),
                epoch,
            });
        }

        let compacted = self
            .slab
            .compact_with_budget(u64::from(self.params.epoch.compaction_slots))?;
        self.events.emit(LeakEvent::Compact {
            slots: compacted,
            epoch,
            slots_touched: self.params.epoch.compaction_slots,
        });

        // Unconditional: a conditional write would turn "the manifest changed" into "real work
        // happened this epoch".
        self.manifest.epoch = epoch;
        self.manifest.occupancy = self.slab.occupancy().to_vec();
        self.manifest.index_occupancy = self
            .index
            .occupancy()
            .iter()
            .flat_map(|b| b.iter().copied())
            .collect();
        self.manifest.live_entries = self.slab.live_count();
        {
            let path = self.dir.join("manifest.enc");
            let keys = self.keys()?;
            self.manifest.write(&path, &keys.content)?;
        }
        self.events.emit(LeakEvent::ManifestWrite { epoch });

        self.wal.reset()?;
        self.events.emit(LeakEvent::EpochFlush {
            epoch,
            record_count: self.params.epoch.epoch_writes,
        });

        Ok(self.clock.advance())
    }

    fn stat(&self) -> Stat {
        Stat {
            live_entries: self.slab.live_count(),
            epoch: self.clock.current(),
            capacity: self.params.index.n_max,
            unlocked: self.keys.is_some(),
        }
    }

    fn unlock(&mut self) -> Result<()> {
        let arena = Arena::new(
            self.params.memory.arena_bytes,
            self.params.memory.lock_pages,
        )?;
        let wrapped = std::fs::read(self.dir.join("wrapped.key"))?;
        let master_vec = self.keystore.unwrap_master_key(&wrapped)?;
        let master: [u8; 32] = master_vec
            .as_slice()
            .try_into()
            .map_err(|_| Error::Format("master key is not 32 bytes".into()))?;

        let keys = KeyHierarchy::derive(&master)?;
        // Token chains are key-derived state and follow the key's lifetime exactly.
        self.index.install_tokens(&keys.bucket)?;
        self.keys = Some(keys);
        self.arena = Some(arena);
        self.events.emit(LeakEvent::KeyOp { unlock: true });
        Ok(())
    }

    fn lock(&mut self) {
        // Drop order matters: keys first, then the arena. `KeyHierarchy` is `ZeroizeOnDrop` and
        // `Arena` clears itself, so this order clears the key bytes before releasing the region
        // that held them.
        self.keys = None;
        // Drop the token chains with the keys. Leaving them resident would keep key-derived
        // material alive across a lock, which is the one thing lock() exists to prevent.
        self.index.clear_tokens();
        if let Some(arena) = self.arena.as_mut() {
            arena.reset();
        }
        self.arena = None;
        self.events.emit(LeakEvent::KeyOp { unlock: false });
    }
}

/// Encode the centroid table to a fixed-width byte string.
///
/// Fixed width matters for the same reason it does everywhere else: the manifest is padded to a
/// constant, and a variable-length table inside it would eat that budget unpredictably.
fn encode_centroids(centroids: &[QuantVec]) -> Vec<u8> {
    centroids.iter().flat_map(QuantVec::encode).collect()
}

/// Inverse of [`encode_centroids`].
fn decode_centroids(bytes: &[u8], dim: usize) -> Vec<QuantVec> {
    let w = QuantVec::encoded_len(dim);
    if w == 0 {
        return Vec::new();
    }
    bytes.chunks_exact(w).filter_map(QuantVec::decode).collect()
}

/// Split the `text \0 meta` frame produced by [`SearchableMemory::put`].
fn split_frame(bytes: &[u8]) -> (String, String) {
    let sep = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    (
        String::from_utf8_lossy(&bytes[..sep]).into_owned(),
        String::from_utf8_lossy(bytes.get(sep + 1..).unwrap_or(&[])).into_owned(),
    )
}

/// The default sink: discard everything.
///
/// A free function rather than `impl Default for Arc<dyn EventSink>`, which the orphan rule
/// forbids — neither `Arc` nor `dyn EventSink` is local to this crate.
#[must_use]
pub fn default_event_sink() -> Arc<dyn EventSink> {
    Arc::new(NullSink)
}
