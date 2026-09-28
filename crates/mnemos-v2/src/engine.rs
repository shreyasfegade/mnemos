//! The v2 engine.
//!
//! See the crate documentation for the layout and the four rules. This file is where each rule
//! is enforced, and each enforcement point says which rule it serves.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use mnemos_core::error::{Error, Result};
use mnemos_core::event::{EventSink, LeakEvent, NullSink};
use mnemos_core::traits::{KeyStore, SearchableMemory, Stat};
use mnemos_core::types::{Epoch, Hit, Memory, MemoryId, QuantVec, Tag};
use mnemos_core::Params;
use mnemos_crypto::kdf::{derive_key, DerivedKey};
use mnemos_crypto::{prf, Arena};
use parking_lot::Mutex;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use zeroize::Zeroize;

use crate::geometry::{Geometry, ID_LEN, LEN_PREFIX};
use crate::manifest::ManifestV2;
use crate::regions::{FixedFile, GrowingFiles, RegionStore};
use crate::seal::{
    decode_frame, header_aad, open_in_place, random_into, seal_in_place, STAGED_AAD,
};
use crate::table::{build_groups, permutation, Table};
use crate::Ablation;

/// HKDF labels. Distinct from v1's so no v2 key equals any v1 key.
mod info {
    pub const HEADER: &[u8] = b"mnemos/v2/header";
    pub const PAYLOAD: &[u8] = b"mnemos/v2/payload";
    pub const MANIFEST: &[u8] = b"mnemos/v2/manifest";
    pub const TABLE: &[u8] = b"mnemos/v2/table";
    pub const TAG: &[u8] = b"mnemos/v2/tag";
    pub const PERM: &[u8] = b"mnemos/v2/perm";
}

struct Keys {
    header: DerivedKey,
    payload: DerivedKey,
    manifest: DerivedKey,
    table: DerivedKey,
    tag: DerivedKey,
    perm: DerivedKey,
}

impl Keys {
    fn derive(master: &[u8; 32]) -> Result<Self> {
        Ok(Self {
            header: derive_key(master, info::HEADER)?,
            payload: derive_key(master, info::PAYLOAD)?,
            manifest: derive_key(master, info::MANIFEST)?,
            table: derive_key(master, info::TABLE)?,
            tag: derive_key(master, info::TAG)?,
            perm: derive_key(master, info::PERM)?,
        })
    }
}

/// One of the `w` record-carrying slots in a flush schedule.
///
/// The three arms write the same number of bytes to one whole region each; they differ only in
/// how the region is chosen and whether a record lands in it. That is what lets a refusal be
/// indistinguishable from an acceptance.
enum Op {
    /// A staged record, written to its own content-derived region.
    Real(Staged),
    /// A rewrite owed to a refused `put`, written to the region the record would have occupied.
    Ghost(u32),
    /// Padding, written to a uniformly random region.
    Pad,
}

/// A record accepted by `put` and not yet written. Holds ciphertext only.
struct Staged {
    tag: Tag,
    region: u32,
    header_ct: Vec<u8>,
    payload_ct: Vec<u8>,
}

/// Operational counters. In memory only.
#[derive(Clone, Copy, Debug, Default)]
pub struct V2Stats {
    /// Flushes completed since this engine was constructed.
    pub flushes: u64,
    /// Region rewrites performed.
    pub region_writes: u64,
    /// Bytes written to region storage.
    pub region_bytes_written: u64,
    /// Bytes written to the manifest.
    pub manifest_bytes_written: u64,
    /// `put` calls refused because the target partition was full.
    pub refused: u64,
    /// Region rewrites owed to refused `put`s and performed on their behalf, so that a refusal
    /// produces the same bytes an acceptance would have.
    pub ghost_writes: u64,
    /// Records placed outside their assigned partition (spill ablation only).
    pub spilled: u64,
    /// Tombstoned records physically removed.
    pub purged: u64,
    /// Regions read.
    pub region_reads: u64,
}

/// A Mnemos v2 store.
pub struct EngineV2 {
    dir: PathBuf,
    params: Params,
    geo: Geometry,
    abl: Ablation,
    keystore: Box<dyn KeyStore>,
    store: Box<dyn RegionStore>,
    events: Arc<dyn EventSink>,
    periodic: bool,
    epoch_ms: Duration,
    durable: bool,

    keys: Option<Keys>,
    arena: Mutex<Option<Arena>>,
    table: Option<Table>,
    region_of: Vec<u32>,
    logical_of: Vec<u32>,

    occupancy: Vec<bool>,
    slot_tag: Vec<[u8; 16]>,
    where_tag: HashMap<[u8; 16], (u32, u32)>,
    tombstones: Vec<[u8; 16]>,
    tomb_set: HashSet<[u8; 16]>,
    /// Accepted records and the rewrites owed to refused ones, in **one queue, in arrival
    /// order**.
    ///
    /// A refusal that wrote nothing would be visible: an observer who has labelled regions sees
    /// that the region it expected to receive the write did not, and so learns that the topic's
    /// live load has reached `C`. Because cohort P's contribution to that load is something the
    /// observer already watches, the remainder is cohort H's load on the topic, which is exactly
    /// the asset G5 protects.
    ///
    /// Owing the rewrite is necessary but not sufficient: the debt has to be paid *on the same
    /// schedule*. Holding refusals in a second queue that a flush drains only after the accepted
    /// records lets a refusal-heavy history build a backlog and pay for it in extra flushes
    /// later, which is itself a distinguisher — measured at advantage 0.32 before this became one
    /// queue. Arrival order makes an accepted and a refused `put` consume the same slot of the
    /// same flush.
    pending: Vec<Op>,
    dirty: bool,

    epoch: u64,
    cursor: u32,
    last_flush: Instant,
    stats: V2Stats,
}

fn arena_size(geo: &Geometry, params: &Params) -> usize {
    params
        .memory
        .arena_bytes
        .max(geo.header_pt + geo.payload_pt + 4096)
}

impl EngineV2 {
    /// Create a store. Starts **unlocked**.
    ///
    /// `centroids` is the logical partition table, one vector per partition.
    ///
    /// # Errors
    /// Parameter, key-custody, crypto or I/O failures.
    pub fn create(
        dir: &Path,
        params: Params,
        keystore: Box<dyn KeyStore>,
        centroids: Vec<QuantVec>,
        abl: Ablation,
    ) -> Result<Self> {
        let geo = Geometry::from_params(&params)?;
        abl.validate()?;
        if centroids.len() != geo.buckets as usize {
            return Err(Error::Format(format!(
                "expected {} centroids, got {}",
                geo.buckets,
                centroids.len()
            )));
        }
        std::fs::create_dir_all(dir)?;

        let mut master = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut master);
        let wrapped = keystore.wrap_master_key(&master)?;
        let keys = Keys::derive(&master)?;
        master.zeroize();

        let groups = if geo.grouped {
            build_groups(&centroids, geo.group_size as usize)
        } else {
            (0..geo.buckets).collect()
        };
        let table = Table { centroids, groups };
        table.write(dir, &keys.table.0)?;

        let store: Box<dyn RegionStore> = if abl.growing_regions {
            Box::new(GrowingFiles::create(dir, geo)?)
        } else {
            Box::new(FixedFile::create(dir, geo)?)
        };
        let manifest = ManifestV2::new(geo.slots());
        manifest.write(
            dir,
            &keys.manifest.0,
            geo.tombstone_capacity,
            params.v2.durable,
        )?;
        std::fs::write(dir.join("wrapped.key"), &wrapped)?;

        let region_of = permutation(
            &keys.perm.0,
            geo.buckets,
            abl.identity_permutation || !params.v2.permute_partitions,
        );
        let logical_of = invert(&region_of);
        let arena = Arena::new(arena_size(&geo, &params), params.memory.lock_pages)?;

        let events: Arc<dyn EventSink> = Arc::new(NullSink);
        events.emit(LeakEvent::Setup {
            n_max: geo.slots() as u64,
            buckets: geo.buckets,
            bucket_capacity: geo.capacity,
            block_size: params.crypto.block_size_bytes,
        });

        Ok(Self {
            dir: dir.to_path_buf(),
            periodic: params.v2.flush_policy == "periodic",
            epoch_ms: Duration::from_millis(params.epoch.epoch_ms),
            durable: params.v2.durable,
            params,
            geo,
            abl,
            keystore,
            store,
            events,
            keys: Some(keys),
            arena: Mutex::new(Some(arena)),
            table: Some(table),
            region_of,
            logical_of,
            occupancy: vec![false; geo.slots()],
            slot_tag: vec![[0u8; 16]; geo.slots()],
            where_tag: HashMap::new(),
            tombstones: Vec::new(),
            tomb_set: HashSet::new(),
            pending: Vec::new(),
            dirty: false,
            epoch: 0,
            cursor: 0,
            last_flush: Instant::now(),
            stats: V2Stats::default(),
        })
    }

    /// Open an existing store. Starts **locked**.
    ///
    /// # Errors
    /// Parameter or I/O failures.
    pub fn open(
        dir: &Path,
        params: Params,
        keystore: Box<dyn KeyStore>,
        abl: Ablation,
    ) -> Result<Self> {
        let geo = Geometry::from_params(&params)?;
        abl.validate()?;
        let store: Box<dyn RegionStore> = if abl.growing_regions {
            Box::new(GrowingFiles::open(dir, geo)?)
        } else {
            Box::new(FixedFile::open(dir, geo)?)
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            periodic: params.v2.flush_policy == "periodic",
            epoch_ms: Duration::from_millis(params.epoch.epoch_ms),
            durable: params.v2.durable,
            params,
            geo,
            abl,
            keystore,
            store,
            events: Arc::new(NullSink),
            keys: None,
            arena: Mutex::new(None),
            table: None,
            region_of: Vec::new(),
            logical_of: Vec::new(),
            occupancy: vec![false; geo.slots()],
            slot_tag: vec![[0u8; 16]; geo.slots()],
            where_tag: HashMap::new(),
            tombstones: Vec::new(),
            tomb_set: HashSet::new(),
            pending: Vec::new(),
            dirty: false,
            epoch: 0,
            cursor: 0,
            last_flush: Instant::now(),
            stats: V2Stats::default(),
        })
    }

    /// Replace the event sink.
    pub fn set_event_sink(&mut self, sink: Arc<dyn EventSink>) {
        self.events = sink;
    }

    /// Geometry in force.
    #[must_use]
    pub const fn geometry(&self) -> &Geometry {
        &self.geo
    }

    /// Operational counters.
    #[must_use]
    pub const fn stats(&self) -> V2Stats {
        self.stats
    }

    /// Operations accepted and not yet written: records, and rewrites owed to refusals.
    ///
    /// The two are counted together on purpose. A caller draining the store must drain both, or a
    /// refusal-heavy history leaves a backlog that a later flush pays for, which is observable.
    #[must_use]
    pub fn staged_len(&self) -> usize {
        self.pending.len()
    }

    /// Deletions recorded and not yet purged.
    #[must_use]
    pub fn tombstones_len(&self) -> usize {
        self.tombstones.len()
    }

    /// Live records in physical region `region`, including tombstoned ones not yet purged.
    #[must_use]
    pub fn region_load(&self, region: u32) -> u32 {
        let base = region as usize * self.geo.capacity as usize;
        self.occupancy[base..base + self.geo.capacity as usize]
            .iter()
            .filter(|o| **o)
            .count() as u32
    }

    /// **Evaluation ground truth only.** The physical region a vector would be assigned to.
    ///
    /// The adversary never has this; experiment harnesses use it to score attacks, exactly as
    /// v1 experiments score against the partition assignment.
    ///
    /// # Errors
    /// [`Error::Locked`].
    pub fn ground_truth_region(&self, v: &QuantVec) -> Result<u32> {
        let t = self.table.as_ref().ok_or(Error::Locked)?;
        Ok(self.region_of[t.assign(v) as usize])
    }

    /// **Evaluation ground truth only.** Physical region of each logical partition.
    #[must_use]
    pub fn ground_truth_permutation(&self) -> Vec<u32> {
        self.region_of.clone()
    }

    fn keys(&self) -> Result<&Keys> {
        self.keys.as_ref().ok_or(Error::Locked)
    }

    fn cap(&self) -> usize {
        self.geo.capacity as usize
    }

    /// Accepted records queued for `region`. A ghost occupies no slot, so it is not counted.
    fn staged_to(&self, region: u32) -> u32 {
        self.pending
            .iter()
            .filter(|op| matches!(op, Op::Real(s) if s.region == region))
            .count() as u32
    }

    /// Flush if the schedule says one is due.
    ///
    /// Under `periodic`, exactly one flush per `epoch_ms`, whether or not anything is staged.
    /// Under `on_demand`, a flush only if records are staged and `epoch_ms` has passed.
    ///
    /// # Errors
    /// As [`SearchableMemory::flush`].
    pub fn tick(&mut self) -> Result<Option<Epoch>> {
        let due = self.last_flush.elapsed() >= self.epoch_ms;
        let pending = !self.pending.is_empty() || self.dirty;
        // Periodic stores flush on the clock alone; the others only when something is owed.
        if due && (self.periodic || pending) {
            return self.flush().map(Some);
        }
        Ok(None)
    }

    /// Flush anything pending, optionally re-encrypt the whole file, then lock.
    ///
    /// The staged-remainder rule: a record accepted before a lock is on disk before the lock
    /// returns. The full rewrite is the session-boundary defence against an adversary who holds
    /// two file versions: see [`EngineV2::rewrite_all`].
    ///
    /// # Errors
    /// As [`SearchableMemory::flush`].
    pub fn close(&mut self) -> Result<()> {
        while self.keys.is_some() && !self.pending.is_empty() {
            self.flush()?;
        }
        if self.keys.is_some() && self.dirty {
            self.flush()?;
        }
        if self.keys.is_some() && self.params.v2.rewrite_at_lock {
            self.rewrite_all()?;
        }
        self.lock();
        Ok(())
    }

    /// Re-encrypt every region under fresh nonces.
    ///
    /// History independence concerns *one* representation against the operation sequence that
    /// produced it; it says nothing about a *sequence* of representations. An adversary holding
    /// two versions of the file from times `t1 < t2` diffs them and reads off exactly the regions
    /// rewritten in between, which is the write channel applied before τ. Re-encrypting
    /// everything at a session boundary makes any diff that spans the boundary total, and so
    /// uninformative. It does not help against an adversary that captures versions more finely
    /// than once per session, which is why the version granularity of a real client is measured
    /// rather than assumed.
    ///
    /// # Errors
    /// Crypto or I/O failures.
    pub fn rewrite_all(&mut self) -> Result<()> {
        let e = self.epoch;
        for region in 0..self.geo.buckets {
            self.rewrite(region, None, e)?;
        }
        self.write_manifest()?;
        if self.durable {
            self.store.sync()?;
        }
        Ok(())
    }

    /// The logical partitions a query reads. Constant count by construction.
    fn probe_set(&self, table: &Table, q: &QuantVec) -> Vec<u32> {
        let ranked = table.ranked(q);
        let k = self.geo.probe_width as usize;
        if self.geo.grouped {
            let mut counts: HashMap<u32, (usize, usize)> = HashMap::new();
            for (rank, (l, _)) in ranked.iter().take(k).enumerate() {
                let g = table.groups[*l as usize];
                let e = counts.entry(g).or_insert((0, rank));
                e.0 += 1;
            }
            let best = counts
                .iter()
                .max_by(|a, b| a.1 .0.cmp(&b.1 .0).then(b.1 .1.cmp(&a.1 .1)))
                .map_or(0, |(g, _)| *g);
            return (0..self.geo.buckets)
                .filter(|l| table.groups[*l as usize] == best)
                .collect();
        }
        let want = (self.geo.probe_width * self.geo.probe_spread) as usize;
        let randomised = self.geo.probe_spread > 1 || self.geo.probe_retention < 1.0;
        let mut set: Vec<u32> = Vec::with_capacity(want);
        if !randomised {
            set.extend(ranked.iter().take(k).map(|(l, _)| *l));
            return set;
        }
        let mut rng = ChaCha20Rng::from_entropy();
        for (l, _) in ranked.iter().take(k) {
            if self.geo.probe_retention >= 1.0 || rng.gen::<f32>() < self.geo.probe_retention {
                set.push(*l);
            }
        }
        while set.len() < want {
            let c = rng.gen_range(0..self.geo.buckets);
            if !set.contains(&c) {
                set.push(c);
            }
        }
        set
    }

    /// Rewrite one physical region. **Rule 1.**
    ///
    /// Purges tombstoned slots, places `place` in a uniformly random free slot, re-encrypts every
    /// live slot under fresh nonces, and refreshes every free slot with random bytes. Under the
    /// `slot_writes_only` ablation only the placed and freed slots reach disk.
    fn rewrite(&mut self, region: u32, place: Option<&Staged>, epoch: u64) -> Result<()> {
        let geo = self.geo;
        let cap = self.cap();
        let base = region as usize * cap;
        let sl = geo.slot_len();
        let (hct, hpt, ppt) = (geo.header_ct(), geo.header_pt, geo.payload_pt);
        let (k_header, k_payload) = {
            let k = self.keys()?;
            (k.header.0, k.payload.0)
        };

        let mut current = vec![0u8; geo.region_len()];
        self.store
            .read_region(region, &self.occupancy[base..base + cap], &mut current)?;

        let freed = self.purge(base, cap);

        let target: Option<usize> = if place.is_some() {
            Some(self.choose_slot(base, cap)?)
        } else {
            None
        };

        let mut image: Vec<u8> = Vec::with_capacity(geo.region_len());
        let mut guard = self.arena.lock();
        let arena = guard.as_mut().ok_or(Error::Locked)?;
        let result = (|| -> Result<()> {
            let buf = arena.buffer_mut();
            let (hs, rest) = buf.split_at_mut(hpt);
            let ps = &mut rest[..ppt];
            for i in 0..cap {
                let aad = header_aad(region, i as u32);
                if Some(i) == target {
                    let st = place.expect("target implies a record");
                    open_in_place(&k_header, &st.header_ct, STAGED_AAD, hs)?;
                    self.write_header(hs, &aad, &k_header, &mut image)?;
                    image.extend_from_slice(&st.payload_ct);
                } else if self.occupancy[base + i] {
                    let slot = &current[i * sl..(i + 1) * sl];
                    self.read_header(&slot[..hct], &aad, &k_header, hs)?;
                    self.write_header(hs, &aad, &k_header, &mut image)?;
                    let tag = self.slot_tag[base + i];
                    open_in_place(&k_payload, &slot[hct..], &tag, ps)?;
                    seal_in_place(&k_payload, ps, &tag, &mut image)?;
                } else if self.abl.unsealed_slots {
                    image.resize(image.len() + sl, 0);
                } else {
                    random_into(&mut image, sl);
                }
            }
            Ok(())
        })();
        arena.wipe();
        drop(guard);
        result?;

        if let (Some(i), Some(st)) = (target, place) {
            self.occupancy[base + i] = true;
            self.slot_tag[base + i] = st.tag.0;
            self.where_tag.insert(st.tag.0, (region, i as u32));
        }

        let written = if self.abl.slot_writes_only {
            let mut slots: Vec<u32> = freed.clone();
            if let Some(i) = target {
                slots.push(i as u32);
            }
            slots.sort_unstable();
            let mut n = 0u64;
            for s in &slots {
                let s_us = *s as usize;
                n += self
                    .store
                    .write_slot(region, *s, &image[s_us * sl..(s_us + 1) * sl])?;
            }
            if !slots.is_empty() {
                self.events.emit(LeakEvent::RegionWrite {
                    region,
                    epoch: Epoch(epoch),
                    slots,
                });
            }
            n
        } else {
            let n = self
                .store
                .write_region(region, &image, &self.occupancy[base..base + cap])?;
            self.events.emit(LeakEvent::RegionWrite {
                region,
                epoch: Epoch(epoch),
                slots: Vec::new(),
            });
            n
        };
        self.stats.region_writes += 1;
        self.stats.region_bytes_written += written;
        Ok(())
    }

    /// Free every tombstoned slot of the region starting at `base`, returning their indices.
    ///
    /// Only the in-memory state changes here; the caller's whole-region rewrite is what removes
    /// the records from disk. That the purge must stay whole-region, even where placement need
    /// not, is the boundary Lemma `lem:slot` states.
    fn purge(&mut self, base: usize, cap: usize) -> Vec<u32> {
        let mut freed: Vec<u32> = Vec::new();
        for i in 0..cap {
            if self.occupancy[base + i] && self.tomb_set.contains(&self.slot_tag[base + i]) {
                let t = self.slot_tag[base + i];
                self.occupancy[base + i] = false;
                self.where_tag.remove(&t);
                self.tomb_set.remove(&t);
                self.tombstones.retain(|x| *x != t);
                self.slot_tag[base + i] = [0u8; 16];
                freed.push(i as u32);
                self.stats.purged += 1;
            }
        }
        freed
    }

    /// The slot a record takes within its region.
    ///
    /// Uniform over the free slots. Under a per-slot write this is what keeps the index of the
    /// slot written independent of how full the region was (Lemma `lem:slot`); `first_free_slot`
    /// is the ablation that shows what happens without it.
    fn choose_slot(&self, base: usize, cap: usize) -> Result<usize> {
        let free: Vec<usize> = (0..cap).filter(|i| !self.occupancy[base + i]).collect();
        let full = || Error::CapacityExceeded {
            live: cap as u64,
            n_max: cap as u64,
        };
        if self.abl.first_free_slot {
            free.first().copied().ok_or_else(full)
        } else {
            free.choose(&mut rand::rngs::OsRng)
                .copied()
                .ok_or_else(full)
        }
    }

    /// Header plaintext out of a stored header: authenticate it, or copy it under the S2 ablation.
    fn read_header(
        &self,
        stored: &[u8],
        aad: &[u8],
        key: &[u8; 32],
        scratch: &mut [u8],
    ) -> Result<()> {
        if self.abl.unsealed_slots {
            scratch.copy_from_slice(&stored[..scratch.len()]);
            Ok(())
        } else {
            open_in_place(key, stored, aad, scratch)
        }
    }

    /// Header plaintext to a stored header: seal it, or write it raw under the S2 ablation.
    fn write_header(
        &self,
        scratch: &mut [u8],
        aad: &[u8],
        key: &[u8; 32],
        out: &mut Vec<u8>,
    ) -> Result<()> {
        if self.abl.unsealed_slots {
            out.extend_from_slice(scratch);
            out.resize(out.len() + self.geo.header_ct() - scratch.len(), 0);
            scratch.zeroize();
            Ok(())
        } else {
            seal_in_place(key, scratch, aad, out)
        }
    }

    fn write_manifest(&mut self) -> Result<()> {
        let m = ManifestV2 {
            epoch: self.epoch,
            cursor: self.cursor,
            occupancy: self.occupancy.clone(),
            tombstones: self.tombstones.clone(),
        };
        let key = self.keys()?.manifest.0;
        let n = m.write(&self.dir, &key, self.geo.tombstone_capacity, self.durable)?;
        self.stats.manifest_bytes_written += n;
        self.events.emit(LeakEvent::ManifestWrite {
            epoch: Epoch(self.epoch),
        });
        Ok(())
    }

    /// The unlock scan: read every region and rebuild the identity map. Reads everything, so
    /// what it reveals is that an unlock happened.
    fn rebuild_maps(&mut self) -> Result<()> {
        let geo = self.geo;
        let cap = self.cap();
        let sl = geo.slot_len();
        let k_header = self.keys()?.header.0;
        let k_tag = self.keys()?.tag.clone();
        self.where_tag.clear();
        let mut buf = vec![0u8; geo.region_len()];
        for region in 0..geo.buckets {
            let base = region as usize * cap;
            self.events.emit(LeakEvent::RegionRead { region });
            self.stats.region_reads += 1;
            self.store
                .read_region(region, &self.occupancy[base..base + cap], &mut buf)?;
            let found: Vec<(usize, [u8; 16])> = {
                let mut guard = self.arena.lock();
                let arena = guard.as_mut().ok_or(Error::Locked)?;
                let r = (|| -> Result<Vec<(usize, [u8; 16])>> {
                    let hs = &mut arena.buffer_mut()[..geo.header_pt];
                    let mut out = Vec::new();
                    for i in 0..cap {
                        if !self.occupancy[base + i] {
                            continue;
                        }
                        let aad = header_aad(region, i as u32);
                        self.read_header(
                            &buf[i * sl..i * sl + geo.header_ct()],
                            &aad,
                            &k_header,
                            hs,
                        )?;
                        let mut id = [0u8; 16];
                        id.copy_from_slice(&hs[..ID_LEN]);
                        out.push((i, prf::tag_for(&k_tag, MemoryId(id), 0).0));
                    }
                    Ok(out)
                })();
                arena.wipe();
                r?
            };
            for (i, tag) in found {
                self.slot_tag[base + i] = tag;
                self.where_tag.insert(tag, (region, i as u32));
            }
        }
        Ok(())
    }
}

fn invert(p: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; p.len()];
    for (l, r) in p.iter().enumerate() {
        inv[*r as usize] = l as u32;
    }
    inv
}

/// Cosine between a query and a header's raw codes, without materialising a vector on the heap.
fn score_raw(q: &QuantVec, codes: &[u8], scale: f32) -> f32 {
    let mut acc: i32 = 0;
    for (a, b) in q.codes.iter().zip(codes.iter()) {
        acc += i32::from(*a) * i32::from(*b as i8);
    }
    acc as f32 * q.scale * scale
}

impl SearchableMemory for EngineV2 {
    fn put(&mut self, text: &str, meta: &str, embedding: &QuantVec) -> Result<MemoryId> {
        let geo = self.geo;
        if embedding.codes.len() != geo.dim {
            return Err(Error::Format(format!(
                "vector has {} dimensions, store expects {}",
                embedding.codes.len(),
                geo.dim
            )));
        }
        if text.len() + meta.len() > geo.max_record_bytes() {
            return Err(Error::Format(format!(
                "record of {} bytes exceeds the slot payload of {} bytes",
                text.len() + meta.len(),
                geo.max_record_bytes()
            )));
        }
        if self.periodic && self.pending.len() >= self.params.v2.max_staged as usize {
            return Err(Error::CapacityExceeded {
                live: self.pending.len() as u64,
                n_max: u64::from(self.params.v2.max_staged),
            });
        }
        let table = self.table.as_ref().ok_or(Error::Locked)?;

        // Admission. Condition S4 as an admissibility rule: a record whose partition cannot hold
        // it is refused before anything is written, so a full partition never changes what a
        // flush writes. The spill ablation instead tries the two next-nearest partitions.
        let ranked = table.ranked(embedding);
        let tries = if self.abl.spill { 3 } else { 1 };
        let mut chosen: Option<(u32, usize)> = None;
        for (depth, (l, _)) in ranked.iter().take(tries).enumerate() {
            let region = self.region_of[*l as usize];
            if self.region_load(region) + self.staged_to(region) < geo.capacity {
                chosen = Some((region, depth));
                break;
            }
        }
        let intended = self.region_of[ranked.first().map_or(0, |(l, _)| *l) as usize];
        let Some((region, depth)) = chosen else {
            // The record is refused to the caller either way. What differs is what the disk
            // shows. Under `refusal_mode = "padded"` the store still owes the intended region a
            // rewrite and still counts the call towards the flush trigger, so neither the bytes
            // written nor the moment they are written distinguishes a refusal from an
            // acceptance. Under `"silent"` nothing is owed, which reproduces the channel that
            // tells an observer a topic's live load has reached capacity.
            self.stats.refused += 1;
            if self.params.v2.refusal_mode == "padded" {
                self.pending.push(Op::Ghost(intended));
                self.stats.ghost_writes += 1;
                if !self.periodic
                    && (self.pending.len() >= geo.epoch_writes as usize
                        || self.last_flush.elapsed() >= self.epoch_ms)
                {
                    self.flush()?;
                }
            }
            return Err(Error::CapacityExceeded {
                live: u64::from(geo.capacity),
                n_max: u64::from(geo.capacity),
            });
        };

        let id = MemoryId::random();
        let (tag, header_ct, payload_ct) = {
            let keys = self.keys()?;
            let tag = prf::tag_for(&keys.tag, id, 0);
            let mut guard = self.arena.lock();
            let arena = guard.as_mut().ok_or(Error::Locked)?;
            let mut header_ct = Vec::with_capacity(geo.header_ct());
            let mut payload_ct = Vec::with_capacity(geo.payload_ct());
            let r = (|| -> Result<()> {
                let buf = arena.buffer_mut();
                let (hs, rest) = buf.split_at_mut(geo.header_pt);
                let ps = &mut rest[..geo.payload_pt];
                let n = text.len() + 1 + meta.len();
                ps[..LEN_PREFIX].copy_from_slice(&(n as u32).to_be_bytes());
                ps[LEN_PREFIX..LEN_PREFIX + text.len()].copy_from_slice(text.as_bytes());
                ps[LEN_PREFIX + text.len()] = 0;
                ps[LEN_PREFIX + text.len() + 1..LEN_PREFIX + n].copy_from_slice(meta.as_bytes());
                seal_in_place(&keys.payload.0, ps, tag.as_bytes(), &mut payload_ct)?;
                hs[..ID_LEN].copy_from_slice(&id.0);
                for (d, c) in hs[ID_LEN..ID_LEN + geo.dim]
                    .iter_mut()
                    .zip(embedding.codes.iter())
                {
                    *d = *c as u8;
                }
                hs[ID_LEN + geo.dim..].copy_from_slice(&embedding.scale.to_le_bytes());
                seal_in_place(&keys.header.0, hs, STAGED_AAD, &mut header_ct)?;
                Ok(())
            })();
            arena.wipe();
            r?;
            (tag, header_ct, payload_ct)
        };
        if depth > 0 {
            self.stats.spilled += 1;
        }
        self.pending.push(Op::Real(Staged {
            tag,
            region,
            header_ct,
            payload_ct,
        }));

        if !self.periodic
            && (self.pending.len() >= geo.epoch_writes as usize
                || self.last_flush.elapsed() >= self.epoch_ms)
        {
            self.flush()?;
        }
        Ok(id)
    }

    fn search(&self, query: &QuantVec) -> Result<Vec<Hit>> {
        let geo = self.geo;
        let keys = self.keys()?;
        let table = self.table.as_ref().ok_or(Error::Locked)?;
        let cap = self.cap();
        let sl = geo.slot_len();

        let mut regions: Vec<u32> = self
            .probe_set(table, query)
            .iter()
            .map(|l| self.region_of[*l as usize])
            .collect();
        regions.sort_unstable();

        // Rule 3: whole regions, and nothing after ranking.
        let mut bufs: Vec<Vec<u8>> = Vec::with_capacity(regions.len());
        for region in &regions {
            let base = *region as usize * cap;
            self.events.emit(LeakEvent::RegionRead { region: *region });
            let mut b = vec![0u8; geo.region_len()];
            self.store
                .read_region(*region, &self.occupancy[base..base + cap], &mut b)?;
            bufs.push(b);
        }

        let mut guard = self.arena.lock();
        let arena = guard.as_mut().ok_or(Error::Locked)?;
        let result = (|| -> Result<Vec<Hit>> {
            let buf = arena.buffer_mut();
            let (hs, rest) = buf.split_at_mut(geo.header_pt);
            let ps = &mut rest[..geo.payload_pt];

            let mut cands: Vec<(f32, usize, u32, [u8; 16])> = Vec::new();
            for (bi, region) in regions.iter().enumerate() {
                let base = *region as usize * cap;
                for i in 0..cap {
                    if !self.occupancy[base + i] || self.tomb_set.contains(&self.slot_tag[base + i])
                    {
                        continue;
                    }
                    let aad = header_aad(*region, i as u32);
                    self.read_header(
                        &bufs[bi][i * sl..i * sl + geo.header_ct()],
                        &aad,
                        &keys.header.0,
                        hs,
                    )?;
                    let scale = f32::from_le_bytes(
                        hs[ID_LEN + geo.dim..]
                            .try_into()
                            .map_err(|_| Error::Crypto)?,
                    );
                    let score = score_raw(query, &hs[ID_LEN..ID_LEN + geo.dim], scale);
                    let mut id = [0u8; 16];
                    id.copy_from_slice(&hs[..ID_LEN]);
                    hs.zeroize();
                    cands.push((score, bi, i as u32, id));
                }
            }
            cands.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(regions[a.1].cmp(&regions[b.1]))
                    .then(a.2.cmp(&b.2))
            });
            cands.truncate(geo.top_k);

            let mut hits = Vec::with_capacity(cands.len());
            for (score, bi, slot, id) in cands {
                let region = regions[bi];
                if self.abl.result_reads {
                    self.events.emit(LeakEvent::RegionSlotRead { region, slot });
                    let base = region as usize * cap;
                    let mut scratch = vec![0u8; sl];
                    self.store.read_slot(
                        region,
                        slot,
                        &self.occupancy[base..base + cap],
                        &mut scratch,
                    )?;
                }
                let s = slot as usize;
                let tag = self.slot_tag[region as usize * cap + s];
                open_in_place(
                    &keys.payload.0,
                    &bufs[bi][s * sl + geo.header_ct()..(s + 1) * sl],
                    &tag,
                    ps,
                )?;
                let decoded = decode_frame(ps);
                ps.zeroize();
                let (text, meta) = decoded?;
                hits.push(Hit {
                    memory: Memory {
                        text,
                        meta,
                        id: MemoryId(id),
                    },
                    score,
                });
            }
            Ok(hits)
        })();
        arena.wipe();
        result
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
        let tag = prf::tag_for(&self.keys()?.tag, id, 0);
        if let Some(pos) = self
            .pending
            .iter()
            .position(|op| matches!(op, Op::Real(s) if prf::tags_equal(s.tag, tag)))
        {
            // The record never reached disk, so drop it. The queue slot becomes a ghost rather
            // than vanishing, so deleting a staged record does not shorten the flush it was in.
            self.pending[pos] = Op::Ghost(rand::rngs::OsRng.gen_range(0..self.geo.buckets));
            return Ok(());
        }
        let Some((region, _)) = self.where_tag.get(&tag.0).copied() else {
            return Err(Error::NotFound);
        };
        if self.tomb_set.contains(&tag.0) {
            return Err(Error::NotFound);
        }
        if self.tombstones.len() >= self.geo.tombstone_capacity {
            return Err(Error::CapacityExceeded {
                live: self.tombstones.len() as u64,
                n_max: self.geo.tombstone_capacity as u64,
            });
        }
        self.tombstones.push(tag.0);
        self.tomb_set.insert(tag.0);
        self.dirty = true;
        if self.abl.immediate_delete {
            // Ablation: v1's behaviour. The region holding the record is rewritten now, which
            // names its partition at the moment of deletion.
            let e = self.epoch;
            self.rewrite(region, None, e)?;
            self.write_manifest()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<Epoch> {
        self.keys()?;
        let geo = self.geo;
        let e = self.epoch;
        let w = geo.epoch_writes as usize;

        // Rule 2: exactly w record slots, taken from the pending queue **in arrival order** so a
        // refused put consumes the same slot an accepted one would have, then padding, in a fresh
        // random order.
        let take = self.pending.len().min(w);
        let mut ops: Vec<Op> = self.pending.drain(..take).collect();
        ops.resize_with(w, || Op::Pad);
        let mut rng = ChaCha20Rng::from_entropy();
        ops.shuffle(&mut rng);

        for op in ops {
            match op {
                Op::Real(st) => {
                    let region = st.region;
                    self.rewrite(region, Some(&st), e)?;
                }
                Op::Ghost(region) => self.rewrite(region, None, e)?,
                Op::Pad => {
                    if self.abl.no_padding {
                        continue;
                    }
                    let region = rng.gen_range(0..geo.buckets);
                    self.rewrite(region, None, e)?;
                }
            }
        }

        // The maintenance rewrite. Its region depends only on the flush count, and it bounds
        // how long a tombstoned record survives on disk to B flushes.
        let c = self.cursor;
        self.rewrite(c, None, e)?;
        self.cursor = (self.cursor + 1) % geo.buckets;

        self.epoch += 1;
        self.write_manifest()?;
        if self.durable {
            self.store.sync()?;
        }
        self.events.emit(LeakEvent::EpochFlush {
            epoch: Epoch(e),
            record_count: geo.epoch_writes,
        });
        self.stats.flushes += 1;
        self.dirty = false;
        self.last_flush = Instant::now();
        Ok(Epoch(self.epoch))
    }

    fn stat(&self) -> Stat {
        let live = self.occupancy.iter().filter(|o| **o).count() as u64;
        Stat {
            live_entries: live - self.tombstones.len() as u64,
            epoch: Epoch(self.epoch),
            capacity: self.geo.slots() as u64,
            unlocked: self.keys.is_some(),
        }
    }

    fn unlock(&mut self) -> Result<()> {
        let wrapped = std::fs::read(self.dir.join("wrapped.key"))?;
        let master_vec = self.keystore.unwrap_master_key(&wrapped)?;
        let mut master: [u8; 32] = master_vec
            .as_slice()
            .try_into()
            .map_err(|_| Error::Format("master key is not 32 bytes".into()))?;
        let keys = Keys::derive(&master)?;
        master.zeroize();

        let table = Table::read(&self.dir, &keys.table.0)?;
        if table.centroids.len() != self.geo.buckets as usize {
            return Err(Error::Format("table size does not match params".into()));
        }
        let m = ManifestV2::read(
            &self.dir,
            &keys.manifest.0,
            self.geo.slots(),
            self.geo.tombstone_capacity,
        )?;
        self.region_of = permutation(
            &keys.perm.0,
            self.geo.buckets,
            self.abl.identity_permutation || !self.params.v2.permute_partitions,
        );
        self.logical_of = invert(&self.region_of);
        self.epoch = m.epoch;
        // The cursor advances one region per flush, so its resting position counts the flushes
        // the store has performed, modulo B. That count is the temporal-activity asset, and an
        // observer who identifies the cursor across a few post-τ flushes can read it off. Drawing
        // a fresh start at each unlock removes the inference; the B-flush deletion bound is a
        // per-session property and is unaffected.
        self.cursor = if self.params.v2.randomise_cursor {
            rand::rngs::OsRng.gen_range(0..self.geo.buckets)
        } else {
            m.cursor
        };
        self.occupancy = m.occupancy;
        self.tomb_set = m.tombstones.iter().copied().collect();
        self.tombstones = m.tombstones;
        self.table = Some(table);
        self.keys = Some(keys);
        *self.arena.lock() = Some(Arena::new(
            arena_size(&self.geo, &self.params),
            self.params.memory.lock_pages,
        )?);
        self.rebuild_maps()?;
        self.last_flush = Instant::now();
        self.events.emit(LeakEvent::KeyOp { unlock: true });
        Ok(())
    }

    fn lock(&mut self) {
        // Best effort; `close` is the fallible form. Drain the whole queue, not one flush of it.
        while self.keys.is_some() && !self.pending.is_empty() {
            if self.flush().is_err() {
                break;
            }
        }
        if self.keys.is_some() && self.dirty {
            let _ = self.flush();
        }
        self.keys = None;
        self.table = None;
        *self.arena.lock() = None;
        self.pending.clear();
        self.where_tag.clear();
        self.slot_tag.iter_mut().for_each(Zeroize::zeroize);
        self.events.emit(LeakEvent::KeyOp { unlock: false });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnemos_core::types::Embedding;
    use mnemos_keystore::software::SoftwareKeystore;

    pub(crate) fn params(b: u32, c: u32, dim: usize) -> Params {
        let mut p = Params::discover().expect("params.toml");
        p.index.buckets = b;
        p.index.bucket_capacity = c;
        p.index.probe_width = 2;
        p.index.probe_spread = 2;
        p.index.top_k = 3;
        p.embedding.dim = dim;
        p.crypto.block_size_bytes = 256;
        p.epoch.epoch_writes = 4;
        p.epoch.epoch_ms = 3_600_000;
        p.memory.arena_bytes = 64 * 1024;
        p.v2.durable = false;
        p.v2.tombstone_capacity = 16;
        p
    }

    pub(crate) fn unit(i: usize, dim: usize) -> QuantVec {
        let mut v = vec![0.02f32; dim];
        v[i % dim] = 1.0;
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        QuantVec::quantise(&Embedding(v.into_iter().map(|x| x / n).collect()))
    }

    fn engine(dir: &Path, p: Params, abl: Ablation) -> EngineV2 {
        let ks = SoftwareKeystore::open_at(&dir.join("k"), true).unwrap();
        let cents = (0..p.index.buckets as usize)
            .map(|i| unit(i, p.embedding.dim))
            .collect();
        EngineV2::create(dir, p, Box::new(ks), cents, abl).unwrap()
    }

    #[test]
    fn put_search_delete_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let p = params(8, 8, 8);
        let mut e = engine(dir.path(), p.clone(), Ablation::default());
        let a = e.put("alpha", "{\"t\":1}", &unit(0, 8)).unwrap();
        let b = e.put("beta", "{}", &unit(3, 8)).unwrap();
        e.flush().unwrap();
        let hits = e.search(&unit(0, 8)).unwrap();
        assert_eq!(hits[0].memory.text, "alpha");
        assert_eq!(hits[0].memory.meta, "{\"t\":1}");
        assert_eq!(hits[0].memory.id, a);

        e.delete(a).unwrap();
        assert!(
            e.search(&unit(0, 8))
                .unwrap()
                .iter()
                .all(|h| h.memory.id != a),
            "tombstoned record returned"
        );
        assert!(e.delete(a).is_err());
        e.close().unwrap();

        let ks = SoftwareKeystore::open_at(&dir.path().join("k"), true).unwrap();
        let mut e = EngineV2::open(dir.path(), p, Box::new(ks), Ablation::default()).unwrap();
        e.unlock().unwrap();
        assert!(
            e.search(&unit(0, 8))
                .unwrap()
                .iter()
                .all(|h| h.memory.id != a),
            "deletion lost on reopen"
        );
        let hits = e.search(&unit(3, 8)).unwrap();
        assert_eq!(hits[0].memory.id, b);
        // The cursor purges every region within B flushes.
        for _ in 0..8 {
            e.flush().unwrap();
        }
        assert_eq!(
            e.tombstones_len(),
            0,
            "tombstone survived a full cursor sweep"
        );
    }

    #[test]
    fn staged_records_are_written_by_close() {
        let dir = tempfile::tempdir().unwrap();
        let p = params(8, 8, 8);
        let mut e = engine(dir.path(), p.clone(), Ablation::default());
        let a = e.put("only", "{}", &unit(1, 8)).unwrap();
        assert_eq!(e.staged_len(), 1);
        e.close().unwrap();
        let ks = SoftwareKeystore::open_at(&dir.path().join("k"), true).unwrap();
        let mut e = EngineV2::open(dir.path(), p, Box::new(ks), Ablation::default()).unwrap();
        e.unlock().unwrap();
        assert_eq!(e.search(&unit(1, 8)).unwrap()[0].memory.id, a);
    }

    #[test]
    fn a_full_partition_refuses_instead_of_spilling() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(4, 2, 4);
        p.index.probe_width = 1;
        p.index.probe_spread = 1;
        let mut e = engine(dir.path(), p, Ablation::default());
        e.put("a", "", &unit(0, 4)).unwrap();
        e.put("b", "", &unit(0, 4)).unwrap();
        assert!(matches!(
            e.put("c", "", &unit(0, 4)),
            Err(Error::CapacityExceeded { .. })
        ));
        assert_eq!(e.stats().refused, 1);
    }

    #[test]
    fn oversized_records_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = params(8, 8, 8);
        let mut e = engine(dir.path(), p, Ablation::default());
        let big = "x".repeat(300);
        assert!(e.put(&big, "", &unit(0, 8)).is_err());
        let fits = "x".repeat(e.geometry().max_record_bytes());
        e.put(&fits, "", &unit(0, 8)).unwrap();
        e.flush().unwrap();
        assert_eq!(
            e.search(&unit(0, 8)).unwrap()[0].memory.text.len(),
            fits.len()
        );
    }

    #[test]
    fn grouped_probes_read_exactly_one_group() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(8, 4, 8);
        p.v2.probe_mode = "grouped".into();
        p.v2.group_size = 4;
        let e = engine(dir.path(), p, Ablation::default());
        let t = e.table.as_ref().unwrap();
        for i in 0..8 {
            let set = e.probe_set(t, &unit(i, 8));
            assert_eq!(set.len(), 4);
            let g = t.groups[set[0] as usize];
            assert!(set.iter().all(|l| t.groups[*l as usize] == g));
        }
    }
}
