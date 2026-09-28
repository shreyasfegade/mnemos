//! The Mnemos-M engine.
//!
//! See the crate documentation for the five rules. This file is where each is enforced, and each
//! enforcement point says which rule it serves.

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
use mnemos_v2::geometry::{ID_LEN, LEN_PREFIX};

use crate::geometry::PART_ID_LEN;
use mnemos_v2::manifest::ManifestV2;
use mnemos_v2::seal::{decode_frame, open_in_place, random_into, seal_in_place, STAGED_AAD};
use mnemos_v2::table::{build_groups, Table};
use parking_lot::Mutex;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use zeroize::Zeroize;

use crate::geometry::MGeometry;
use crate::journal;
use crate::slabs::Slab;
use crate::Ablation;

/// HKDF labels. Distinct from v1's and v2's, so no M key equals any other design's key.
mod info {
    pub const HEADER: &[u8] = b"mnemos/m/header";
    pub const PAYLOAD: &[u8] = b"mnemos/m/payload";
    pub const MANIFEST: &[u8] = b"mnemos/m/manifest";
    pub const TABLE: &[u8] = b"mnemos/m/table";
    pub const TAG: &[u8] = b"mnemos/m/tag";
    pub const JOURNAL: &[u8] = b"mnemos/m/journal";
}

struct Keys {
    header: DerivedKey,
    payload: DerivedKey,
    manifest: DerivedKey,
    table: DerivedKey,
    tag: DerivedKey,
    journal: DerivedKey,
}

impl Keys {
    fn derive(master: &[u8; 32]) -> Result<Self> {
        Ok(Self {
            header: derive_key(master, info::HEADER)?,
            payload: derive_key(master, info::PAYLOAD)?,
            manifest: derive_key(master, info::MANIFEST)?,
            table: derive_key(master, info::TABLE)?,
            tag: derive_key(master, info::TAG)?,
            journal: derive_key(master, info::JOURNAL)?,
        })
    }
}

/// Associated data binding a header to its absolute slot.
fn header_aad(slot: u32) -> [u8; 14] {
    let mut a = [0u8; 14];
    a[..10].copy_from_slice(b"mnemos/m/h");
    a[10..].copy_from_slice(&slot.to_le_bytes());
    a
}

/// `Staged::slot` under the sweep, where a record's slot is not chosen until the cursor reaches a
/// free one.
const UNPLACED: u32 = u32::MAX;

/// A record accepted by `put` and not yet written. Holds ciphertext and, under region flushes, its
/// already-chosen slot.
struct Staged {
    tag: Tag,
    /// Absolute slot index, drawn uniformly at `put` time under region flushes (**Rule 1**), or
    /// [`UNPLACED`] under the sweep.
    slot: u32,
    /// Nearest-centroid partition, for the in-memory index under `m.search = "ivf"`; 0 otherwise.
    /// Computed from the header's own codes so put and unlock can never disagree.
    part: u16,
    header_ct: Vec<u8>,
    payload_ct: Vec<u8>,
}

/// Operational counters. In memory only.
#[derive(Clone, Copy, Debug, Default)]
pub struct MStats {
    /// Flushes completed since this engine was constructed.
    pub flushes: u64,
    /// Flushes that carried no real record. Under the clock these are the common case, and their
    /// existence is the point: a flush does not imply a write.
    pub empty_flushes: u64,
    /// Region rewrites performed.
    pub region_writes: u64,
    /// Bytes written to the slab.
    pub region_bytes_written: u64,
    /// Bytes written to the manifest.
    pub manifest_bytes_written: u64,
    /// `put` calls refused because the **whole store** was full.
    pub refused: u64,
    /// Tombstoned records physically removed.
    pub purged: u64,
    /// Regions read, which happens only at unlock.
    pub region_reads: u64,
    /// Bytes the resident index occupies.
    pub resident_bytes: u64,
    /// Slots rewritten by the sweep.
    pub slot_writes: u64,
    /// Bytes written to the sweep's journal.
    pub journal_bytes_written: u64,
    /// Slots read from disk after unlock: the sweep's cursor range when content is not resident,
    /// and Mnemos-M/disk's candidate headers and result payloads.
    pub slot_reads: u64,
}

/// A Mnemos-M store.
pub struct EngineM {
    dir: PathBuf,
    params: Params,
    geo: MGeometry,
    abl: Ablation,
    keystore: Box<dyn KeyStore>,
    slab: Slab,
    events: Arc<dyn EventSink>,
    flush_every: Duration,
    durable: bool,

    keys: Option<Keys>,
    /// The locked arena holds the resident index and the per-operation scratch, and nothing else
    /// in the process holds plaintext.
    arena: Mutex<Option<Arena>>,
    table: Option<Table>,

    occupancy: Vec<bool>,
    /// Number of `true` entries in `occupancy`, kept so the sweep's capacity check is O(1).
    occupied: usize,
    slot_tag: Vec<[u8; 16]>,
    where_tag: HashMap<[u8; 16], u32>,
    /// Free slots, in no particular order. A `put` draws uniformly from this.
    free: Vec<u32>,
    tombstones: Vec<[u8; 16]>,
    tomb_set: HashSet<[u8; 16]>,
    staged: Vec<Staged>,
    dirty: bool,

    epoch: u64,
    cursor: u32,
    last_flush: Instant,
    stats: MStats,
}

/// A search candidate: a placed slot (`Ok(g)`) or a staged record (`Err(i)`).
type Key = std::result::Result<u32, usize>;

/// Nearest-centroid partition of a header, read straight from its codes.
///
/// The ordering matches [`Table::ranked`]: a header's own scale is a positive common factor, so
/// the argmax of `codes . c.codes * c.scale` is the argmax of the cosine. Ties go to the lowest
/// index. Put and unlock both call this on the same bytes, so a record's partition is a function
/// of what is sealed on disk and never of when it was computed.
fn part_of(table: &Table, codes: &[u8]) -> u16 {
    let mut best = (f32::NEG_INFINITY, 0u16);
    for (i, c) in table.centroids.iter().enumerate() {
        let mut acc: i32 = 0;
        for (a, b) in codes.iter().zip(c.codes.iter()) {
            acc += i32::from(*a as i8) * i32::from(*b);
        }
        let v = acc as f32 * c.scale;
        if v > best.0 {
            best = (v, i as u16);
        }
    }
    best.1
}

/// Cosine between a query and a header's raw codes, without materialising a vector on the heap.
fn score_raw(q: &QuantVec, codes: &[u8], scale: f32) -> f32 {
    let mut acc: i32 = 0;
    for (a, b) in q.codes.iter().zip(codes.iter()) {
        acc += i32::from(*a) * i32::from(*b as i8);
    }
    acc as f32 * q.scale * scale
}

/// Ablations that have no meaning under the sweep are refused rather than silently ignored.
fn check_ablation(geo: &MGeometry, abl: Ablation) -> Result<()> {
    if geo.sweep && (abl.content_placement || abl.no_padding) {
        return Err(Error::Params(
            "m: the content_placement and no_padding ablations are defined for flush_mode =              \"regions\" only"
                .into(),
        ));
    }
    Ok(())
}

impl EngineM {
    /// Arena size: the resident index plus one slot's worth of scratch.
    fn arena_size(geo: &MGeometry) -> usize {
        geo.resident_bytes() + geo.header_pt + geo.payload_pt + 4096
    }

    /// Create a store. Starts **unlocked**.
    ///
    /// `centroids` is kept only so the content-placement ablation and the evaluation ground truth
    /// have a partition table to consult; the shipped engine never reads it.
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
        let geo = MGeometry::from_params(&params)?;
        check_ablation(&geo, abl)?;
        if centroids.len() != geo.regions as usize {
            return Err(Error::Format(format!(
                "expected {} centroids, got {}",
                geo.regions,
                centroids.len()
            )));
        }
        std::fs::create_dir_all(dir)?;

        let mut master = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut master);
        let wrapped = keystore.wrap_master_key(&master)?;
        let keys = Keys::derive(&master)?;
        master.zeroize();

        let groups = build_groups(&centroids, 1);
        let table = Table { centroids, groups };
        table.write(dir, &keys.table.0)?;

        let slab = Slab::create(dir, geo)?;
        let manifest = ManifestV2::new(geo.slots());
        manifest.write(
            dir,
            &keys.manifest.0,
            geo.tombstone_capacity,
            params.m.durable,
        )?;
        std::fs::write(dir.join("wrapped.key"), &wrapped)?;

        let arena = Arena::new(Self::arena_size(&geo), params.memory.lock_pages)?;
        let events: Arc<dyn EventSink> = Arc::new(NullSink);
        events.emit(LeakEvent::Setup {
            n_max: geo.slots() as u64,
            buckets: geo.regions,
            bucket_capacity: geo.capacity,
            block_size: params.crypto.block_size_bytes,
        });

        let mut stats = MStats {
            resident_bytes: geo.resident_bytes() as u64,
            ..MStats::default()
        };
        stats.resident_bytes = geo.resident_bytes() as u64;

        Ok(Self {
            dir: dir.to_path_buf(),
            flush_every: Duration::from_millis(geo.flush_ms),
            durable: params.m.durable,
            params,
            geo,
            abl,
            keystore,
            slab,
            events,
            keys: Some(keys),
            arena: Mutex::new(Some(arena)),
            table: Some(table),
            occupancy: vec![false; geo.slots()],
            occupied: 0,
            slot_tag: vec![[0u8; 16]; geo.slots()],
            where_tag: HashMap::new(),
            free: (0..geo.slots() as u32).collect(),
            tombstones: Vec::new(),
            tomb_set: HashSet::new(),
            staged: Vec::new(),
            dirty: false,
            epoch: 0,
            cursor: 0,
            last_flush: Instant::now(),
            stats,
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
        let geo = MGeometry::from_params(&params)?;
        check_ablation(&geo, abl)?;
        let slab = Slab::open(dir, geo)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            flush_every: Duration::from_millis(geo.flush_ms),
            durable: params.m.durable,
            params,
            geo,
            abl,
            keystore,
            slab,
            events: Arc::new(NullSink),
            keys: None,
            arena: Mutex::new(None),
            table: None,
            occupancy: vec![false; geo.slots()],
            occupied: 0,
            slot_tag: vec![[0u8; 16]; geo.slots()],
            where_tag: HashMap::new(),
            free: Vec::new(),
            tombstones: Vec::new(),
            tomb_set: HashSet::new(),
            staged: Vec::new(),
            dirty: false,
            epoch: 0,
            cursor: 0,
            last_flush: Instant::now(),
            stats: MStats {
                resident_bytes: geo.resident_bytes() as u64,
                ..MStats::default()
            },
        })
    }

    /// Replace the event sink.
    pub fn set_event_sink(&mut self, sink: Arc<dyn EventSink>) {
        self.events = sink;
    }

    /// Geometry in force.
    #[must_use]
    pub const fn geometry(&self) -> &MGeometry {
        &self.geo
    }

    /// Operational counters.
    #[must_use]
    pub const fn stats(&self) -> MStats {
        self.stats
    }

    /// Records staged and not yet written.
    #[must_use]
    pub fn staged_len(&self) -> usize {
        self.staged.len()
    }

    /// Deletions recorded and not yet purged.
    #[must_use]
    pub fn tombstones_len(&self) -> usize {
        self.tombstones.len()
    }

    /// **Evaluation ground truth only.** The region a *content-placed* design would have used.
    ///
    /// Mnemos-M never consults this. Experiments use it to ask what Mnemos-P would have leaked
    /// about the same record.
    ///
    /// # Errors
    /// [`Error::Locked`].
    pub fn content_region(&self, v: &QuantVec) -> Result<u32> {
        let t = self.table.as_ref().ok_or(Error::Locked)?;
        Ok(t.assign(v))
    }

    /// **Evaluation ground truth only.** The region a record actually occupies.
    ///
    /// # Errors
    /// [`Error::Locked`].
    pub fn actual_region(&self, id: MemoryId) -> Result<Option<u32>> {
        let tag = prf::tag_for(&self.keys()?.tag, id, 0);
        Ok(self.where_tag.get(&tag.0).map(|s| *s / self.geo.capacity))
    }

    fn keys(&self) -> Result<&Keys> {
        self.keys.as_ref().ok_or(Error::Locked)
    }

    fn cap(&self) -> usize {
        self.geo.capacity as usize
    }

    /// Flush if the clock says one is due. **Rule 3.**
    ///
    /// Under the shipped configuration a flush happens every Δ whether or not anything is staged,
    /// so the occurrence of a flush carries no information about whether the agent wrote. Under
    /// the `on_demand_flush` ablation a flush happens only when records are waiting, which is what
    /// restores the write-timing and derived-write channels.
    ///
    /// # Errors
    /// As [`SearchableMemory::flush`].
    pub fn tick(&mut self) -> Result<Option<Epoch>> {
        if self.keys.is_none() {
            return Ok(None);
        }
        let due = self.last_flush.elapsed() >= self.flush_every;
        if !due {
            return Ok(None);
        }
        if self.abl.on_demand_flush && self.staged.is_empty() && !self.dirty {
            return Ok(None);
        }
        self.flush().map(Some)
    }

    /// Flush anything pending, then lock.
    ///
    /// # Errors
    /// As [`SearchableMemory::flush`].
    pub fn close(&mut self) -> Result<()> {
        if self.keys.is_some() {
            if self.geo.sweep {
                self.lock_sweep()?;
                // `lock` would sweep again; the store has just done so.
                self.lock_now();
                return Ok(());
            } else if !self.staged.is_empty() || self.dirty {
                self.flush()?;
            }
        }
        self.lock();
        Ok(())
    }

    /// Drop the keys and everything decrypted, without flushing.
    fn lock_now(&mut self) {
        self.keys = None;
        self.table = None;
        // Dropping the arena zeroises the resident index, which is where every plaintext lives.
        *self.arena.lock() = None;
        self.where_tag.clear();
        self.free.clear();
        self.slot_tag.iter_mut().for_each(Zeroize::zeroize);
        self.staged.clear();
        self.occupied = 0;
        self.events.emit(LeakEvent::KeyOp { unlock: false });
    }

    /// The sweep's lock: one flush, always, so that locking is not itself evidence of a write,
    /// then more only while the backlog exceeds what the journal can hold.
    ///
    /// The second clause is the sweep's one residual: it runs only when the agent staged records
    /// faster than the cursor found free slots for them, and then the number of extra flushes at
    /// lock reveals that the backlog exceeded `journal_slots`. Its size is bounded by
    /// `journal_slots` and the placement rate `sweep_slots * (1 - load)` per flush, and E21
    /// measures how often a realistic write rate reaches it.
    fn lock_sweep(&mut self) -> Result<()> {
        self.flush()?;
        while self.staged.len() > self.geo.journal_slots as usize {
            self.flush()?;
        }
        Ok(())
    }

    /// Draw a free slot. **Rule 1.**
    ///
    /// Uniform over free slots, which is what makes a write's region independent of the record's
    /// content. Under the content-placement ablation the slot is instead drawn from the free slots
    /// of the record's nearest centroid's region, which is exactly what Mnemos-P does and exactly
    /// what opens the write channel.
    fn choose_slot(&mut self, embedding: &QuantVec) -> Result<u32> {
        if self.abl.content_placement {
            let region = {
                let t = self.table.as_ref().ok_or(Error::Locked)?;
                t.assign(embedding)
            };
            let base = region * self.geo.capacity;
            let pick = self
                .free
                .iter()
                .position(|s| *s >= base && *s < base + self.geo.capacity)
                .ok_or(Error::CapacityExceeded {
                    live: u64::from(self.geo.capacity),
                    n_max: u64::from(self.geo.capacity),
                })?;
            return Ok(self.free.swap_remove(pick));
        }
        if self.free.is_empty() {
            return Err(Error::CapacityExceeded {
                live: self.geo.slots() as u64,
                n_max: self.geo.slots() as u64,
            });
        }
        let i = rand::rngs::OsRng.gen_range(0..self.free.len());
        Ok(self.free.swap_remove(i))
    }

    /// Rewrite one whole region. **Rule 2.**
    ///
    /// Purges tombstoned slots, places `place` if it belongs here, re-seals every live slot under
    /// a fresh nonce, and refreshes every free slot with random bytes.
    fn rewrite(&mut self, region: u32, place: Option<&Staged>, epoch: u64) -> Result<()> {
        let geo = self.geo;
        let cap = self.cap();
        let base = region as usize * cap;
        let sl = geo.slot_len();
        let hct = geo.header_ct();
        let (hpt, ppt) = (geo.header_pt, geo.payload_pt);
        let res_hdr = geo.resident_header_bytes();
        let res_bytes = geo.resident_bytes();
        let (k_header, k_payload) = {
            let k = self.keys()?;
            (k.header.0, k.payload.0)
        };

        let mut current = vec![0u8; geo.region_len()];
        self.slab.read_region(region, &mut current)?;

        // Purge first, while `self` is freely mutable.
        for i in 0..cap {
            self.purge_if_tombstoned(base + i);
        }

        let target: Option<usize> = place
            .map(|s| s.slot as usize)
            .filter(|s| *s >= base && *s < base + cap);

        let mut image: Vec<u8> = Vec::with_capacity(geo.region_len());
        let mut guard = self.arena.lock();
        let arena = guard.as_mut().ok_or(Error::Locked)?;
        let result = (|| -> Result<()> {
            let buf = arena.buffer_mut();
            let (resident, scratch) = buf.split_at_mut(res_bytes);
            let (hdrs, pays) = resident.split_at_mut(res_hdr);
            let (hs, rest) = scratch.split_at_mut(hpt);
            let ps = &mut rest[..ppt];

            for i in 0..cap {
                let g = base + i;
                let aad = header_aad(g as u32);
                if Some(g) == target {
                    let st = place.expect("target implies a record");
                    // The staged header was sealed before its position existed; open it, record it
                    // in the resident index, and re-seal it bound to this slot.
                    open_in_place(&k_header, &st.header_ct, STAGED_AAD, hs)?;
                    hdrs[g * hpt..(g + 1) * hpt].copy_from_slice(hs);
                    if geo.index_ivf {
                        let o = geo.part_ids_offset() + g * PART_ID_LEN;
                        pays[o..o + PART_ID_LEN].copy_from_slice(&st.part.to_le_bytes());
                    }
                    seal_in_place(&k_header, hs, &aad, &mut image)?;
                    if geo.payload_resident {
                        open_in_place(&k_payload, &st.payload_ct, st.tag.as_bytes(), ps)?;
                        pays[g * ppt..(g + 1) * ppt].copy_from_slice(ps);
                        ps.zeroize();
                    }
                    // The payload ciphertext is already sealed under a fresh nonce and its AAD is
                    // position-independent, so it is written as it stands.
                    image.extend_from_slice(&st.payload_ct);
                } else if self.occupancy[g] {
                    // The header plaintext is already resident, so a rewrite never reads it back
                    // off the disk; it is copied out, re-sealed under a fresh nonce, and the
                    // resident copy is left as it was.
                    hs.copy_from_slice(&hdrs[g * hpt..(g + 1) * hpt]);
                    seal_in_place(&k_header, hs, &aad, &mut image)?;
                    let tag = self.slot_tag[g];
                    if geo.payload_resident {
                        ps.copy_from_slice(&pays[g * ppt..(g + 1) * ppt]);
                    } else {
                        open_in_place(&k_payload, &current[i * sl + hct..(i + 1) * sl], &tag, ps)?;
                    }
                    seal_in_place(&k_payload, ps, &tag, &mut image)?;
                } else {
                    random_into(&mut image, sl);
                }
            }
            Ok(())
        })();
        // Only the scratch is wiped; the resident index must survive the operation.
        {
            let buf = arena.buffer_mut();
            buf[res_bytes..].zeroize();
        }
        drop(guard);
        result?;

        if let (Some(g), Some(st)) = (target, place) {
            self.occupancy[g] = true;
            self.occupied += 1;
            self.slot_tag[g] = st.tag.0;
            self.where_tag.insert(st.tag.0, g as u32);
        }

        let written = self.slab.write_region(region, &image)?;
        self.events.emit(LeakEvent::RegionWrite {
            region,
            epoch: Epoch(epoch),
            slots: Vec::new(),
        });
        self.stats.region_writes += 1;
        self.stats.region_bytes_written += written;
        Ok(())
    }

    /// Free slot `g` if it holds a tombstoned record. Shared by both flush modes.
    fn purge_if_tombstoned(&mut self, g: usize) {
        if self.occupancy[g] && self.tomb_set.contains(&self.slot_tag[g]) {
            let t = self.slot_tag[g];
            self.occupancy[g] = false;
            self.occupied -= 1;
            self.where_tag.remove(&t);
            self.tomb_set.remove(&t);
            self.tombstones.retain(|x| *x != t);
            self.slot_tag[g] = [0u8; 16];
            self.free.push(g as u32);
            self.stats.purged += 1;
        }
    }

    /// One flush under `flush_mode = "sweep"`.
    ///
    /// Rewrites the `sweep_slots` slots after the cursor and nothing else in the slab, then the
    /// manifest and the journal, whose lengths are fixed. Which slots are written is a function of
    /// the cursor, which moves by `sweep_slots` per flush from a start drawn at unlock, so the
    /// whole write pattern is a function of the clock. Every slot written is a fresh sealing:
    /// a live record re-encrypted, a staged record placed in a free slot, or random bytes in a
    /// free slot, and without the key the three are the same distribution.
    fn sweep_flush(&mut self) -> Result<Epoch> {
        let geo = self.geo;
        let n = geo.slots();
        let k = geo.sweep_slots as usize;
        let e = self.epoch;
        if self.staged.is_empty() {
            self.stats.empty_flushes += 1;
        }
        let start = self.cursor as usize;
        let first = k.min(n - start);
        self.sweep_run(start, first, e)?;
        if k > first {
            self.sweep_run(0, k - first, e)?;
        }
        self.cursor = ((start + k) % n) as u32;

        self.epoch += 1;
        self.write_manifest()?;
        self.write_journal()?;
        if self.durable {
            self.slab.sync()?;
        }
        self.events.emit(LeakEvent::EpochFlush {
            epoch: Epoch(e),
            record_count: geo.sweep_slots,
        });
        self.stats.flushes += 1;
        self.dirty = false;
        self.last_flush = Instant::now();
        Ok(Epoch(self.epoch))
    }

    /// Rewrite the contiguous slots `s..s + len`.
    // One pass per slot kind -- placed, live, free -- inside one arena borrow; splitting it would
    // hand plaintext across a function boundary for no gain in clarity.
    #[allow(clippy::too_many_lines)]
    fn sweep_run(&mut self, s: usize, len: usize, epoch: u64) -> Result<()> {
        let geo = self.geo;
        let sl = geo.slot_len();
        let hct = geo.header_ct();
        let (hpt, ppt) = (geo.header_pt, geo.payload_pt);
        let res_hdr = geo.resident_header_bytes();
        let res_bytes = geo.resident_bytes();
        let (k_header, k_payload) = {
            let k = self.keys()?;
            (k.header.0, k.payload.0)
        };

        for g in s..s + len {
            self.purge_if_tombstoned(g);
        }
        // Placement, oldest staged record first, into whichever free slots the cursor reached.
        let mut places: Vec<Option<Staged>> = Vec::with_capacity(len);
        for g in s..s + len {
            if !self.occupancy[g] && !self.staged.is_empty() {
                places.push(Some(self.staged.remove(0)));
            } else {
                places.push(None);
            }
        }

        // Content that is not resident is read back from the cursor range itself. The read set is
        // the same range the write is about to cover, so it too is a function of the clock.
        let resident = geo.header_resident && geo.payload_resident;
        let mut current = Vec::new();
        if !resident {
            current = vec![0u8; len * sl];
            self.slab.read_slots(s as u64, len, &mut current)?;
            let cap = self.geo.capacity as usize;
            for g in s..s + len {
                self.events.emit(LeakEvent::RegionSlotRead {
                    region: (g / cap) as u32,
                    slot: (g % cap) as u32,
                });
            }
            self.stats.slot_reads += len as u64;
        }

        let mut image: Vec<u8> = Vec::with_capacity(len * sl);
        let mut guard = self.arena.lock();
        let arena = guard.as_mut().ok_or(Error::Locked)?;
        let result = (|| -> Result<()> {
            let buf = arena.buffer_mut();
            let (res, scratch) = buf.split_at_mut(res_bytes);
            let (hdrs, pays) = res.split_at_mut(res_hdr);
            let (hs, rest) = scratch.split_at_mut(hpt);
            let ps = &mut rest[..ppt];
            for (j, g) in (s..s + len).enumerate() {
                let aad = header_aad(g as u32);
                if let Some(st) = &places[j] {
                    open_in_place(&k_header, &st.header_ct, STAGED_AAD, hs)?;
                    if geo.header_resident {
                        hdrs[g * hpt..(g + 1) * hpt].copy_from_slice(hs);
                    }
                    if geo.index_ivf {
                        let o = geo.part_ids_offset() + g * PART_ID_LEN;
                        pays[o..o + PART_ID_LEN].copy_from_slice(&st.part.to_le_bytes());
                    }
                    seal_in_place(&k_header, hs, &aad, &mut image)?;
                    if geo.payload_resident {
                        open_in_place(&k_payload, &st.payload_ct, st.tag.as_bytes(), ps)?;
                        pays[g * ppt..(g + 1) * ppt].copy_from_slice(ps);
                        ps.zeroize();
                    }
                    image.extend_from_slice(&st.payload_ct);
                } else if self.occupancy[g] {
                    let tag = self.slot_tag[g];
                    if geo.header_resident {
                        hs.copy_from_slice(&hdrs[g * hpt..(g + 1) * hpt]);
                    } else {
                        open_in_place(&k_header, &current[j * sl..j * sl + hct], &aad, hs)?;
                    }
                    seal_in_place(&k_header, hs, &aad, &mut image)?;
                    if geo.payload_resident {
                        ps.copy_from_slice(&pays[g * ppt..(g + 1) * ppt]);
                    } else {
                        open_in_place(&k_payload, &current[j * sl + hct..(j + 1) * sl], &tag, ps)?;
                    }
                    seal_in_place(&k_payload, ps, &tag, &mut image)?;
                } else {
                    random_into(&mut image, sl);
                }
            }
            Ok(())
        })();
        {
            let buf = arena.buffer_mut();
            buf[res_bytes..].zeroize();
        }
        drop(guard);
        current.zeroize();
        result?;

        for (j, g) in (s..s + len).enumerate() {
            if let Some(st) = &places[j] {
                self.occupancy[g] = true;
                self.occupied += 1;
                self.slot_tag[g] = st.tag.0;
                self.where_tag.insert(st.tag.0, g as u32);
            }
        }

        let written = self.slab.write_slots(s as u64, &image)?;
        let cap = self.geo.capacity as usize;
        let mut g = s;
        while g < s + len {
            let region = g / cap;
            let end = ((region + 1) * cap).min(s + len);
            self.events.emit(LeakEvent::RegionWrite {
                region: region as u32,
                epoch: Epoch(epoch),
                slots: ((g % cap) as u32..=((end - 1) % cap) as u32).collect(),
            });
            g = end;
        }
        self.stats.slot_writes += len as u64;
        self.stats.region_bytes_written += written;
        Ok(())
    }

    /// Rewrite the journal with the oldest `journal_slots` staged records, or none.
    fn write_journal(&mut self) -> Result<()> {
        let key = self.keys()?.journal.0;
        let j = self.geo.journal_slots as usize;
        let entries: Vec<journal::Entry> = self
            .staged
            .iter()
            .take(j)
            .map(|st| journal::Entry {
                tag: st.tag.0,
                header_ct: st.header_ct.clone(),
                payload_ct: st.payload_ct.clone(),
            })
            .collect();
        let refs: Vec<&journal::Entry> = entries.iter().collect();
        let n = journal::write(&self.dir, &key, &self.geo, self.epoch, &refs, self.durable)?;
        self.stats.journal_bytes_written += n;
        self.events.emit(LeakEvent::JournalWrite {
            epoch: Epoch(self.epoch),
        });
        Ok(())
    }

    /// At unlock, return journaled records that were never placed to the staging queue.
    ///
    /// A record can be both placed and journaled if a flush was interrupted between the slab write
    /// and the journal write; its tag is then already resident and the journal copy is dropped.
    fn restore_journal(&mut self) -> Result<()> {
        let (key, k_header) = {
            let k = self.keys()?;
            (k.journal.0, k.header.0)
        };
        let geo = self.geo;
        for e in journal::read(&self.dir, &key, &geo)? {
            if self.where_tag.contains_key(&e.tag) {
                continue;
            }
            let mut part = 0u16;
            if geo.index_ivf {
                let table = self.table.as_ref().ok_or(Error::Locked)?;
                let res_bytes = geo.resident_bytes();
                let mut guard = self.arena.lock();
                let arena = guard.as_mut().ok_or(Error::Locked)?;
                let buf = arena.buffer_mut();
                let hs = &mut buf[res_bytes..res_bytes + geo.header_pt];
                let r = open_in_place(&k_header, &e.header_ct, STAGED_AAD, hs);
                if r.is_ok() {
                    part = part_of(table, &hs[ID_LEN..ID_LEN + geo.dim]);
                }
                hs.zeroize();
                r?;
            }
            self.staged.push(Staged {
                tag: Tag(e.tag),
                slot: UNPLACED,
                part,
                header_ct: e.header_ct,
                payload_ct: e.payload_ct,
            });
        }
        Ok(())
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

    /// The unlock scan: read the whole slab once and rebuild the resident index. **Rule 4.**
    ///
    /// This is the only time Mnemos-M reads the disk after creation, and it reads all of it, so
    /// what it reveals is that an unlock happened.
    /// Under `m.search = "ivf"`, which partitions a query probes; `None` for exact search.
    ///
    /// Computed against the resident table, in memory: no event and no disk access, so neither
    /// adversary class observes which partitions were probed.
    fn probe_mask(&self, query: &QuantVec) -> Result<Option<Vec<bool>>> {
        if !self.geo.index_ivf {
            return Ok(None);
        }
        let table = self.table.as_ref().ok_or(Error::Locked)?;
        let mut m = vec![false; self.geo.regions as usize];
        for (p, _) in table
            .ranked(query)
            .into_iter()
            .take(self.geo.probe as usize)
        {
            m[p as usize] = true;
        }
        Ok(Some(m))
    }

    fn rebuild_resident(&mut self) -> Result<()> {
        let geo = self.geo;
        let cap = self.cap();
        let sl = geo.slot_len();
        let hct = geo.header_ct();
        let (hpt, ppt) = (geo.header_pt, geo.payload_pt);
        let res_hdr = geo.resident_header_bytes();
        let res_bytes = geo.resident_bytes();
        let k_header = self.keys()?.header.0;
        let k_payload = self.keys()?.payload.0;
        let k_tag = self.keys()?.tag.clone();
        let table = if geo.index_ivf {
            Some(self.table.clone().ok_or(Error::Locked)?)
        } else {
            None
        };

        self.where_tag.clear();
        self.free.clear();
        let mut buf = vec![0u8; geo.region_len()];
        for region in 0..geo.regions {
            let base = region as usize * cap;
            self.events.emit(LeakEvent::RegionRead { region });
            self.stats.region_reads += 1;
            self.slab.read_region(region, &mut buf)?;

            let found: Vec<(usize, [u8; 16])> = {
                let mut guard = self.arena.lock();
                let arena = guard.as_mut().ok_or(Error::Locked)?;
                let r = (|| -> Result<Vec<(usize, [u8; 16])>> {
                    let b = arena.buffer_mut();
                    let (resident, scratch) = b.split_at_mut(res_bytes);
                    let (hdrs, pays) = resident.split_at_mut(res_hdr);
                    let (hs, rest) = scratch.split_at_mut(hpt);
                    let ps = &mut rest[..ppt];
                    let mut out = Vec::new();
                    for i in 0..cap {
                        let g = base + i;
                        if !self.occupancy[g] {
                            continue;
                        }
                        let aad = header_aad(g as u32);
                        open_in_place(&k_header, &buf[i * sl..i * sl + hct], &aad, hs)?;
                        if geo.header_resident {
                            hdrs[g * hpt..(g + 1) * hpt].copy_from_slice(hs);
                        }
                        if let Some(t) = &table {
                            let part = part_of(t, &hs[ID_LEN..ID_LEN + geo.dim]);
                            let o = geo.part_ids_offset() + g * PART_ID_LEN;
                            pays[o..o + PART_ID_LEN].copy_from_slice(&part.to_le_bytes());
                        }
                        let mut id = [0u8; 16];
                        id.copy_from_slice(&hs[..ID_LEN]);
                        let tag = prf::tag_for(&k_tag, MemoryId(id), 0);
                        if geo.payload_resident {
                            open_in_place(
                                &k_payload,
                                &buf[i * sl + hct..(i + 1) * sl],
                                &tag.0,
                                ps,
                            )?;
                            pays[g * ppt..(g + 1) * ppt].copy_from_slice(ps);
                        }
                        out.push((i, tag.0));
                    }
                    hs.zeroize();
                    ps.zeroize();
                    Ok(out)
                })();
                r?
            };
            for (i, tag) in found {
                self.slot_tag[base + i] = tag;
                self.where_tag.insert(tag, (base + i) as u32);
            }
        }
        self.occupied = 0;
        for g in 0..geo.slots() {
            if self.occupancy[g] {
                self.occupied += 1;
            } else {
                self.free.push(g as u32);
            }
        }
        Ok(())
    }
}

impl SearchableMemory for EngineM {
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
        self.keys()?;

        // Rule 1. The slot is drawn before the record's content has influenced anything on disk.
        // Under the sweep it is not drawn at all: the record takes the first free slot the cursor
        // reaches, and all `put` checks is the global bound, which counts staged records too.
        let slot = if geo.sweep {
            if self.occupied + self.staged.len() >= geo.slots() {
                self.stats.refused += 1;
                return Err(Error::CapacityExceeded {
                    live: geo.slots() as u64,
                    n_max: geo.slots() as u64,
                });
            }
            UNPLACED
        } else {
            match self.choose_slot(embedding) {
                Ok(s) => s,
                Err(e) => {
                    self.stats.refused += 1;
                    return Err(e);
                }
            }
        };

        let id = MemoryId::random();
        let (tag, part, header_ct, payload_ct) = {
            let keys = self.keys()?;
            // Only the ivf index needs the table; exact search must not depend on it.
            let table = if geo.index_ivf {
                Some(self.table.as_ref().ok_or(Error::Locked)?)
            } else {
                None
            };
            let mut part: u16 = 0;
            let tag = prf::tag_for(&keys.tag, id, 0);
            let mut guard = self.arena.lock();
            let arena = guard.as_mut().ok_or(Error::Locked)?;
            let res_bytes = geo.resident_bytes();
            let mut header_ct = Vec::with_capacity(geo.header_ct());
            let mut payload_ct = Vec::with_capacity(geo.payload_ct());
            let r = (|| -> Result<()> {
                let buf = arena.buffer_mut();
                let scratch = &mut buf[res_bytes..];
                let (hs, rest) = scratch.split_at_mut(geo.header_pt);
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
                if let Some(t) = table {
                    part = part_of(t, &hs[ID_LEN..ID_LEN + geo.dim]);
                }
                seal_in_place(&keys.header.0, hs, STAGED_AAD, &mut header_ct)?;
                Ok(())
            })();
            {
                let buf = arena.buffer_mut();
                buf[res_bytes..].zeroize();
            }
            r?;
            (tag, part, header_ct, payload_ct)
        };

        self.staged.push(Staged {
            tag,
            slot,
            part,
            header_ct,
            payload_ct,
        });

        // Rule 3: the clock decides when a flush happens, not this call. The ablation restores
        // the write-triggered behaviour so the channel it opens can be measured.
        if self.abl.on_demand_flush && self.staged.len() >= geo.epoch_writes as usize {
            self.flush()?;
        }
        Ok(id)
    }

    /// **Rule 4.** Ranks the resident index and touches no disk.
    ///
    /// Staged records are ranked too, from their staged ciphertexts in the arena scratch, so a
    /// record is searchable from the moment `put` returns whatever the flush mode. Under
    /// Mnemos-M/disk (`header_residency = "disk"`) the resident index is only a position map and
    /// each slot's partition, and the headers of the probed partitions' records are read by
    /// position: invisible to the W class, which sees no reads, and visible to RW by design.
    // Ranking and payload retrieval each hold the arena for exactly as long as plaintext exists;
    // the length is the price of not copying plaintext out of it.
    #[allow(clippy::too_many_lines)]
    fn search(&self, query: &QuantVec) -> Result<Vec<Hit>> {
        let geo = self.geo;
        let keys = self.keys()?;
        let sl = geo.slot_len();
        let hct = geo.header_ct();
        let (hpt, ppt) = (geo.header_pt, geo.payload_pt);
        let res_hdr = geo.resident_header_bytes();
        let res_bytes = geo.resident_bytes();

        let probe = self.probe_mask(query)?;

        let mut cands: Vec<(f32, Key, [u8; 16])> = Vec::new();
        {
            let mut guard = self.arena.lock();
            let arena = guard.as_mut().ok_or(Error::Locked)?;
            let r = (|| -> Result<()> {
                let buf = arena.buffer_mut();
                let (resident, scratch) = buf.split_at_mut(res_bytes);
                let (hdrs, rest) = resident.split_at(res_hdr);
                let ids = &rest[geo.part_ids_offset()..];
                let hs = &mut scratch[..hpt];
                let mut hdr_ct = vec![0u8; hct];
                for g in 0..geo.slots() {
                    if !self.occupancy[g] || self.tomb_set.contains(&self.slot_tag[g]) {
                        continue;
                    }
                    if let Some(m) = &probe {
                        let o = g * PART_ID_LEN;
                        let part = u16::from_le_bytes([ids[o], ids[o + 1]]);
                        if !m[part as usize] {
                            continue;
                        }
                    }
                    let h: &[u8] = if geo.header_resident {
                        &hdrs[g * hpt..(g + 1) * hpt]
                    } else {
                        let cap = geo.capacity;
                        self.events.emit(LeakEvent::RegionSlotRead {
                            region: g as u32 / cap,
                            slot: g as u32 % cap,
                        });
                        self.slab.read_header(g as u64, &mut hdr_ct)?;
                        open_in_place(&keys.header.0, &hdr_ct, &header_aad(g as u32), hs)?;
                        hs
                    };
                    let scale = f32::from_le_bytes(
                        h[ID_LEN + geo.dim..hpt]
                            .try_into()
                            .map_err(|_| Error::Crypto)?,
                    );
                    let score = score_raw(query, &h[ID_LEN..ID_LEN + geo.dim], scale);
                    let mut id = [0u8; 16];
                    id.copy_from_slice(&h[..ID_LEN]);
                    cands.push((score, Ok(g as u32), id));
                }
                for (i, st) in self.staged.iter().enumerate() {
                    if let Some(m) = &probe {
                        if !m[st.part as usize] {
                            continue;
                        }
                    }
                    open_in_place(&keys.header.0, &st.header_ct, STAGED_AAD, hs)?;
                    let scale = f32::from_le_bytes(
                        hs[ID_LEN + geo.dim..hpt]
                            .try_into()
                            .map_err(|_| Error::Crypto)?,
                    );
                    let score = score_raw(query, &hs[ID_LEN..ID_LEN + geo.dim], scale);
                    let mut id = [0u8; 16];
                    id.copy_from_slice(&hs[..ID_LEN]);
                    cands.push((score, Err(i), id));
                }
                hs.zeroize();
                Ok(())
            })();
            let buf = arena.buffer_mut();
            buf[res_bytes..].zeroize();
            r?;
        }
        // Top-k by selection, then order only those: the same result as a full sort under the
        // same total order (score descending, placed before staged, slot ascending).
        let rank = |k: &Key| -> (u8, usize) {
            match k {
                Ok(g) => (0, *g as usize),
                Err(i) => (1, *i),
            }
        };
        let order = |a: &(f32, Key, [u8; 16]), b: &(f32, Key, [u8; 16])| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(rank(&a.1).cmp(&rank(&b.1)))
        };
        if cands.len() > geo.top_k {
            cands.select_nth_unstable_by(geo.top_k - 1, order);
            cands.truncate(geo.top_k);
        }
        cands.sort_by(order);

        // Payloads. Resident by default, so still no disk access. The `disk_search` ablation
        // makes the read set depend on the results, which is the channel it exists to measure.
        let mut hits = Vec::with_capacity(cands.len());
        for (score, key, id) in cands {
            let mut disk: Option<Vec<u8>> = None;
            if let Ok(g) = key {
                if !geo.payload_resident || self.abl.disk_search {
                    let region = g / geo.capacity;
                    let slot = g % geo.capacity;
                    self.events.emit(LeakEvent::RegionSlotRead { region, slot });
                    let mut scratch = vec![0u8; sl];
                    self.slab.read_slot(region, slot, &mut scratch)?;
                    disk = Some(scratch);
                }
            }
            let mut guard = self.arena.lock();
            let arena = guard.as_mut().ok_or(Error::Locked)?;
            let text_meta = (|| -> Result<(String, String)> {
                let buf = arena.buffer_mut();
                let (resident, scratch) = buf.split_at_mut(res_bytes);
                let ps = &mut scratch[hpt..hpt + ppt];
                match key {
                    Err(i) => {
                        let st = &self.staged[i];
                        open_in_place(&keys.payload.0, &st.payload_ct, st.tag.as_bytes(), ps)?;
                    }
                    Ok(g) => {
                        if let Some(d) = &disk {
                            let tag = self.slot_tag[g as usize];
                            open_in_place(&keys.payload.0, &d[hct..sl], &tag, ps)?;
                        } else {
                            let pays = &resident[res_hdr..];
                            ps.copy_from_slice(&pays[g as usize * ppt..(g as usize + 1) * ppt]);
                        }
                    }
                }
                decode_frame(ps)
            })();
            {
                let buf = arena.buffer_mut();
                buf[res_bytes..].zeroize();
            }
            drop(guard);
            let (text, meta) = text_meta?;
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

    /// **Rule 5.** Records a tombstone and writes nothing.
    fn delete(&mut self, id: MemoryId) -> Result<()> {
        let tag = prf::tag_for(&self.keys()?.tag, id, 0);
        if let Some(pos) = self.staged.iter().position(|s| prf::tags_equal(s.tag, tag)) {
            let st = self.staged.remove(pos);
            if st.slot != UNPLACED {
                self.free.push(st.slot);
            }
            return Ok(());
        }
        if !self.where_tag.contains_key(&tag.0) {
            return Err(Error::NotFound);
        }
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
        Ok(())
    }

    /// **Rule 2.** Exactly `w + 1` whole-region rewrites, every one of them uniformly distributed.
    ///
    /// The real records' regions are uniform because their slots were drawn uniformly, and the
    /// padding regions are uniform because they are drawn uniformly. The two are therefore
    /// identically distributed, which is the property Mnemos-P cannot have: there, a real record's
    /// region is a function of its content and only the padding is uniform.
    fn flush(&mut self) -> Result<Epoch> {
        self.keys()?;
        if self.geo.sweep {
            return self.sweep_flush();
        }
        let geo = self.geo;
        let e = self.epoch;
        let w = geo.epoch_writes as usize;

        let take = self.staged.len().min(w);
        if take == 0 {
            self.stats.empty_flushes += 1;
        }
        let mut ops: Vec<Option<Staged>> = self.staged.drain(..take).map(Some).collect();
        ops.resize_with(w, || None);
        let mut rng = ChaCha20Rng::from_entropy();
        ops.shuffle(&mut rng);

        for op in ops {
            if let Some(st) = op {
                let region = st.slot / geo.capacity;
                self.rewrite(region, Some(&st), e)?;
            } else if !self.abl.no_padding {
                let region = rng.gen_range(0..geo.regions);
                self.rewrite(region, None, e)?;
            }
        }

        // The maintenance rewrite. Its region depends only on the flush count, and under the
        // clock it bounds a tombstoned record's life on disk to R*Delta milliseconds.
        let c = self.cursor;
        self.rewrite(c, None, e)?;
        self.cursor = (self.cursor + 1) % geo.regions;

        self.epoch += 1;
        self.write_manifest()?;
        if self.durable {
            self.slab.sync()?;
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
        let m = ManifestV2::read(
            &self.dir,
            &keys.manifest.0,
            self.geo.slots(),
            self.geo.tombstone_capacity,
        )?;
        self.epoch = m.epoch;
        // The cursor's start is drawn afresh, so its position cannot be read back as the number
        // of flushes the store performed before this session.
        self.cursor = if self.geo.sweep {
            rand::rngs::OsRng.gen_range(0..self.geo.slots() as u32)
        } else {
            rand::rngs::OsRng.gen_range(0..self.geo.regions)
        };
        self.occupancy = m.occupancy;
        self.tomb_set = m.tombstones.iter().copied().collect();
        self.tombstones = m.tombstones;
        self.table = Some(table);
        self.keys = Some(keys);
        *self.arena.lock() = Some(Arena::new(
            Self::arena_size(&self.geo),
            self.params.memory.lock_pages,
        )?);
        self.rebuild_resident()?;
        self.restore_journal()?;
        self.last_flush = Instant::now();
        self.events.emit(LeakEvent::KeyOp { unlock: true });
        Ok(())
    }

    fn lock(&mut self) {
        if self.keys.is_some() {
            if self.geo.sweep {
                let _ = self.lock_sweep();
            } else if !self.staged.is_empty() || self.dirty {
                let _ = self.flush();
            }
        }
        self.lock_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnemos_core::types::Embedding;
    use mnemos_keystore::software::SoftwareKeystore;

    fn params(r: u32, c: u32, dim: usize) -> Params {
        let mut p = Params::discover().expect("params.toml");
        p.index.buckets = r;
        p.index.bucket_capacity = c;
        p.index.top_k = 3;
        p.embedding.dim = dim;
        p.crypto.block_size_bytes = 256;
        p.epoch.epoch_writes = 4;
        p.memory.arena_bytes = 64 * 1024;
        p.m.durable = false;
        p.m.tombstone_capacity = 16;
        p.m.flush_ms = 3_600_000;
        p
    }

    fn unit(i: usize, dim: usize) -> QuantVec {
        let mut v = vec![0.02f32; dim];
        v[i % dim] = 1.0;
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        QuantVec::quantise(&Embedding(v.into_iter().map(|x| x / n).collect()))
    }

    fn engine(dir: &Path, p: Params, abl: Ablation) -> EngineM {
        let ks = SoftwareKeystore::open_at(&dir.join("k"), true).unwrap();
        let cents = (0..p.index.buckets as usize)
            .map(|i| unit(i, p.embedding.dim))
            .collect();
        EngineM::create(dir, p, Box::new(ks), cents, abl).unwrap()
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
        assert!(e
            .search(&unit(0, 8))
            .unwrap()
            .iter()
            .all(|h| h.memory.id != a));
        e.close().unwrap();

        let ks = SoftwareKeystore::open_at(&dir.path().join("k"), true).unwrap();
        let mut e = EngineM::open(dir.path(), p, Box::new(ks), Ablation::default()).unwrap();
        e.unlock().unwrap();
        assert!(e
            .search(&unit(0, 8))
            .unwrap()
            .iter()
            .all(|h| h.memory.id != a));
        assert_eq!(e.search(&unit(3, 8)).unwrap()[0].memory.id, b);
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
    fn ivf_search_finds_the_same_records_before_and_after_a_reopen() {
        // Partition ids are computed at put from the header's codes and recomputed at unlock
        // from the same sealed bytes. If the two ever disagreed a record would sit in a list its
        // own query never probes, and it would vanish from search after a reopen only.
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(8, 8, 8);
        p.m.search = "ivf".into();
        p.m.probe_width = 1;
        let mut e = engine(dir.path(), p.clone(), Ablation::default());
        let ids: Vec<MemoryId> = (0..8)
            .map(|i| e.put(&format!("m{i}"), "{}", &unit(i, 8)).unwrap())
            .collect();
        while e.staged_len() > 0 {
            e.flush().unwrap();
        }
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(
                e.search(&unit(i, 8)).unwrap()[0].memory.id,
                *id,
                "before reopen, m{i}"
            );
        }
        e.close().unwrap();

        let ks = SoftwareKeystore::open_at(&dir.path().join("k"), true).unwrap();
        let mut e = EngineM::open(dir.path(), p, Box::new(ks), Ablation::default()).unwrap();
        e.unlock().unwrap();
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(
                e.search(&unit(i, 8)).unwrap()[0].memory.id,
                *id,
                "after reopen, m{i}"
            );
        }
    }

    #[test]
    fn ivf_probes_only_its_partitions() {
        // With one partition probed, a query must not return a record from another partition
        // even when that is the only record in the store: that is what makes the search
        // sublinear, and it is the recall cost E11 measures.
        let dir = tempfile::tempdir().unwrap();
        let mut p = params(8, 8, 8);
        p.m.search = "ivf".into();
        p.m.probe_width = 1;
        let mut e = engine(dir.path(), p, Ablation::default());
        e.put("only", "{}", &unit(0, 8)).unwrap();
        while e.staged_len() > 0 {
            e.flush().unwrap();
        }
        assert_eq!(e.search(&unit(0, 8)).unwrap().len(), 1);
        assert!(
            e.search(&unit(5, 8)).unwrap().is_empty(),
            "probed a partition it should not"
        );
    }

    #[test]
    fn placement_is_independent_of_content() {
        // The same vector, written many times into a fresh store, must not concentrate in any
        // region. Under content placement it would land in exactly one.
        let dir = tempfile::tempdir().unwrap();
        let mut e = engine(dir.path(), params(16, 16, 8), Ablation::default());
        let mut seen = std::collections::HashSet::new();
        for i in 0..32 {
            let id = e.put(&format!("same {i}"), "{}", &unit(0, 8)).unwrap();
            e.flush().unwrap();
            seen.insert(e.actual_region(id).unwrap().unwrap());
        }
        assert!(
            seen.len() > 4,
            "32 identical vectors occupied only {} regions",
            seen.len()
        );
    }

    #[test]
    fn content_placement_ablation_concentrates() {
        let dir = tempfile::tempdir().unwrap();
        let abl = Ablation {
            content_placement: true,
            ..Ablation::default()
        };
        let mut e = engine(dir.path(), params(16, 16, 8), abl);
        let mut seen = std::collections::HashSet::new();
        for i in 0..8 {
            let id = e.put(&format!("same {i}"), "{}", &unit(0, 8)).unwrap();
            e.flush().unwrap();
            seen.insert(e.actual_region(id).unwrap().unwrap());
        }
        assert_eq!(
            seen.len(),
            1,
            "negative control: content placement must concentrate"
        );
    }

    #[test]
    fn a_flush_with_nothing_staged_still_writes() {
        let dir = tempfile::tempdir().unwrap();
        let mut e = engine(dir.path(), params(8, 8, 8), Ablation::default());
        let before = e.stats();
        e.flush().unwrap();
        let after = e.stats();
        assert_eq!(
            after.region_writes - before.region_writes,
            5,
            "w + 1 rewrites regardless"
        );
        assert_eq!(after.empty_flushes, 1);
    }

    #[test]
    fn capacity_is_global_not_per_region() {
        // Every record has the same vector, so a per-partition design would refuse after C.
        // Mnemos-M must accept until the whole store is full.
        let dir = tempfile::tempdir().unwrap();
        let p = params(8, 4, 8);
        let mut e = engine(dir.path(), p, Ablation::default());
        let mut ok = 0;
        for i in 0..32 {
            if e.put(&format!("m{i}"), "", &unit(0, 8)).is_ok() {
                ok += 1;
            }
        }
        assert_eq!(
            ok, 32,
            "global capacity is 8*4 = 32; only {ok} were admitted"
        );
        assert!(e.put("one too many", "", &unit(0, 8)).is_err());
        assert_eq!(e.stats().refused, 1);
    }
}
