//! Write-ahead log — **always exactly `epoch_writes` records long on disk.**
//!
//! A conventional WAL grows as writes accumulate and truncates on flush. That would make its
//! length a live counter of pending writes, readable by anyone watching the disk. Since our
//! adversary can also *induce* writes, a pending-write counter is a near-perfect correlation
//! channel: feed a task, watch the counter tick, know the write landed.
//!
//! So the file is created at full size, holds exactly `epoch_writes` fixed-size records at all
//! times, and every unused record contains dummy material indistinguishable from a real one.
//! On flush it is overwritten wholesale rather than truncated.
//!
//! Cost: a full batch's worth of bytes every epoch even when one real entry was produced. At
//! `epoch_writes = 16` and 4 KiB records that is 64 KiB per epoch — negligible, and it removes
//! an entire leakage channel.
//!
//! # The staging buffer is in memory — and what that costs
//!
//! Real records accumulate in RAM and reach disk only at flush. [`Wal::pending`] is therefore
//! never observable. If that count ever reached disk — as a length, a counter, or a log line —
//! it would reinstate exactly the oracle this module exists to remove.
//!
//! **This means the module does not provide write-ahead durability, despite the name.** The
//! trade is deliberate and worth stating plainly rather than leaving to be discovered:
//!
//! | | Conventional WAL | This one |
//! |---|---|---|
//! | Crash before flush | Entries recoverable | **Entries lost** |
//! | Per-`put` disk write | Yes | No |
//! | Pending-write count on disk | Yes | No |
//!
//! A per-`put` disk write is a correlation channel for an adversary who *chose the task that
//! caused it*: feed the agent a prompt, watch a write land, learn the entry exists. That is the
//! post-compromise half of the threat model and the harder half to defend. Durability is
//! recoverable by other means — the caller can flush — whereas a correlation channel, once
//! open, cannot be closed after the fact.
//!
//! ## The loss window, exactly
//!
//! Up to `epoch_writes` entries, or `epoch_ms` of elapsed time, whichever trigger fires first.
//! At the shipped parameters that is **16 entries or 500 ms**.
//!
//! ## Consequences callers must handle
//!
//! - A process that exits before a flush loses everything staged. `mnemos-cli` flushes at the
//!   end of every `put` for exactly this reason: each invocation is its own process, so
//!   anything else silently discards the write.
//! - An embedded caller holding a long-lived `Engine` gets the batching for free and should
//!   **not** flush per write — that would reduce every batch to one real record and hand the
//!   adversary the per-write timing the batching exists to hide.

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use mnemos_core::error::Result;
use mnemos_core::traits::Record;
use mnemos_core::types::{BucketId, Epoch, Tag};
use mnemos_core::Params;
use mnemos_crypto::aead::dummy_record;
use mnemos_crypto::kdf::DerivedKey;
use mnemos_crypto::prf;

/// Fixed-length encrypted write-ahead log.
pub struct Wal {
    file: File,
    /// Records staged in memory. **Never written to disk as a count.**
    staged: Vec<Record>,
    /// Capacity, equal to `params.epoch.epoch_writes`.
    capacity: usize,
    /// Bytes per on-disk record.
    record_len: usize,
    /// Partition count, `B`. Needed so a dummy record can be given a *usable* partition id.
    ///
    /// Finding F5: dummies previously took an unbounded `u32` here, so with `B = 256` a dummy
    /// had a 256/2^32 chance of naming a partition that exists. `EncryptedIvf::insert`
    /// rejected the rest and the flush path discarded the error, so the index received writes
    /// for real records only -- handing an adversary both the real-record count per flush and
    /// their semantic partitions.
    buckets: u32,
}

impl Wal {
    /// Create at full size, entirely dummy-filled.
    ///
    /// # Errors
    /// I/O failures.
    pub fn create(path: &Path, params: &Params, record_len: usize) -> Result<Self> {
        let capacity = params.epoch.epoch_writes as usize;
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)?;

        let mut wal = Self {
            file,
            staged: Vec::new(),
            capacity,
            record_len,
            buckets: params.index.buckets,
        };
        wal.rewrite_full_dummy()?;
        Ok(wal)
    }

    /// Open an existing WAL.
    ///
    /// # Errors
    /// I/O failures.
    pub fn open(path: &Path, params: &Params, record_len: usize) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        Ok(Self {
            file,
            staged: Vec::new(),
            capacity: params.epoch.epoch_writes as usize,
            record_len,
            buckets: params.index.buckets,
        })
    }

    /// Stage a record. Returns `true` when the batch is full and a flush is due.
    ///
    /// # Errors
    /// I/O failures.
    pub fn append(&mut self, record: Record) -> Result<bool> {
        self.staged.push(record);
        Ok(self.is_full())
    }

    /// Produce exactly `capacity` records — the real ones plus dummies — PRF-shuffled.
    ///
    /// The caller must not filter or reorder this. **The padding is the mechanism**: a batch of
    /// 16 where only one is real is indistinguishable from a batch of 16 all real.
    ///
    /// # Errors
    /// I/O failures.
    ///
    /// # Panics
    /// If the PRF permutation is not a bijection over `0..len`, which its own test rules out.
    pub fn take_padded_batch(&mut self, k_epoch: &[u8; 32], _epoch: Epoch) -> Result<Vec<Record>> {
        let mut batch = std::mem::take(&mut self.staged);

        // Dummies carry random tags, random payloads and a uniform-random bucket, so they are
        // drawn from the same distribution as real records in every observable dimension.
        while batch.len() < self.capacity {
            // Tag and partition id are drawn from randomness independent of the payload. The
            // slab stores `tag || payload`, so a tag copied out of the payload would repeat its
            // first 16 bytes on disk and let a keyless reader pick out dummy slots, and a
            // partition id read from payload bytes would sit in plaintext beside them.
            let d = dummy_record(self.record_len);
            let side = dummy_record(20);
            let mut t = [0u8; 16];
            t.copy_from_slice(&side[..16]);
            // Reduced into range. A dummy whose partition does not exist is not a dummy --
            // it is an absence, and absences are countable. See finding F5.
            let bucket =
                u32::from_le_bytes(side[16..20].try_into().expect("4 bytes")) % self.buckets.max(1);
            batch.push(Record {
                tag: Tag(t),
                ciphertext: d,
                bucket: BucketId(bucket),
            });
        }
        batch.truncate(self.capacity);

        // Shuffle so position within the batch does not follow the order the agent produced
        // the entries. Seeded from the epoch key rather than a system RNG so a crash-replay
        // reproduces the same layout — a fresh shuffle would give two stores with identical
        // content different on-disk arrangements, breaking history independence.
        let perm = prf::batch_permutation(&DerivedKey(*k_epoch), batch.len());
        let mut shuffled: Vec<Option<Record>> = batch.into_iter().map(Some).collect();
        let mut out = Vec::with_capacity(self.capacity);
        for i in perm {
            out.push(
                shuffled[i]
                    .take()
                    .expect("permutation visits each index once"),
            );
        }
        Ok(out)
    }

    /// Overwrite the whole file with fresh dummies after a successful flush.
    ///
    /// Overwrite, never truncate: a shorter file is a smaller number.
    ///
    /// # Errors
    /// I/O failures.
    pub fn reset(&mut self) -> Result<()> {
        self.staged.clear();
        self.rewrite_full_dummy()
    }

    /// Real records staged. In-memory only, never observable.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.staged.len()
    }

    /// Whether the batch is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.staged.len() >= self.capacity
    }

    fn rewrite_full_dummy(&mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        for _ in 0..self.capacity {
            self.file.write_all(&dummy_record(self.record_len))?;
        }
        self.file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params::discover().expect("params.toml")
    }

    fn rec(i: u8) -> Record {
        Record {
            tag: Tag([i; 16]),
            ciphertext: vec![i; 128],
            bucket: BucketId(u32::from(i)),
        }
    }

    #[test]
    fn a_dummy_payload_does_not_repeat_its_tag() {
        // The slab writes `tag || payload`. A dummy whose tag was copied out of its own payload
        // would repeat 16 bytes on disk, and a keyless reader could pick out every dummy slot
        // and subtract the dummies' partitions from each flush.
        let dir = tempfile::tempdir().unwrap();
        let p = params();
        let mut wal = Wal::create(&dir.path().join("wal.enc"), &p, 128).unwrap();
        let batch = wal.take_padded_batch(&[2u8; 32], Epoch(0)).unwrap();
        assert_eq!(batch.len(), p.epoch.epoch_writes as usize);
        for r in &batch {
            assert_ne!(
                &r.ciphertext[..16],
                r.tag.as_bytes(),
                "dummy tag duplicates its payload"
            );
            assert!(r.bucket.0 < p.index.buckets, "dummy partition out of range");
        }
    }

    #[test]
    fn file_size_is_constant_across_the_whole_write_cycle() {
        // The regression test for the leak this module exists to close: stage one record,
        // several, a full batch, flush, reset — the on-disk length must never change.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.enc");
        let p = params();
        let mut wal = Wal::create(&path, &p, 128).unwrap();
        let baseline = std::fs::metadata(&path).unwrap().len();

        for i in 0..p.epoch.epoch_writes as u8 {
            wal.append(rec(i)).unwrap();
            assert_eq!(
                std::fs::metadata(&path).unwrap().len(),
                baseline,
                "staging must not touch the file — pending count would leak"
            );
        }

        let batch = wal.take_padded_batch(&[9u8; 32], Epoch(0)).unwrap();
        assert_eq!(batch.len(), p.epoch.epoch_writes as usize);

        wal.reset().unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), baseline);
    }

    #[test]
    fn batch_is_always_exactly_epoch_writes_long() {
        let dir = tempfile::tempdir().unwrap();
        let p = params();
        for real in [0usize, 1, 5] {
            let mut wal = Wal::create(&dir.path().join(format!("w{real}")), &p, 128).unwrap();
            for i in 0..real {
                wal.append(rec(i as u8)).unwrap();
            }
            let batch = wal.take_padded_batch(&[1u8; 32], Epoch(0)).unwrap();
            assert_eq!(
                batch.len(),
                p.epoch.epoch_writes as usize,
                "a batch with {real} real records must still be a full batch"
            );
        }
    }

    #[test]
    fn batch_order_does_not_follow_staging_order() {
        // If position within the batch tracked the order the agent produced entries, epoch
        // batching would hide the count but not the sequence.
        let dir = tempfile::tempdir().unwrap();
        let p = params();
        let mut wal = Wal::create(&dir.path().join("w"), &p, 128).unwrap();
        for i in 0..p.epoch.epoch_writes as u8 {
            wal.append(rec(i)).unwrap();
        }
        let batch = wal.take_padded_batch(&[4u8; 32], Epoch(0)).unwrap();
        let staged_order: Vec<u8> = (0..p.epoch.epoch_writes as u8).collect();
        let batch_order: Vec<u8> = batch.iter().map(|r| r.tag.as_bytes()[0]).collect();
        assert_ne!(batch_order, staged_order, "batch must be shuffled");
    }

    #[test]
    fn shuffle_is_reproducible_for_crash_replay() {
        let dir = tempfile::tempdir().unwrap();
        let p = params();
        let mk = |name: &str| {
            let mut w = Wal::create(&dir.path().join(name), &p, 128).unwrap();
            for i in 0..p.epoch.epoch_writes as u8 {
                w.append(rec(i)).unwrap();
            }
            w.take_padded_batch(&[8u8; 32], Epoch(3))
                .unwrap()
                .iter()
                .map(|r| r.tag.as_bytes()[0])
                .collect::<Vec<_>>()
        };
        assert_eq!(
            mk("a"),
            mk("b"),
            "same epoch key must replay the same arrangement"
        );
    }
}
