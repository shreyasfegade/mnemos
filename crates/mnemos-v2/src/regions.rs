//! Region storage backends.
//!
//! [`FixedFile`] is the shipped layout: one preallocated file of `B` equal regions.
//! [`GrowingFiles`] exists only for the S1 ablation: one file per region holding only live
//! slots, so that file sizes reveal occupancy the way an unpadded store's would.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use mnemos_core::error::{Error, Result};

use crate::geometry::{Geometry, FILE_HEADER_LEN};
use crate::seal::random_into;

const MAGIC: &[u8; 8] = b"MNEMOS2P";
const VERSION: u32 = 1;

/// Storage for partition regions. The engine always works with full `C`-slot region images;
/// a backend decides what reaches disk.
pub trait RegionStore: Send + Sync {
    /// Read region `region` into `out` (`region_len` bytes). Free slots of a backend that does
    /// not store them come back as zeros.
    ///
    /// # Errors
    /// I/O failures.
    fn read_region(&self, region: u32, occupancy: &[bool], out: &mut [u8]) -> Result<()>;

    /// Write a full region image. Returns bytes written.
    ///
    /// # Errors
    /// I/O failures.
    fn write_region(&mut self, region: u32, image: &[u8], occupancy: &[bool]) -> Result<u64>;

    /// Write one slot. Returns bytes written.
    ///
    /// # Errors
    /// I/O failures, or [`Error::Format`] if the backend has no stable slot positions.
    fn write_slot(&mut self, region: u32, slot: u32, bytes: &[u8]) -> Result<u64>;

    /// Read one slot.
    ///
    /// # Errors
    /// I/O failures.
    fn read_slot(&self, region: u32, slot: u32, occupancy: &[bool], out: &mut [u8]) -> Result<()>;

    /// Flush file data to stable storage.
    ///
    /// # Errors
    /// I/O failures.
    fn sync(&mut self) -> Result<()>;
}

#[cfg(windows)]
fn pread(f: &File, mut buf: &mut [u8], mut off: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_read(buf, off)?;
        if n == 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        }
        buf = &mut buf[n..];
        off += n as u64;
    }
    Ok(())
}

#[cfg(unix)]
fn pread(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, off)
}

#[cfg(windows)]
fn pwrite(f: &File, mut buf: &[u8], mut off: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_write(buf, off)?;
        if n == 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::WriteZero));
        }
        buf = &buf[n..];
        off += n as u64;
    }
    Ok(())
}

#[cfg(unix)]
fn pwrite(f: &File, buf: &[u8], off: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.write_all_at(buf, off)
}

/// The shipped layout: `partitions.bin`, preallocated and filled with random bytes.
pub struct FixedFile {
    file: File,
    geo: Geometry,
}

impl FixedFile {
    /// File name inside the store directory.
    pub const NAME: &'static str = "partitions.bin";

    /// Create and fill with random bytes, one region at a time.
    ///
    /// # Errors
    /// I/O failures.
    pub fn create(dir: &Path, geo: Geometry) -> Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(dir.join(Self::NAME))?;
        let mut header = vec![0u8; FILE_HEADER_LEN as usize];
        header[..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&geo.buckets.to_le_bytes());
        header[16..20].copy_from_slice(&geo.capacity.to_le_bytes());
        header[20..28].copy_from_slice(&(geo.slot_len() as u64).to_le_bytes());
        file.write_all(&header)?;
        let mut region = Vec::with_capacity(geo.region_len());
        for _ in 0..geo.buckets {
            region.clear();
            random_into(&mut region, geo.region_len());
            file.write_all(&region)?;
        }
        file.sync_all()?;
        Ok(Self { file, geo })
    }

    /// Open and check the header against the geometry.
    ///
    /// # Errors
    /// I/O failures or [`Error::Format`] on a mismatch.
    pub fn open(dir: &Path, geo: Geometry) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.join(Self::NAME))?;
        let mut header = vec![0u8; FILE_HEADER_LEN as usize];
        pread(&file, &mut header, 0)?;
        if &header[..8] != MAGIC {
            return Err(Error::Format("bad partition file magic".into()));
        }
        // A header read from an untrusted file is parsed into errors, never into panics.
        let field = |r: std::ops::Range<usize>| -> Result<&[u8]> {
            header
                .get(r)
                .ok_or_else(|| Error::Format("truncated partition file header".into()))
        };
        let b = u32::from_le_bytes(
            field(12..16)?
                .try_into()
                .map_err(|_| Error::Format("header".into()))?,
        );
        let c = u32::from_le_bytes(
            field(16..20)?
                .try_into()
                .map_err(|_| Error::Format("header".into()))?,
        );
        let s = u64::from_le_bytes(
            field(20..28)?
                .try_into()
                .map_err(|_| Error::Format("header".into()))?,
        );
        if b != geo.buckets || c != geo.capacity || s != geo.slot_len() as u64 {
            return Err(Error::Format(
                "partition file geometry does not match params".into(),
            ));
        }
        Ok(Self { file, geo })
    }
}

impl RegionStore for FixedFile {
    fn read_region(&self, region: u32, _occupancy: &[bool], out: &mut [u8]) -> Result<()> {
        pread(
            &self.file,
            &mut out[..self.geo.region_len()],
            self.geo.region_offset(region),
        )?;
        Ok(())
    }

    fn write_region(&mut self, region: u32, image: &[u8], _occupancy: &[bool]) -> Result<u64> {
        debug_assert_eq!(image.len(), self.geo.region_len());
        pwrite(&self.file, image, self.geo.region_offset(region))?;
        Ok(image.len() as u64)
    }

    fn write_slot(&mut self, region: u32, slot: u32, bytes: &[u8]) -> Result<u64> {
        let off = self.geo.region_offset(region) + u64::from(slot) * self.geo.slot_len() as u64;
        pwrite(&self.file, bytes, off)?;
        Ok(bytes.len() as u64)
    }

    fn read_slot(&self, region: u32, slot: u32, _occupancy: &[bool], out: &mut [u8]) -> Result<()> {
        let off = self.geo.region_offset(region) + u64::from(slot) * self.geo.slot_len() as u64;
        pread(&self.file, &mut out[..self.geo.slot_len()], off)?;
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.file.sync_data()?;
        Ok(())
    }
}

/// S1 ablation: `regions/rNNNNN.bin`, each holding only its live slots in slot order.
pub struct GrowingFiles {
    dir: PathBuf,
    geo: Geometry,
}

impl GrowingFiles {
    /// Directory name inside the store directory.
    pub const DIR: &'static str = "regions";

    fn path(&self, region: u32) -> PathBuf {
        self.dir.join(format!("r{region:05}.bin"))
    }

    /// Create empty region files.
    ///
    /// # Errors
    /// I/O failures.
    pub fn create(dir: &Path, geo: Geometry) -> Result<Self> {
        let d = dir.join(Self::DIR);
        std::fs::create_dir_all(&d)?;
        let s = Self { dir: d, geo };
        for r in 0..geo.buckets {
            std::fs::write(s.path(r), [])?;
        }
        Ok(s)
    }

    /// Open.
    ///
    /// # Errors
    /// I/O failures.
    pub fn open(dir: &Path, geo: Geometry) -> Result<Self> {
        Ok(Self {
            dir: dir.join(Self::DIR),
            geo,
        })
    }
}

impl RegionStore for GrowingFiles {
    fn read_region(&self, region: u32, occupancy: &[bool], out: &mut [u8]) -> Result<()> {
        let raw = std::fs::read(self.path(region))?;
        let sl = self.geo.slot_len();
        out[..self.geo.region_len()].fill(0);
        let mut k = 0usize;
        for (i, occ) in occupancy.iter().enumerate() {
            if *occ {
                let src = raw.get(k * sl..(k + 1) * sl).ok_or_else(|| {
                    Error::Format("growing region shorter than its occupancy".into())
                })?;
                out[i * sl..(i + 1) * sl].copy_from_slice(src);
                k += 1;
            }
        }
        Ok(())
    }

    fn write_region(&mut self, region: u32, image: &[u8], occupancy: &[bool]) -> Result<u64> {
        let sl = self.geo.slot_len();
        let mut buf = Vec::new();
        for (i, occ) in occupancy.iter().enumerate() {
            if *occ {
                buf.extend_from_slice(&image[i * sl..(i + 1) * sl]);
            }
        }
        std::fs::write(self.path(region), &buf)?;
        Ok(buf.len() as u64)
    }

    fn write_slot(&mut self, _region: u32, _slot: u32, _bytes: &[u8]) -> Result<u64> {
        Err(Error::Format(
            "growing layout has no stable slot positions".into(),
        ))
    }

    fn read_slot(&self, region: u32, slot: u32, occupancy: &[bool], out: &mut [u8]) -> Result<()> {
        let mut full = vec![0u8; self.geo.region_len()];
        self.read_region(region, occupancy, &mut full)?;
        let sl = self.geo.slot_len();
        out[..sl].copy_from_slice(&full[slot as usize * sl..(slot as usize + 1) * sl]);
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        Ok(())
    }
}
