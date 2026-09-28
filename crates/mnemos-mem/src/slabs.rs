//! The slab file: `R` equal regions, preallocated and filled with random bytes.
//!
//! There is only one backend. Mnemos-P needs a growing-region backend to realise the S1 ablation;
//! here that ablation would be measuring the same thing twice, so the file is always fixed size.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use mnemos_core::error::{Error, Result};
use mnemos_v2::seal::random_into;

use crate::geometry::{MGeometry, FILE_HEADER_LEN};

const MAGIC: &[u8; 8] = b"MNEMOSMS";
const VERSION: u32 = 1;

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

/// `slab.bin`.
pub struct Slab {
    file: File,
    geo: MGeometry,
}

impl Slab {
    /// File name inside the store directory.
    pub const NAME: &'static str = "slab.bin";

    /// Create and fill with random bytes, one region at a time.
    ///
    /// # Errors
    /// I/O failures.
    pub fn create(dir: &Path, geo: MGeometry) -> Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(dir.join(Self::NAME))?;
        let mut header = vec![0u8; FILE_HEADER_LEN as usize];
        header[..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&geo.regions.to_le_bytes());
        header[16..20].copy_from_slice(&geo.capacity.to_le_bytes());
        header[20..28].copy_from_slice(&(geo.slot_len() as u64).to_le_bytes());
        file.write_all(&header)?;
        let mut region = Vec::with_capacity(geo.region_len());
        for _ in 0..geo.regions {
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
    pub fn open(dir: &Path, geo: MGeometry) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.join(Self::NAME))?;
        let mut header = vec![0u8; FILE_HEADER_LEN as usize];
        pread(&file, &mut header, 0)?;
        if &header[..8] != MAGIC {
            return Err(Error::Format("bad slab magic".into()));
        }
        // A header read from an untrusted file is parsed into errors, never into panics.
        let field = |r: std::ops::Range<usize>| -> Result<&[u8]> {
            header
                .get(r)
                .ok_or_else(|| Error::Format("truncated slab header".into()))
        };
        let r = u32::from_le_bytes(
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
        if r != geo.regions || c != geo.capacity || s != geo.slot_len() as u64 {
            return Err(Error::Format("slab geometry does not match params".into()));
        }
        Ok(Self { file, geo })
    }

    /// Read one whole region.
    ///
    /// # Errors
    /// I/O failures.
    pub fn read_region(&self, region: u32, out: &mut [u8]) -> Result<()> {
        pread(
            &self.file,
            &mut out[..self.geo.region_len()],
            self.geo.region_offset(region),
        )?;
        Ok(())
    }

    /// Write one whole region image. Returns bytes written.
    ///
    /// # Errors
    /// I/O failures.
    pub fn write_region(&mut self, region: u32, image: &[u8]) -> Result<u64> {
        debug_assert_eq!(image.len(), self.geo.region_len());
        pwrite(&self.file, image, self.geo.region_offset(region))?;
        Ok(image.len() as u64)
    }

    /// Read one slot. Used only by the `disk_search` ablation and the non-resident payload mode.
    ///
    /// # Errors
    /// I/O failures.
    pub fn read_slot(&self, region: u32, slot: u32, out: &mut [u8]) -> Result<()> {
        let off = self.geo.region_offset(region) + u64::from(slot) * self.geo.slot_len() as u64;
        pread(&self.file, &mut out[..self.geo.slot_len()], off)?;
        Ok(())
    }

    /// Read `n` consecutive slots starting at absolute slot `g`. Used by the sweep, whose read set
    /// is the cursor's range and therefore a function of the clock alone.
    ///
    /// # Errors
    /// I/O failures.
    pub fn read_slots(&self, g: u64, n: usize, out: &mut [u8]) -> Result<()> {
        let len = n * self.geo.slot_len();
        pread(&self.file, &mut out[..len], self.geo.slot_offset(g))?;
        Ok(())
    }

    /// Write consecutive slot images starting at absolute slot `g`. Returns bytes written.
    ///
    /// # Errors
    /// I/O failures.
    pub fn write_slots(&mut self, g: u64, image: &[u8]) -> Result<u64> {
        debug_assert_eq!(image.len() % self.geo.slot_len(), 0);
        pwrite(&self.file, image, self.geo.slot_offset(g))?;
        Ok(image.len() as u64)
    }

    /// Read the sealed header of absolute slot `g`, and nothing after it. Mnemos-M/disk's search
    /// reads this for each candidate in its probed partitions.
    ///
    /// # Errors
    /// I/O failures.
    pub fn read_header(&self, g: u64, out: &mut [u8]) -> Result<()> {
        pread(
            &self.file,
            &mut out[..self.geo.header_ct()],
            self.geo.slot_offset(g),
        )?;
        Ok(())
    }

    /// Flush file data to stable storage.
    ///
    /// # Errors
    /// I/O failures.
    pub fn sync(&mut self) -> Result<()> {
        self.file.sync_data()?;
        Ok(())
    }
}
