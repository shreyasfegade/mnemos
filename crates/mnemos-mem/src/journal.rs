//! `journal.enc`: the fixed-size sealed holding area of the sweep flush.
//!
//! Under `flush_mode = "sweep"` a staged record is placed only when the cursor reaches a free slot,
//! which can be several flushes after the `put`. Until then it must survive a crash, and whatever
//! is written to make it survive must not reveal that it exists. So every flush rewrites this file
//! in full, sealed under a fresh nonce, whether it holds `journal_slots` records or none: its
//! length is a function of the parameters and its bytes are indistinguishable from random without
//! the key. This is the holding area of a deterministic write-only ORAM (Roche et al., CCS 2017),
//! with the difference that the position map lives in the locked arena rather than on disk.
//!
//! An entry carries the staged record's two ciphertexts exactly as `put` sealed them, plus its
//! tag. Nothing derived from content is added in the clear inside the plaintext either: the
//! partition id a record would need under `search = "ivf"` is recomputed from the header at
//! unlock, in the arena, rather than persisted here.

use std::path::Path;

use mnemos_core::error::{Error, Result};
use mnemos_v2::seal::{open_vec, seal_vec};

use crate::geometry::MGeometry;

/// File name inside the store directory.
pub const NAME: &str = "journal.enc";
const MAGIC: &[u8; 8] = b"MNEMOSMJ";

/// One journal entry, as `put` sealed it.
pub struct Entry {
    /// PRF tag of the record's identifier.
    pub tag: [u8; 16],
    /// Header sealed under the staging AAD.
    pub header_ct: Vec<u8>,
    /// Payload sealed under the tag.
    pub payload_ct: Vec<u8>,
}

fn entry_len(geo: &MGeometry) -> usize {
    1 + 16 + geo.header_ct() + geo.payload_ct()
}

/// Plaintext length. A function of the geometry alone.
#[must_use]
pub fn body_len(geo: &MGeometry) -> usize {
    8 + geo.journal_slots as usize * entry_len(geo)
}

/// Seal and write the journal atomically. Returns bytes written.
///
/// `entries` beyond `journal_slots` are not written; the caller keeps them staged in memory.
///
/// # Errors
/// I/O or crypto failures.
pub fn write(
    dir: &Path,
    key: &[u8; 32],
    geo: &MGeometry,
    epoch: u64,
    entries: &[&Entry],
    durable: bool,
) -> Result<u64> {
    let el = entry_len(geo);
    let mut body = Vec::with_capacity(body_len(geo));
    body.extend_from_slice(&epoch.to_le_bytes());
    for i in 0..geo.journal_slots as usize {
        let start = body.len();
        if let Some(e) = entries.get(i) {
            body.push(1);
            body.extend_from_slice(&e.tag);
            body.extend_from_slice(&e.header_ct);
            body.extend_from_slice(&e.payload_ct);
        }
        body.resize(start + el, 0);
    }
    debug_assert_eq!(body.len(), body_len(geo));
    let sealed = seal_vec(key, body, MAGIC)?;
    let mut out = Vec::with_capacity(8 + sealed.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&sealed);

    let tmp = dir.join("journal.tmp");
    {
        use std::io::Write as _;
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&out)?;
        if durable {
            f.sync_all()?;
        }
    }
    std::fs::rename(&tmp, dir.join(NAME))?;
    Ok(out.len() as u64)
}

/// Read the journal, if one exists. A store created under `flush_mode = "regions"` has none.
///
/// # Errors
/// I/O, crypto, or format failures.
pub fn read(dir: &Path, key: &[u8; 32], geo: &MGeometry) -> Result<Vec<Entry>> {
    let path = dir.join(NAME);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read(path)?;
    if raw.len() < 8 || &raw[..8] != MAGIC {
        return Err(Error::Format("bad journal magic".into()));
    }
    let body = open_vec(key, &raw[8..], MAGIC)?;
    if body.len() != body_len(geo) {
        return Err(Error::Format("journal size does not match params".into()));
    }
    let el = entry_len(geo);
    let (hct, pct) = (geo.header_ct(), geo.payload_ct());
    let mut out = Vec::new();
    for i in 0..geo.journal_slots as usize {
        let e = &body[8 + i * el..8 + (i + 1) * el];
        if e[0] != 1 {
            continue;
        }
        let mut tag = [0u8; 16];
        tag.copy_from_slice(&e[1..17]);
        out.push(Entry {
            tag,
            header_ct: e[17..17 + hct].to_vec(),
            payload_ct: e[17 + hct..17 + hct + pct].to_vec(),
        });
    }
    Ok(out)
}
