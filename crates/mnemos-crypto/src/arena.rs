//! The locked plaintext arena.
//!
//! **This module is the mechanism behind threat-model assumption TA-4**, and it is the reason
//! the core is Rust rather than Python (ADR-0001):
//!
//! > plaintext resides only in a locked, promptly-zeroised arena of ≤ 2 MiB, never on disk,
//! > swap, or in logs.
//!
//! Decrypted memory contents, decrypted query vectors, and the master key all live here and
//! nowhere else inside the TCB. Two properties are enforced:
//!
//! 1. **Locked** — `VirtualLock` on Windows, `mlock` on Unix, so pages never reach the
//!    pagefile. A plaintext memory in `pagefile.sys` survives reboots and is trivially
//!    recovered by exactly the adversary we are defending against.
//! 2. **Zeroised** — on drop and on [`Arena::reset`], via `zeroize`, which the compiler is not
//!    permitted to elide.
//!
//! ## Why a bump arena rather than the heap
//!
//! A general allocator reuses freed blocks without clearing them and may move data on realloc,
//! leaving copies behind. A fixed-size bump arena we allocate once, lock once, and clear
//! wholesale gives an exact answer to "where is the plaintext right now" — which is what the
//! threat model demands and what E1 verifies by reading the backing pages after `lock()`.

use zeroize::Zeroize;

use mnemos_core::error::{Error, Result};

/// A fixed-size, page-locked, zeroise-on-drop region.
///
/// Allocated once at unlock and released at lock. Never grows: exceeding capacity is an error
/// rather than a reallocation, because a reallocation would leave the old contents somewhere
/// we no longer control.
pub struct Arena {
    buf: Vec<u8>,
    cursor: usize,
    locked: bool,
}

impl Arena {
    /// Allocate and lock `capacity` bytes.
    ///
    /// # Errors
    /// Returns [`Error::Crypto`] if the pages cannot be locked and `lock_pages` is required.
    /// **This is deliberately fatal**: silently continuing with unlocked pages would leave the
    /// system apparently working while TA-4 is false, which is the failure mode this project
    /// least wants.
    pub fn new(capacity: usize, lock_pages: bool) -> Result<Self> {
        let buf = vec![0u8; capacity];
        let locked = if lock_pages {
            lock_memory(&buf)?;
            true
        } else {
            // Only reachable from tests; Params::validate rejects lock_pages = false.
            false
        };
        Ok(Self {
            buf,
            cursor: 0,
            locked,
        })
    }

    /// Bytes still available.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.cursor
    }

    /// Carve out `len` zeroed bytes.
    ///
    /// # Errors
    /// [`Error::Crypto`] if the arena is exhausted. Callers must size the arena for the worst
    /// case — `(k_probe + d) × C` vectors plus `top_k` decrypted entries — rather than growing.
    pub fn alloc(&mut self, len: usize) -> Result<&mut [u8]> {
        if len > self.remaining() {
            return Err(Error::Crypto);
        }
        let start = self.cursor;
        self.cursor += len;
        Ok(&mut self.buf[start..start + len])
    }

    /// Zeroise everything and rewind. Called at the end of every operation, not just at lock —
    /// plaintext should not outlive the call that produced it.
    pub fn reset(&mut self) {
        self.buf[..self.cursor].zeroize();
        self.cursor = 0;
    }

    /// The whole locked buffer, for callers that manage their own offsets within it.
    ///
    /// Callers must call [`Arena::wipe`] before returning from the operation that wrote
    /// plaintext into it.
    pub fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    /// Zeroise the whole buffer regardless of the bump cursor.
    pub fn wipe(&mut self) {
        self.buf.as_mut_slice().zeroize();
        self.cursor = 0;
    }

    /// Total capacity in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Whether the pages are locked. Recorded in experiment metadata so no benchmark can
    /// silently have been run without the guarantee in place.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.locked
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        self.buf.as_mut_slice().zeroize();
        if self.locked {
            // Best-effort: the process is tearing down this region regardless, and the pages
            // have already been zeroised above.
            unlock_memory(&self.buf);
        }
    }
}

#[cfg(windows)]
fn lock_memory(buf: &[u8]) -> Result<()> {
    use windows_sys::Win32::System::Memory::VirtualLock;
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, GetProcessWorkingSetSize, SetProcessWorkingSetSize,
    };

    if buf.is_empty() {
        return Ok(());
    }

    // VirtualLock charges against the process working-set *minimum*, and Windows' default
    // minimum is well under our 2 MiB arena — so the first attempt commonly fails with
    // ERROR_WORKING_SET_QUOTA on an otherwise healthy machine. Raise the quota and retry
    // rather than reporting a spurious failure.
    //
    // SAFETY: all three calls take a pseudo-handle from GetCurrentProcess (never invalid,
    // never needs closing) and, for the getter, two out-pointers to stack locals we own.
    unsafe {
        let ok = VirtualLock(buf.as_ptr().cast(), buf.len());
        if ok != 0 {
            return Ok(());
        }

        let proc = GetCurrentProcess();
        let mut min: usize = 0;
        let mut max: usize = 0;
        if GetProcessWorkingSetSize(proc, &raw mut min, &raw mut max) == 0 {
            return Err(Error::Crypto);
        }

        // ADD headroom rather than taking a max. `min.max(len)` looks equivalent and is not:
        // once the current minimum already exceeds the arena size — which happens as soon as a
        // second region is locked, or after any earlier raise — the max is a no-op, we re-set
        // the identical value, and the retry fails for exactly the reason it was meant to fix.
        // Every lock must enlarge the quota by its own size plus slack.
        let want_min = min.saturating_add(buf.len().saturating_mul(2)).max(1 << 20);
        let want_max = max.max(want_min.saturating_mul(2));
        if SetProcessWorkingSetSize(proc, want_min, want_max) == 0 {
            return Err(Error::Crypto);
        }

        if VirtualLock(buf.as_ptr().cast(), buf.len()) == 0 {
            return Err(Error::Crypto);
        }
    }
    Ok(())
}

#[cfg(unix)]
fn lock_memory(buf: &[u8]) -> Result<()> {
    if buf.is_empty() {
        return Ok(());
    }
    // SAFETY: pointer and length describe a live allocation we own for the call's duration.
    let rc = unsafe { libc::mlock(buf.as_ptr().cast(), buf.len()) };
    if rc != 0 {
        // Almost always RLIMIT_MEMLOCK being too low. The caller turns this into a hard
        // failure rather than continuing unlocked, because continuing would make TA-4 false
        // while leaving the system apparently working.
        return Err(Error::Crypto);
    }
    Ok(())
}

#[cfg(windows)]
fn unlock_memory(buf: &[u8]) {
    use windows_sys::Win32::System::Memory::VirtualUnlock;

    if buf.is_empty() {
        return;
    }
    // SAFETY: same region previously passed to VirtualLock, still owned by us.
    // Best-effort: the pages are already zeroised and the process owns them either way.
    unsafe { VirtualUnlock(buf.as_ptr().cast(), buf.len()) };
}

#[cfg(unix)]
fn unlock_memory(buf: &[u8]) {
    if buf.is_empty() {
        return;
    }
    // SAFETY: same region previously passed to mlock, still owned by us.
    unsafe { libc::munlock(buf.as_ptr().cast(), buf.len()) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_clears_issued_bytes() {
        let mut a = Arena::new(4096, false).unwrap();
        {
            let slice = a.alloc(32).unwrap();
            slice.copy_from_slice(&[0xAB; 32]);
        }
        a.reset();
        assert!(
            a.buf[..32].iter().all(|&b| b == 0),
            "reset must zeroise issued bytes"
        );
        assert_eq!(a.remaining(), 4096);
    }

    #[test]
    fn exhaustion_is_an_error_not_a_realloc() {
        let mut a = Arena::new(64, false).unwrap();
        assert!(
            a.alloc(65).is_err(),
            "arena must never grow — see module docs"
        );
    }

    #[test]
    fn real_page_locking_works_at_the_configured_size() {
        // TA-4's mechanism, exercised for real at the size production uses. If this fails on a
        // developer machine it is a genuine environment problem (Windows working-set quota,
        // or RLIMIT_MEMLOCK on Unix) and must be fixed rather than worked around by passing
        // lock_pages = false — that flag exists for tests, not for making a red test green.
        let arena = Arena::new(2 * 1024 * 1024, true)
            .expect("2 MiB must be lockable; see module docs if this fails");
        assert!(arena.is_locked());
    }

    #[test]
    fn drop_zeroises_before_release() {
        // Sanity: the arena clears its own storage on drop. We cannot observe freed memory
        // safely, so this checks the reachable precondition — that reset() clears everything
        // issued, which Drop also invokes over the whole buffer.
        let mut a = Arena::new(256, false).unwrap();
        a.alloc(128).unwrap().fill(0xCD);
        a.reset();
        assert!(a.buf.iter().all(|&b| b == 0));
    }
}
