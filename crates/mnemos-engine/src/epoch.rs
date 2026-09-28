//! The epoch state machine — **the knob Theorem 1's Δ depends on.**
//!
//! Writes are buffered and flushed as a fixed-size, dummy-padded batch. Two triggers, whichever
//! fires first: `epoch_writes` staged records, or `epoch_ms` elapsed.
//!
//! ## Why the schedule is the knob
//!
//! Epoch membership is the *only* signal a post-compromise entry carries that a historical
//! entry does not. Its granularity is `epoch_writes`. So:
//!
//! - `epoch_writes = 1` — every write is its own flush. The adversary learns the exact epoch,
//!   hence near-exact time, of every write it induces. **Δ is maximal.**
//! - `epoch_writes = 64` — a write is one of 64 indistinguishable records. **Δ shrinks.**
//! - `epoch_writes → ∞` — one infinite batch; the cohorts become indistinguishable and Δ → 0.
//!
//! That limiting behaviour is what makes Theorem 1's "δ is a function of the epoch schedule
//! alone" both true and testable, and it is what E5/E6 sweep. **If the measured Δ does not
//! respond to this parameter, either the batching is broken or the leakage was never there** —
//! and both are findings worth having in September rather than in November.
//!
//! The time trigger exists so a quiet agent still makes progress. It is also a leakage channel
//! of its own: at `epoch_ms = 500` an adversary learns activity to within half a second. Both
//! triggers are swept; the tradeoff is reported rather than defaulted away.

use std::time::{Duration, Instant};

use mnemos_core::types::Epoch;
use mnemos_core::Params;

/// Tracks when the next flush is due.
pub struct EpochClock {
    epoch: Epoch,
    staged: u32,
    started: Instant,
    writes_trigger: u32,
    time_trigger: Duration,
}

impl EpochClock {
    /// Start at epoch 0.
    #[must_use]
    pub fn new(params: &Params) -> Self {
        Self {
            epoch: Epoch(0),
            staged: 0,
            started: Instant::now(),
            writes_trigger: params.epoch.epoch_writes,
            time_trigger: Duration::from_millis(params.epoch.epoch_ms),
        }
    }

    /// Record a staged write. Returns `true` when a flush is due.
    pub fn stage(&mut self) -> bool {
        self.staged += 1;
        self.due()
    }

    /// Whether a flush is due on either trigger.
    #[must_use]
    pub fn due(&self) -> bool {
        self.staged >= self.writes_trigger || self.started.elapsed() >= self.time_trigger
    }

    /// Advance after a completed flush.
    pub fn advance(&mut self) -> Epoch {
        self.epoch = self.epoch.next();
        self.staged = 0;
        self.started = Instant::now();
        self.epoch
    }

    /// Current epoch.
    #[must_use]
    pub fn current(&self) -> Epoch {
        self.epoch
    }

    /// Staged real records. **In-memory only.** If this number ever reaches disk — as a WAL
    /// length, a counter, or a log line — it becomes a live pending-write oracle for an
    /// adversary who is choosing the tasks that cause those writes.
    #[must_use]
    pub fn staged(&self) -> u32 {
        self.staged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_trigger_fires_at_the_configured_batch_size() {
        let params = Params::discover().unwrap();
        let mut clock = EpochClock::new(&params);
        for _ in 0..(params.epoch.epoch_writes - 1) {
            assert!(!clock.stage(), "must not flush before the batch is full");
        }
        assert!(clock.stage(), "must flush on the epoch_writes-th record");
    }

    #[test]
    fn advance_resets_the_batch() {
        let params = Params::discover().unwrap();
        let mut clock = EpochClock::new(&params);
        clock.stage();
        let e = clock.advance();
        assert_eq!(e, Epoch(1));
        assert_eq!(clock.staged(), 0);
    }
}
