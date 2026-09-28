"""Confidence intervals and honest baselines for the attack results.

Why this module exists
======================
Every advantage figure in this project was, until now, a **point estimate**: the mean over
eight seeds, reported bare. A bare mean answers "what happened when we ran it" and not "what
would happen if we ran it again", and the second question is the one a reviewer asks. Three
specific gaps are closed here.

**1. Confidence intervals, via a cluster bootstrap.**
There are two independent sources of variation in an A4 measurement:

* *attack randomness* -- which partitions the injector happens to localise, which varies with
  the trial seed;
* *corpus sampling* -- which entries happen to be in the corpus at all.

Resampling only seeds would understate the interval (it treats this exact corpus as the whole
population); resampling only entries would understate it differently (it treats one attack run
as definitive). :func:`bootstrap_pair` resamples **both**, seeds at the outer level and entries
at the inner level -- a two-stage or *cluster* bootstrap. Within one bootstrap iteration the
same resampled seed set is used for both cohorts, which preserves the pairing that makes the
ratio meaningful.

**2. The ratio is bootstrapped directly, not divided after the fact.**
``Adv_hist / Adv_post`` is a ratio of two random quantities. The ratio of two means is a biased
estimator of the mean ratio, and dividing the endpoints of two separate intervals is not a
confidence interval for anything at all. So the ratio is recomputed **inside** each bootstrap
iteration and the percentiles are taken over the resulting distribution.

**3. A most-frequent-class baseline alongside uniform chance.**
Advantage has always been measured against ``1/n_topics`` -- the accuracy of a uniform random
guess. That is the right baseline only if topics are equally likely. Real corpora are skewed, so
an adversary who ignores the store entirely and always answers with the modal topic already
beats ``1/n_topics``. Reporting only the uniform-chance advantage would credit the attack for
exploiting skew it never looked at. :func:`mfc_accuracy` gives that trivial adversary's score,
and every result below carries both numbers.

What these intervals do and do not cover
========================================
They quantify sampling variability of *this* estimator on *this* corpus under *this* attack.
They say nothing about whether the corpus is representative of real agent memory, or whether the
substitute embedding backend changes the partitioning. Those are separate limitations, listed as
such in the paper rather than buried inside an interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Bootstrap resamples. 10k gives percentile endpoints stable to about 0.001, finer than any
#: advantage difference the paper claims.
N_BOOT = 10_000

#: Two-sided 95% interval.
ALPHA = 0.05


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile confidence interval."""

    point: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely above zero -- i.e. the effect is resolved."""
        return self.lo > 0.0

    def as_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi}


def mfc_accuracy(labels: np.ndarray) -> float:
    """Accuracy of the trivial "always answer with the modal class" adversary."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return 0.0
    return float(np.bincount(labels).max()) / float(labels.size)


def _weighted_accuracy(hits: np.ndarray, seed_w: np.ndarray, entry_w: np.ndarray) -> float:
    """Mean of ``hits`` under multinomial resampling weights on both axes.

    ``hits`` is ``(n_seeds, n_entries)``. Resampling with replacement is equivalent to weighting
    by multinomial counts, and expressing it that way turns the inner loop into two
    matrix-vector products instead of fancy-indexing a large array -- which is what makes 10,000
    resamples take seconds rather than minutes.
    """
    per_entry = seed_w @ hits / seed_w.sum()
    return float(per_entry @ entry_w / entry_w.sum())


def _weighted_mfc(labels: np.ndarray, entry_w: np.ndarray, n_classes: int) -> float:
    """Most-frequent-class accuracy under the same entry resampling.

    Recomputed per resample rather than held fixed: the modal class's share is itself a
    statistic of the sample, so freezing it would make the baseline artificially precise and the
    advantage-over-baseline interval artificially narrow.
    """
    totals = np.bincount(labels, weights=entry_w, minlength=n_classes)
    return float(totals.max() / entry_w.sum())


@dataclass(frozen=True)
class PairResult:
    """Bootstrapped advantage for both cohorts, plus their ratio."""

    adv_hist: Interval
    adv_post: Interval
    delta: Interval
    ratio: Interval
    adv_hist_vs_mfc: Interval
    adv_post_vs_mfc: Interval
    mfc_hist: float
    mfc_post: float
    uniform_chance: float
    n_boot: int
    n_seeds: int

    def as_dict(self) -> dict:
        return {
            "adv_hist": self.adv_hist.as_dict(),
            "adv_post": self.adv_post.as_dict(),
            "delta": self.delta.as_dict(),
            "ratio": self.ratio.as_dict(),
            "adv_hist_vs_mfc": self.adv_hist_vs_mfc.as_dict(),
            "adv_post_vs_mfc": self.adv_post_vs_mfc.as_dict(),
            "mfc_hist": self.mfc_hist,
            "mfc_post": self.mfc_post,
            "uniform_chance": self.uniform_chance,
            "n_boot": self.n_boot,
            "n_seeds": self.n_seeds,
        }


def bootstrap_pair(
    hist_hits: np.ndarray,
    post_hits: np.ndarray,
    hist_labels: np.ndarray,
    post_labels: np.ndarray,
    n_topics: int,
    n_boot: int = N_BOOT,
    seed: int = 20_260_814,
) -> PairResult:
    """Cluster-bootstrap both cohorts jointly.

    Parameters
    ----------
    hist_hits, post_hits
        ``(n_seeds, n_entries)`` boolean arrays: was entry *j* correctly classified in trial *i*.
    hist_labels, post_labels
        True topic index per entry, for the most-frequent-class baseline.
    n_topics
        Size of the label space, for the uniform-chance baseline.
    """
    rng = np.random.default_rng(seed)
    h = np.asarray(hist_hits, dtype=np.float64)
    p = np.asarray(post_hits, dtype=np.float64)
    hl = np.asarray(hist_labels, dtype=np.int64)
    pl = np.asarray(post_labels, dtype=np.int64)

    n_seeds, n_hist = h.shape
    n_post = p.shape[1]
    chance = 1.0 / max(1, n_topics)

    adv_h = np.empty(n_boot)
    adv_p = np.empty(n_boot)
    adv_h_mfc = np.empty(n_boot)
    adv_p_mfc = np.empty(n_boot)

    # Fast path for a completely defeated attack. Mnemos scores exactly zero on every entry in
    # every trial, so every resample is also exactly zero -- there is nothing for the bootstrap
    # to explore, and running 10,000 iterations to rediscover that wastes minutes across the
    # sweep. The interval [0, 0] is the correct answer, not an approximation of it.
    if not h.any() and not p.any():
        zero = Interval(0.0, 0.0, 0.0)
        return PairResult(
            adv_hist=zero,
            adv_post=zero,
            delta=zero,
            ratio=zero,
            adv_hist_vs_mfc=zero,
            adv_post_vs_mfc=zero,
            mfc_hist=mfc_accuracy(hl),
            mfc_post=mfc_accuracy(pl),
            uniform_chance=chance,
            n_boot=0,
            n_seeds=n_seeds,
        )

    uniform_seeds = np.full(n_seeds, 1.0 / n_seeds)
    uniform_hist = np.full(n_hist, 1.0 / n_hist)
    uniform_post = np.full(n_post, 1.0 / n_post)

    for b in range(n_boot):
        # Outer level: resample TRIALS. Shared across cohorts so the ratio stays paired.
        seed_w = rng.multinomial(n_seeds, uniform_seeds).astype(np.float64)
        # Inner level: resample ENTRIES, independently per cohort -- the cohorts are disjoint
        # sets of entries, so there is no pairing between them to preserve.
        hw = rng.multinomial(n_hist, uniform_hist).astype(np.float64)
        pw = rng.multinomial(n_post, uniform_post).astype(np.float64)

        acc_h = _weighted_accuracy(h, seed_w, hw)
        acc_p = _weighted_accuracy(p, seed_w, pw)
        adv_h[b] = max(0.0, acc_h - chance)
        adv_p[b] = max(0.0, acc_p - chance)
        adv_h_mfc[b] = max(0.0, acc_h - _weighted_mfc(hl, hw, n_topics))
        adv_p_mfc[b] = max(0.0, acc_p - _weighted_mfc(pl, pw, n_topics))

    # The ratio is formed INSIDE the resampling, never by dividing two separate intervals.
    ratio = np.where(adv_p > 0.0, adv_h / np.where(adv_p > 0.0, adv_p, 1.0), np.nan)

    point_h = max(0.0, float(np.mean(h)) - chance)
    point_p = max(0.0, float(np.mean(p)) - chance)

    return PairResult(
        adv_hist=_interval(point_h, adv_h),
        adv_post=_interval(point_p, adv_p),
        delta=_interval(point_p - point_h, adv_p - adv_h),
        ratio=_interval((point_h / point_p) if point_p > 0 else 0.0, ratio),
        adv_hist_vs_mfc=_interval(max(0.0, float(np.mean(h)) - mfc_accuracy(hl)), adv_h_mfc),
        adv_post_vs_mfc=_interval(max(0.0, float(np.mean(p)) - mfc_accuracy(pl)), adv_p_mfc),
        mfc_hist=mfc_accuracy(hl),
        mfc_post=mfc_accuracy(pl),
        uniform_chance=chance,
        n_boot=n_boot,
        n_seeds=n_seeds,
    )


def _interval(point: float, draws: np.ndarray) -> Interval:
    """Percentile interval, ignoring undefined draws."""
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return Interval(point=point, lo=float("nan"), hi=float("nan"))
    lo, hi = np.percentile(finite, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return Interval(point=float(point), lo=float(lo), hi=float(hi))
