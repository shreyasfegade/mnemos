"""Partition-labelling estimators for structure transfer.

All three take the same adversary view: for each topic ``t`` the adversary induced, a list of
flushes, each an integer array of the ``w`` partitions that flush wrote (the induced record's
partition among them, position unknown), plus an estimate ``pi`` of the organic write
distribution over partitions obtained from flushes it did not induce.

``oracle_intersection``
    The estimator the earlier paper headlined. It groups induced records by the partition they
    actually landed in, which the adversary cannot know, and intersects each group's flushes.
    Kept only as an upper bound.

``frequency``
    Counts, per topic, how often each partition appears in that topic's flushes and labels a
    partition with the topic whose rate most exceeds the organic null rate, if the excess is
    significant.

``em``
    Maximum likelihood by expectation-maximisation. Each flush of topic ``t`` holds one record
    drawn from an unknown partition distribution ``theta_t`` and ``w - 1`` organic records drawn
    from ``pi``. The posterior that position ``j`` holds the induced record is proportional to
    ``theta_t[s_j] / pi[s_j]``.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

Flushes = dict[int, list[np.ndarray]]


def oracle_intersection(
    true_partition_groups: dict[int, list[int]],
    n_buckets: int,
    w: int,
    organic_partitions: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """``true_partition_groups[b]`` lists the topics of induced records that landed in ``b``."""
    labels = np.full(n_buckets, -1, dtype=np.int64)
    for b, topics in true_partition_groups.items():
        surviving: set[int] | None = None
        for _ in topics:
            org = rng.choice(organic_partitions, size=w - 1, replace=True) if w > 1 else []
            s = {int(b), *(int(x) for x in org)}
            surviving = s if surviving is None else surviving & s
        if surviving is not None and len(surviving) == 1:
            labels[surviving.pop()] = Counter(topics).most_common(1)[0][0]
    return labels


def frequency(
    flushes: Flushes, pi: np.ndarray, w: int, n_buckets: int, z: float = 2.0
) -> np.ndarray:
    labels = np.full(n_buckets, -1, dtype=np.int64)
    topics = [t for t, f in flushes.items() if f]
    if not topics:
        return labels
    null = 1.0 - (1.0 - pi) ** max(w - 1, 0)
    lift = np.full((len(topics), n_buckets), -np.inf)
    thresh = np.zeros((len(topics), n_buckets))
    counts = np.zeros((len(topics), n_buckets))
    for i, t in enumerate(topics):
        c = np.zeros(n_buckets)
        for s in flushes[t]:
            c[np.unique(s)] += 1.0
        n = len(flushes[t])
        counts[i] = c
        lift[i] = c / n - null
        thresh[i] = z * np.sqrt(np.maximum(null * (1.0 - null), 1e-12) / n)
    best = np.argmax(lift, axis=0)
    cols = np.arange(n_buckets)
    ok = (lift[best, cols] > thresh[best, cols]) & (counts[best, cols] >= 1)
    labels[ok] = np.asarray(topics)[best[ok]]
    return labels


def em(
    flushes: Flushes,
    pi: np.ndarray,
    n_buckets: int,
    iters: int = 100,
    alpha: float = 1e-3,
    min_mass: float = 0.5,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Returns partition labels and each topic's estimated partition distribution."""
    labels = np.full(n_buckets, -1, dtype=np.int64)
    topics = [t for t, f in flushes.items() if f]
    thetas: dict[int, np.ndarray] = {}
    mass = np.zeros((len(topics), n_buckets))
    inv_pi = 1.0 / np.maximum(pi, 1e-12)
    for i, t in enumerate(topics):
        S = np.vstack(flushes[t])  # (n, w)
        n, w = S.shape
        # Initialise from the organic-corrected counts.
        c = np.bincount(S.ravel(), minlength=n_buckets).astype(np.float64)
        theta = np.maximum(c - (w - 1) * n * pi, 1e-3)
        theta /= theta.sum()
        for _ in range(iters):
            like = theta[S] * inv_pi[S]  # (n, w)
            r = like / like.sum(axis=1, keepdims=True)
            new = np.bincount(S.ravel(), weights=r.ravel(), minlength=n_buckets) + alpha
            new /= new.sum()
            if np.max(np.abs(new - theta)) < 1e-9:
                theta = new
                break
            theta = new
        thetas[t] = theta
        mass[i] = theta * n
    if topics:
        best = np.argmax(mass, axis=0)
        cols = np.arange(n_buckets)
        ok = mass[best, cols] >= min_mass
        labels[ok] = np.asarray(topics)[best[ok]]
    return labels, thetas


def score(
    labels: np.ndarray, partitions: np.ndarray, topics: np.ndarray, n_topics: int | None = None
) -> dict[str, float]:
    """Accuracy, coverage and purity over a set of records.

    ``accuracy`` counts a record in an unlabelled partition as wrong. That is the accuracy of an
    adversary that *abstains* where it has no label, and it sits below the chance rate whenever
    coverage is low -- not because the adversary learned anything misleading, but because it
    declined to guess. Reported as an advantage over chance it reads as a negative result where
    there is only an absent one.

    ``accuracy_guess`` (when ``n_topics`` is given) is the adversary that guesses uniformly where
    it has no label, credited at the expected 1/K. No rational adversary does worse than that on
    the records it cannot label, so this is the quantity an advantage over chance should be taken
    from. Both are returned so the difference is visible rather than chosen silently.
    """
    lab = labels[partitions]
    covered = lab >= 0
    hits = lab == topics
    n = len(topics)
    out = {
        "accuracy": float(hits.mean()) if n else 0.0,
        "coverage": float(covered.mean()) if n else 0.0,
        "purity": float(hits.sum() / covered.sum()) if covered.any() else float("nan"),
        "mapped": int((labels >= 0).sum()),
    }
    if n_topics is not None:
        out["accuracy_guess"] = float((hits.sum() + (~covered).sum() / n_topics) / n) if n else 0.0
    return out


def balanced_guess_accuracy(pred, truth, n_topics: int) -> float:
    """Mean per-topic recall, for the adversary that guesses uniformly where it has no label
    (``pred`` < 0). Unlike accuracy, it does not reward the topic prior: an adversary that always
    names cohort H's most frequent topic scores exactly 1/K however skewed the prior is, so an
    advantage here is information about individual records, not about the topic mix."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    recalls = []
    for c in np.unique(truth):
        mask = truth == c
        hits = int((pred[mask] == c).sum())
        unlabelled = int((pred[mask] < 0).sum())
        recalls.append((hits + unlabelled / n_topics) / int(mask.sum()))
    return float(np.mean(recalls)) if recalls else 0.0


def guess_accuracy(n_hits: int, n_unlabelled: int, n: int, n_topics: int) -> float:
    """``accuracy_guess`` for harnesses that count hits themselves: see :func:`score`."""
    return (n_hits + n_unlabelled / n_topics) / n if n else 0.0


def group_by_true_partition(parts: np.ndarray, topics: np.ndarray) -> dict[int, list[int]]:
    g: dict[int, list[int]] = defaultdict(list)
    for p, t in zip(parts, topics, strict=True):
        g[int(p)].append(int(t))
    return g
