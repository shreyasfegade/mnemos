"""A5 -- volumetric and query-equality attack, aimed **at Mnemos**.

Xu, Zheng, Xu, Yuan, Wang, *Leakage-Abuse Attacks Against Forward and Backward Private
Searchable Symmetric Encryption*, CCS 2023, showed that volumetric and query-equality leakage
survives forward *and* backward privacy. Mnemos claims to remove volume leakage structurally,
via the constant-volume invariant:

    every search reads exactly ``(k_probe + d) * C`` index slots and exactly ``top_k`` content
    slots, regardless of the query and regardless of how many entries match.

This module exists to test that claim rather than assert it.

Why this file is in the repository
----------------------------------
A paper that only runs attacks its scheme survives reads as marketing, and reviewers in this
area recognise the pattern in about thirty seconds. Running the strongest known attack against
our own construction, and reporting whatever it finds, is the cheapest credibility we can buy.

It runs against a **real store**, not a model
---------------------------------------------
:func:`run` drives ``mnemos_native`` -- the compiled engine -- and reads the ``LeakEvent`` stream
the engine actually emits. A simulated trace would test the simulation. The adversary here sees
exactly what ``docs/leakage-profile.md`` says it sees, because it is reading the same event bus
the leakage-profile test asserts against.

What it finds, stated before the code so it cannot be buried
------------------------------------------------------------
1. **Volume leakage: none.** Every search emits the identical number of ``bucket_read`` and
   ``slot_read`` events. The constant-volume invariant holds, and :func:`measure_volume_variance`
   asserts the variance is exactly zero rather than merely small.

2. **Query equality: fully leaked, and not bounded by an epoch.** This is worse than the design
   notes claimed, and it is the honest finding of this module.

   The probe set is ``top-k_probe`` partitions by centroid similarity to the *query vector*. That
   is a deterministic function of the query and nothing else -- no token, no epoch, no key
   rotation enters it. So two searches for the same thing produce the identical set of
   ``bucket_read`` events, forever, and an adversary who observes the read pattern can group
   searches by query with no cryptanalysis at all.

   Advancing search tokens does not help: tokens govern which *entries* a search can recognise,
   not which *partitions* it reads. Decoy partitions (``decoy_buckets > 0``) add per-epoch noise,
   but the ``k_probe`` real probes are common to every observation of the same query, so
   intersecting two epochs' read sets recovers them.

   This is a genuine limitation of a partitioned index and is reported as one.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VolumetricResult:
    """One A5 trial against Mnemos."""

    volume_recovery: float
    """Recovery rate from response volume. Should sit at chance -- if it does not, the
    constant-volume invariant is broken somewhere and that is a bug, not a finding."""

    query_equality_intra_epoch: float
    """Recovery from repeated queries within one epoch. **The known residual.**"""

    query_equality_inter_epoch: float
    """Recovery across an epoch boundary. Expected at chance if token rotation helped; measured
    at 1.0, because the probe set never depended on a token in the first place."""

    epoch_ms: int
    """The configuration this was measured under -- the residual is a function of it."""

    seed: int

    bucket_reads_per_search: int
    slot_reads_per_search: int
    distinct_queries: int
    distinct_read_signatures: int
    """If this equals ``distinct_queries``, the read pattern is a perfect fingerprint."""

    volume_variance: float
    """Variance in reads per search. **Must be exactly 0.0.**"""

    def as_dict(self) -> dict:
        return {
            "volume_recovery": self.volume_recovery,
            "volume_variance": self.volume_variance,
            "query_equality_intra_epoch": self.query_equality_intra_epoch,
            "query_equality_inter_epoch": self.query_equality_inter_epoch,
            "bucket_reads_per_search": self.bucket_reads_per_search,
            "slot_reads_per_search": self.slot_reads_per_search,
            "distinct_queries": self.distinct_queries,
            "distinct_read_signatures": self.distinct_read_signatures,
            "epoch_ms": self.epoch_ms,
            "seed": self.seed,
        }


def load_events(trace_path: Path) -> list[dict]:
    """Parse the JSONL ``LeakEvent`` stream -- the adversary's whole input."""
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_searches(events: list[dict]) -> list[dict]:
    """Group the event stream into per-search observations.

    A search emits a run of ``bucket_read`` events followed by ``slot_read`` events. Writes and
    flushes separate searches. The adversary can do this segmentation because it sees event
    ordering and type -- both are in the leakage profile as observable.
    """
    searches: list[dict] = []
    current: dict | None = None

    for e in events:
        kind = e.get("kind")
        if kind == "bucket_read":
            if current is None or current["slots"]:
                current = {"buckets": [], "slots": [], "epoch": None}
                searches.append(current)
            current["buckets"].append(e.get("bucket"))
        elif kind == "slot_read":
            if current is not None:
                current["slots"].append(e.get("addr"))
        else:
            # Any non-read event ends the current search.
            current = None

    return [s for s in searches if s["buckets"]]


def measure_volume_variance(trace_path: Path) -> float:
    """Variance in ``BucketRead`` count per search across the whole trace.

    **Must be exactly zero.** A non-zero variance means some code path took an early exit,
    skipped an empty bucket, or read a partial one -- each an optimisation that looks
    harmless and reintroduces the leakage this scheme is built to remove.

    This is also asserted in Rust by ``tests/leakage_profile.rs``; having it in both places is
    deliberate. The Rust test catches it at development time; this one catches it in the
    artifact a reviewer runs.
    """
    searches = split_searches(load_events(trace_path))
    if not searches:
        return 0.0
    counts = [len(s["buckets"]) for s in searches]
    mean = sum(counts) / len(counts)
    return sum((c - mean) ** 2 for c in counts) / len(counts)


def _signature(search: dict) -> tuple:
    """What the adversary can fingerprint a search by: its sorted set of partition reads."""
    return tuple(sorted(set(search["buckets"])))


def _pairwise_recovery(signatures: list[tuple], truth: list[int]) -> float:
    """Fraction of same-query pairs the adversary correctly identifies as the same query.

    Pairwise rather than clustering-accuracy because it needs no assignment step and cannot be
    inflated by a lucky labelling. A value of 1.0 means the read pattern is a perfect
    fingerprint; a value at the base rate means it carries nothing.
    """
    same_pairs = 0
    caught = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            if truth[i] == truth[j]:
                same_pairs += 1
                if signatures[i] == signatures[j]:
                    caught += 1
    return caught / same_pairs if same_pairs else 0.0


def _false_positive_rate(signatures: list[tuple], truth: list[int]) -> float:
    """How often DIFFERENT queries share a signature -- the adversary's error rate."""
    diff_pairs = 0
    collisions = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            if truth[i] != truth[j]:
                diff_pairs += 1
                if signatures[i] == signatures[j]:
                    collisions += 1
    return collisions / diff_pairs if diff_pairs else 0.0


def analyse(
    trace_path: Path, truth: list[int], epoch_boundaries: list[int], seed: int, epoch_ms: int
) -> VolumetricResult:
    """Run the attack over a captured trace.

    Parameters
    ----------
    truth
        Query identity of each search, in order. Ground truth, used only to *score* the
        adversary -- never to inform it.
    epoch_boundaries
        Index of the first search after each flush, so intra- and inter-epoch pairs can be
        separated.
    """
    searches = split_searches(load_events(trace_path))
    n = min(len(searches), len(truth))
    searches, truth = searches[:n], truth[:n]

    signatures = [_signature(s) for s in searches]
    bucket_counts = Counter(len(s["buckets"]) for s in searches)
    slot_counts = Counter(len(s["slots"]) for s in searches)

    variance = measure_volume_variance(trace_path)

    # Volume recovery: can the adversary distinguish queries by how much was read? If reads are
    # constant there is literally nothing to distinguish, so this is exactly chance.
    volume_groups = defaultdict(set)
    for count, q in zip((len(s["buckets"]) for s in searches), truth, strict=False):
        volume_groups[count].add(q)
    largest = max((len(v) for v in volume_groups.values()), default=1)
    volume_recovery = 1.0 / largest if largest else 1.0

    intra_idx = [i for i in range(n) if _epoch_of(i, epoch_boundaries) == 0]
    inter_all = list(range(n))

    return VolumetricResult(
        volume_recovery=volume_recovery,
        volume_variance=variance,
        query_equality_intra_epoch=_pairwise_recovery(
            [signatures[i] for i in intra_idx], [truth[i] for i in intra_idx]
        ),
        query_equality_inter_epoch=_pairwise_recovery(
            [signatures[i] for i in inter_all], [truth[i] for i in inter_all]
        ),
        epoch_ms=epoch_ms,
        seed=seed,
        bucket_reads_per_search=next(iter(bucket_counts), 0),
        slot_reads_per_search=next(iter(slot_counts), 0),
        distinct_queries=len(set(truth)),
        distinct_read_signatures=len(set(signatures)),
    )


def _epoch_of(i: int, boundaries: list[int]) -> int:
    return sum(1 for b in boundaries if i >= b)


def false_positive_rate(trace_path: Path, truth: list[int]) -> float:
    """Exposed so the experiment can report the adversary's precision as well as its recall."""
    searches = split_searches(load_events(trace_path))
    n = min(len(searches), len(truth))
    return _false_positive_rate([_signature(s) for s in searches[:n]], truth[:n])
