"""A4 -- induced-write injection, and the **structure-transfer** attack it enables.

This is the attack the prior-work check pointed at: *"the intersection -- an autonomous agent as
the writer, whose writes an adversary can induce post-compromise, analyzed through a DSSE
leakage lens -- is an open seam."*

Why injection is fundamentally stronger than snapshot counting
==============================================================
Attack A3 tries to infer structure from static counts and is capped near 12% by tie collisions
(finding F1). Injection is a different kind of thing entirely, because **the adversary has
ground truth for its own writes**:

    Feed the agent a SQL task. The agent authors a SQL memory. That memory lands in whichever
    partition holds SQL. Watch which partition grew, and you have learned that partition's
    topic -- with certainty, not inference.

The adversary controls the *stimulus*, not the content. That is weaker than classic file
injection (Zhang-Katz-Papamanthou) in one way and stronger in another: it cannot choose the
exact bytes written, but the writes are **authentic** -- indistinguishable from organic agent
output, because they are organic agent output. No content filter can separate them.

The real finding: structure transfer
====================================
Mapping partitions is not the payload. The payload is what the map unlocks:

    Partitions are shared between cohorts. A partition holding post-compromise SQL memories
    also holds *pre-compromise* SQL memories. So a map learned entirely from post-compromise
    injections applies directly to the historical corpus.

**The historical cohort is not protected by having been written before the adversary arrived.**
Its protection depends on whether the structure learned after tau transfers backwards -- and by
default it does, completely.

This sharpens the asymmetry claim rather than weakening it. ``L_pre ⊑ L_post`` still holds, but
the *magnitude* of what is computable about H rises with injection budget. The historical
corpus is safe only when transfer is blocked, and blocking transfer is precisely what padding
and epoch batching do. Attack, defence, and Theorem 1's epoch knob all land on the same
mechanism.

Defences, and how each fails
============================
**Epoch batching** hides which of ``epoch_writes`` records in a flush was the injection. It
does not survive repetition: inject the same topic ``r`` times and the true partition appears
in every flush's touched set, while co-flushed organic writes vary. Intersecting the sets
recovers it. Batching raises the cost from ``r=1`` to ``r≈log(B)``; it does not stop a patient
adversary. **That is a result worth stating plainly** -- it says batching alone is insufficient.

**Padding to capacity** is what actually works. If every partition occupies constant bytes
whatever it holds, no growth is observable, the intersection has nothing to intersect, and the
attack collapses to chance regardless of injection budget.

Fairness
--------
Reads only what the adversary observes: the set of partitions touched per flush, and its own
injection ground truth. Never the corpus labels, never engine internals.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from eval.lab.buckets import Partitioning
from eval.lab.corpus import Corpus, Entry


@dataclass(frozen=True, eq=False)
class InjectionResult:
    """One A4 trial.

    ``eq=False`` because the per-entry outcome arrays below are numpy arrays, and a generated
    ``__eq__`` over them returns an array rather than a bool.
    """

    scheme: str
    epoch_writes: int
    injections_per_topic: int
    padded: bool

    buckets_mapped: int
    """Partitions the adversary confidently labelled."""
    buckets_total: int

    hist_correct: int
    """Historical entries whose topic was correctly recovered **via structure transfer**."""
    hist_total: int
    post_correct: int
    """Post-compromise entries correctly recovered."""
    post_total: int

    n_topics: int
    seed: int

    hist_hits: np.ndarray
    """Per-entry correctness over cohort H, in corpus order. **Required for bootstrapping.**

    Storing only ``hist_correct`` would leave nothing to resample: a single proportion carries
    no information about how it would vary, so a confidence interval could not be computed
    after the fact. The per-entry vector is the smallest thing that does."""
    post_hits: np.ndarray
    """Per-entry correctness over cohort P."""
    hist_labels: np.ndarray
    """True topic index per historical entry -- the input to the most-frequent-class baseline."""
    post_labels: np.ndarray

    @property
    def hist_accuracy(self) -> float:
        return self.hist_correct / self.hist_total if self.hist_total else 0.0

    @property
    def post_accuracy(self) -> float:
        return self.post_correct / self.post_total if self.post_total else 0.0

    @property
    def adv_hist(self) -> float:
        """Historical-cohort advantage over chance. **The transfer result.**"""
        return max(0.0, self.hist_accuracy - 1.0 / max(1, self.n_topics))

    @property
    def adv_post(self) -> float:
        return max(0.0, self.post_accuracy - 1.0 / max(1, self.n_topics))

    @property
    def delta(self) -> float:
        """``Adv_post - Adv_hist`` -- Theorem 1's quantity, measured."""
        return self.adv_post - self.adv_hist

    @property
    def mfc_hist(self) -> float:
        """Accuracy of the trivial "always guess the most common topic" adversary on cohort H.

        This is a *different* and generally *stronger* baseline than uniform chance. Uniform
        chance assumes topics are equally likely; real corpora are skewed, so a classifier that
        ignores its input entirely and always answers with the modal topic already beats
        ``1/n_topics``. Reporting advantage only over uniform chance would credit the attack for
        exploiting corpus skew it never actually looked at."""
        return _mfc(self.hist_labels)

    @property
    def mfc_post(self) -> float:
        return _mfc(self.post_labels)

    @property
    def adv_hist_vs_mfc(self) -> float:
        """The conservative headline: advantage over the trivial adversary, not over a coin."""
        return max(0.0, self.hist_accuracy - self.mfc_hist)

    @property
    def adv_post_vs_mfc(self) -> float:
        return max(0.0, self.post_accuracy - self.mfc_post)


def _mfc(labels: np.ndarray) -> float:
    """Most-frequent-class accuracy: the modal label's share."""
    if labels.size == 0:
        return 0.0
    counts = np.bincount(labels.astype(np.int64))
    return float(counts.max()) / float(labels.size)


def _simulate_flushes(
    injected_bucket: int,
    n_buckets: int,
    epoch_writes: int,
    repetitions: int,
    rng: np.random.Generator,
) -> list[set[int]]:
    """Partitions observed as touched, per flush, across repeated injections of one topic.

    Each flush contains the adversary's injected write plus ``epoch_writes - 1`` organic
    writes from the agent's ordinary work, landing in partitions the adversary does not
    control. The adversary sees the union, never the attribution.
    """
    flushes = []
    for _ in range(repetitions):
        organic = rng.choice(n_buckets, size=max(0, epoch_writes - 1), replace=True)
        flushes.append({injected_bucket, *(int(b) for b in organic)})
    return flushes


def _intersect(flushes: list[set[int]]) -> set[int]:
    """Candidate partitions consistent with every observed flush.

    The injected partition is present in all of them by construction; an organic partition
    survives only by appearing in every flush by chance, which decays geometrically in the
    number of repetitions. This is what defeats epoch batching.
    """
    if not flushes:
        return set()
    out = set(flushes[0])
    for f in flushes[1:]:
        out &= f
    return out


def run(
    corpus: Corpus,
    part: Partitioning,
    epoch_writes: int,
    injections_per_topic: int,
    padded: bool,
    seed: int,
) -> InjectionResult:
    """Run one A4 trial.

    Parameters
    ----------
    padded
        ``True`` models Mnemos: partitions occupy constant bytes, so no growth is observable
        and the adversary has nothing to intersect. ``False`` models an unpadded partitioned
        index -- including a forward-private one, since forward privacy says nothing about
        region sizes.
    """
    rng = np.random.default_rng(seed)

    topics = sorted({e.topic for e in corpus.all_entries})
    n_topics = len(topics)

    bucket_label: dict[int, str] = {}

    if not padded:
        # A real injector does not send one canonical task per topic. It sends many varied
        # ones -- "fix the deadlock in the pool", "why is this query slow", "add an index" --
        # which scatter across every partition that topic occupies. With 256 partitions and
        # 10 topics each topic spans roughly 25 of them, so mapping a topic means localising
        # all of them, not one.
        #
        # We model the induced writes by sampling from the POST cohort: an induced write is
        # indistinguishable from organic agent output, because it is organic agent output.
        # The adversary knows each task's topic (it chose it) but must discover the partition.
        budget = injections_per_topic * n_topics
        pool = corpus.post
        picks = rng.choice(len(pool), size=min(budget, len(pool)), replace=False)

        # Group injections by the partition they landed in, so repeated hits on the same
        # partition compound into a sharper intersection -- which is how a patient adversary
        # beats epoch batching.
        by_bucket: dict[int, list[str]] = defaultdict(list)
        for i in picks:
            e = pool[int(i)]
            by_bucket[part.bucket_of[e.doc_id]].append(e.topic)

        for target, observed_topics in by_bucket.items():
            flushes = _simulate_flushes(
                target, part.n_buckets, epoch_writes, len(observed_topics), rng
            )
            surviving = _intersect(flushes)

            # A confident mapping requires the intersection to have resolved to a single
            # partition. Ambiguous results are discarded rather than guessed -- a real
            # adversary would simply inject more, and scoring ambiguous cases as hits would
            # inflate the attack.
            if len(surviving) == 1:
                b = surviving.pop()
                # Label with the majority topic among writes the adversary steered here.
                bucket_label[b] = Counter(observed_topics).most_common(1)[0][0]

    # --- structure transfer -----------------------------------------------------------
    # The map was learned entirely from post-compromise injections. It is now applied to
    # BOTH cohorts. Nothing about the historical cohort was observed directly.
    topic_index = {t: i for i, t in enumerate(topics)}

    def hits(entries: list[Entry]) -> np.ndarray:
        """Per-entry correctness, as a boolean vector rather than a count."""
        return np.fromiter(
            (bucket_label.get(part.bucket_of[e.doc_id]) == e.topic for e in entries),
            dtype=bool,
            count=len(entries),
        )

    def labels(entries: list[Entry]) -> np.ndarray:
        return np.fromiter(
            (topic_index[e.topic] for e in entries), dtype=np.int64, count=len(entries)
        )

    hist_hits = hits(corpus.historical)
    post_hits = hits(corpus.post)

    return InjectionResult(
        scheme="mnemos-padded" if padded else "unpadded-partition-index",
        epoch_writes=epoch_writes,
        injections_per_topic=injections_per_topic,
        padded=padded,
        buckets_mapped=len(bucket_label),
        buckets_total=part.n_buckets,
        hist_correct=int(hist_hits.sum()),
        hist_total=len(corpus.historical),
        post_correct=int(post_hits.sum()),
        post_total=len(corpus.post),
        n_topics=n_topics,
        seed=seed,
        hist_hits=hist_hits,
        post_hits=post_hits,
        hist_labels=labels(corpus.historical),
        post_labels=labels(corpus.post),
    )
