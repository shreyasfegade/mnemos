"""E3-sensitivity -- what does attack A3 actually need from the corpus?

Motivation
==========
The first A3 runs produced ``advantage = 0.000`` on every scheme, which reads like a dead
finding. It is not; it is an uninformative measurement, and the difference matters.

A3 matches *observed* structure in the snapshot against *expected* structure inferred from
auxiliary knowledge. If the corpus has no structure to observe, the attack has nothing to work
with and a null result says something about the corpus, not about the scheme. The D3 corpus is
structureless in exactly the two ways that matter:

**Topic prevalence is uniform.** Exactly 500 notes per topic, ten topics. Real agent memory is
nothing like this -- an agent works mostly on one codebase and occasionally on others, so
prevalence is heavily skewed.

**Authoring order is topic-blocked, then we shuffled it flat.** The raw corpus puts all 500
``sql`` notes consecutively, which is unrealistic in the opposite direction and would make
cohorts H and P differ by topic rather than by time (see ``lab/corpus.py``). Shuffling fixes
that confound but removes temporal structure entirely. Real agents fall in between: work
clusters into sprints and projects.

So rather than tuning the corpus until the attack succeeds -- which would be fitting the
experiment to the desired answer -- this sweep measures **how much of each property the attack
requires**. The output is a sensitivity surface, and it converts an ambiguous null into a
precise, falsifiable statement of the form:

    A3 achieves advantage >= X once topic prevalence skew exceeds S and temporal clustering
    exceeds T; below that it is indistinguishable from guessing.

That statement is publishable either way. If real agent memory sits above the threshold, the
finding holds and we know why. If it sits below, the finding is dead and we know that in
August rather than in November.

Axes
====
``skew``
    Zipf exponent on topic prevalence. ``0.0`` reproduces D3's uniform 500-per-topic;
    ``1.0`` is a strongly dominant-topic corpus.

``clustering``
    Fraction of authoring order that is topic-blocked. ``0.0`` is fully interleaved (our
    current shuffle); ``1.0`` is fully blocked (raw D3). Real work is somewhere in between.

Reading the output
==================
The threshold contour is the deliverable, not any single cell. Report the whole surface,
including the large region where the attack fails -- that region is the honest part.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval.attacks import a3_snapshot as A3
from eval.lab.corpus import Corpus, Entry

N_DOCS = 4000
N_TOPICS = 10
KEYWORDS_PER_TOPIC = 50
TAGS_PER_DOC = 3


@dataclass(frozen=True)
class Cell:
    """One point on the sensitivity surface."""

    skew: float
    clustering: float
    aux_fraction: float
    seed: int
    advantage: float
    keyword_recovery: float
    tie_ceiling: float
    dominant_topic_share: float
    """Prevalence of the most common topic. A readable proxy for ``skew``."""


def synth_corpus(
    skew: float,
    clustering: float,
    seed: int,
    n_docs: int = N_DOCS,
) -> Corpus:
    """Build a corpus with controlled topic skew and temporal clustering.

    Keywords are topic-scoped: each topic owns a private block of ``KEYWORDS_PER_TOPIC``
    terms. That is what makes keyword counts informative about topics at all -- and it is
    realistic, since domain vocabulary really is domain-specific.
    """
    rng = np.random.default_rng(seed)

    # --- topic prevalence -------------------------------------------------------------
    ranks = np.arange(1, N_TOPICS + 1)
    p_topic = 1.0 / ranks**skew
    p_topic /= p_topic.sum()

    topics = rng.choice(N_TOPICS, size=n_docs, p=p_topic)

    # --- temporal clustering ----------------------------------------------------------
    # clustering = 0 -> fully interleaved; 1 -> fully topic-blocked. We interpolate by
    # sorting a `clustering`-weighted blend of the topic id and a random jitter, which
    # yields a continuous sweep between the two regimes rather than a binary switch.
    jitter = rng.random(n_docs)
    key = clustering * (topics / N_TOPICS) + (1.0 - clustering) * jitter
    order = np.argsort(key, kind="stable")
    topics = topics[order]

    # --- keywords ---------------------------------------------------------------------
    entries: list[Entry] = []
    for doc_id, t in enumerate(topics):
        t = int(t)
        base = t * KEYWORDS_PER_TOPIC
        # Zipf within the topic's own vocabulary, so some terms are common and some rare --
        # the head/tail structure count attacks depend on.
        local_ranks = np.arange(1, KEYWORDS_PER_TOPIC + 1)
        p_kw = 1.0 / local_ranks**1.0
        p_kw /= p_kw.sum()
        picks = rng.choice(KEYWORDS_PER_TOPIC, size=TAGS_PER_DOC, replace=False, p=p_kw)
        entries.append(
            Entry(
                doc_id=doc_id,
                seq=doc_id,
                keywords=tuple(f"kw{base + int(k):05d}" for k in picks),
                topic=f"topic{t}",
                text="",
            )
        )

    vocab = tuple(sorted({w for e in entries for w in e.keywords}))
    split = n_docs // 2
    return Corpus(historical=entries[:split], post=entries[split:], vocabulary=vocab)


def _sophos_snapshot(corpus: Corpus, cohort: list[Entry], rng: np.random.Generator):
    """Per-keyword counts under opaque labels -- the ``sophos-local`` leak."""
    from eval.lab.schemes import SnapshotView

    counts = corpus.keyword_counts(cohort)
    keywords = list(counts)
    perm = rng.permutation(len(keywords))
    label_of = {keywords[int(p)]: i for i, p in enumerate(perm)}
    view = SnapshotView(
        token_counts={label_of[w]: c for w, c in counts.items()},
        total_entries=len(cohort),
        scheme="sophos-local",
    )
    return view, {lab: kw for kw, lab in label_of.items()}


def run_cell(skew: float, clustering: float, aux: float, seed: int) -> Cell:
    corpus = synth_corpus(skew, clustering, seed)
    cohort = corpus.historical
    rng = np.random.default_rng(seed)

    snap, truth = _sophos_snapshot(corpus, cohort, rng)
    r = A3.run(snap, corpus, cohort, aux, seed, truth=truth)

    topic_counts: dict[str, int] = {}
    for e in cohort:
        topic_counts[e.topic] = topic_counts.get(e.topic, 0) + 1
    dominant = max(topic_counts.values()) / len(cohort)

    return Cell(
        skew=skew,
        clustering=clustering,
        aux_fraction=aux,
        seed=seed,
        advantage=r.advantage,
        keyword_recovery=r.keyword_recovery_rate,
        tie_ceiling=r.tie_ceiling,
        dominant_topic_share=dominant,
    )


def main(out: Path | None = None) -> list[Cell]:
    skews = [0.0, 0.5, 1.0, 1.5, 2.0]
    clusterings = [0.0, 0.5, 1.0]
    seeds = range(5)

    cells = [run_cell(s, c, 0.25, seed) for s in skews for c in clusterings for seed in seeds]

    print(f"{'skew':>5} {'clust':>6} {'dom%':>6} {'ceiling':>8} {'kw_rec':>7} {'ADV':>7}")
    for s in skews:
        for c in clusterings:
            group = [x for x in cells if x.skew == s and x.clustering == c]
            print(
                f"{s:5.1f} {c:6.1f} {np.mean([g.dominant_topic_share for g in group]):6.2f} "
                f"{np.mean([g.tie_ceiling for g in group]):8.3f} "
                f"{np.mean([g.keyword_recovery for g in group]):7.3f} "
                f"{np.mean([g.advantage for g in group]):7.3f}"
            )

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(c) for c in cells], indent=2), encoding="utf-8")
    return cells


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[3] / "python/eval/results/e3_sensitivity.json")
