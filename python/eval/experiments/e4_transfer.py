"""E4 -- the structure-transfer experiment. **The paper's central result.**

Claim under test
================
    Post-compromise injection maps the semantic structure of a partitioned memory store, and
    that map applies *backwards* to entries written before the compromise. Writing something
    down before the attacker arrived does not protect it much.

Design
======
An adversary compromises the host at tau. It thereafter feeds the agent tasks whose topics it
chooses, and watches which partitions grow. Because partitions are contiguous file regions,
growth is visible with **no key and no cryptanalysis** -- forward privacy says nothing about
region sizes.

Epoch batching hides which of ``epoch_writes`` records in a flush was the injection. It does
not survive repetition: the injected partition appears in every flush, co-flushed organic
writes do not, so intersecting the touched-partition sets across repeated injections recovers
it. The adversary then labels each localised partition with the topic it steered there.

Finally -- and this is the payload -- the map is applied to **both cohorts**. Nothing about the
historical cohort was ever observed directly. Its exposure is pure transfer.

Two corpora, and the difference between them is a result
========================================================
Run it twice::

    python -m eval.experiments.e4_transfer --backend bge                      # D3, synthetic
    python -m eval.experiments.e4_transfer --backend bge \
        --corpus ../data/d2-beir/notes.jsonl                                  # D2, real prose

At the largest injection budget:

===================  ==========  =====================  ============================
corpus               chance      Adv_hist [95% CI]      Adv_hist/Adv_post [95% CI]
===================  ==========  =====================  ============================
D3 (synthetic)       0.100       0.876 [0.869, 0.881]   0.994 [0.985, 1.003]
D2 (BEIR, 8 domains) 0.125       0.587 [0.574, 0.600]   0.943 [0.916, 0.970]
===================  ==========  =====================  ============================

**The attack is weaker on real prose**, and that is expected: template-assembled text separates
more cleanly in embedding space than writing produced by people, so D3 flatters the attack. A
result reported only on D3 would be partly a measurement of the generator.

**The asymmetry becomes resolvable on real prose.** On D3 the ratio's interval contains 1.0 and
"historicity buys nothing" cannot be rejected. On D2 it does not: historicity buys 5.7%
[3.0%, 8.4%] -- small, real, and measurable. D2 is the paper's primary number.

Note the two corpora have different topic counts (10 and 8), so their chance floors differ.
Every advantage here is computed against the corpus's *own* floor, and ``Row.n_topics`` records
which one, because a figure script that assumes 1/10 silently mislabels every D2 point.

Mnemos (padded) measures 0.000 with a 95% upper bound of 0.000 at every setting on both
corpora -- but note that this is the *analytical model* of a padded store. For the attack driven
against the compiled engine, see :mod:`eval.experiments.e9_transfer_live`, which is what found
finding F5.

Relation to Theorem 1
=====================
``L_pre in L_post`` still holds and Delta is still positive at every non-degenerate setting, so
the ordering result stands. The *magnitude* is small, which reframes dual-epoch security: its
value is not that historical data is inherently safer, but that history independence plus
padding is what makes it so. The theorem describes a gap; the engineering creates it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval import stats as S
from eval.attacks import a4_injection as A4
from eval.lab import buckets as B
from eval.lab import corpus as C

N_BUCKETS = 256
SEEDS = range(8)
EPOCH_WRITES = (1, 4, 16, 64, 256)
BUDGETS_PER_TOPIC = (1, 5, 20, 50, 100, 250)
DEFAULT_EPOCH_WRITES = 16
DEFAULT_BUDGET_PER_TOPIC = 100


@dataclass
class Row:
    axis: str
    x: float
    padded: bool
    buckets_mapped: float
    adv_hist: float
    adv_post: float
    delta: float
    ratio: float
    """``Adv_hist / Adv_post`` -- how little historicity buys."""

    # --- 95% cluster-bootstrap intervals (eval.stats) -----------------------------------
    # Every headline number in the paper is one of these. A point estimate answers "what
    # happened when we ran it"; the interval answers "what happens if you run it again", which
    # is the question that decides whether the result is publishable.
    adv_hist_lo: float = 0.0
    adv_hist_hi: float = 0.0
    adv_post_lo: float = 0.0
    adv_post_hi: float = 0.0
    delta_lo: float = 0.0
    delta_hi: float = 0.0
    ratio_lo: float = 0.0
    ratio_hi: float = 0.0

    # --- the honest baseline ------------------------------------------------------------
    # Uniform chance (1/n_topics) is the right baseline only for a balanced corpus. A skewed
    # one lets a classifier that ignores its input entirely beat it, just by always answering
    # with the modal topic. Both baselines are carried so a reader can see the difference.
    mfc_hist: float = 0.0
    mfc_post: float = 0.0
    n_topics: int = 0
    """Carried so a results file records the baseline its own advantages were computed against.

    D3 has 10 topics and D2 has 8. A reader (or a figure script) that assumes 1/10 silently
    mislabels every D2 number."""
    adv_hist_vs_mfc: float = 0.0
    adv_hist_vs_mfc_lo: float = 0.0
    adv_hist_vs_mfc_hi: float = 0.0
    adv_post_vs_mfc: float = 0.0
    n_boot: int = 0


def _agg(results: list[A4.InjectionResult], axis: str, x: float, padded: bool) -> Row:
    """Aggregate trials into one row, with bootstrap intervals.

    The per-entry hit vectors from every trial are stacked into an ``(n_seeds, n_entries)``
    matrix, which is exactly the shape the cluster bootstrap needs: resample rows to capture
    attack randomness, columns to capture corpus sampling.
    """
    ah = float(np.mean([r.adv_hist for r in results]))
    ap = float(np.mean([r.adv_post for r in results]))

    boot = S.bootstrap_pair(
        hist_hits=np.vstack([r.hist_hits for r in results]),
        post_hits=np.vstack([r.post_hits for r in results]),
        hist_labels=results[0].hist_labels,
        post_labels=results[0].post_labels,
        n_topics=results[0].n_topics,
    )

    return Row(
        axis=axis,
        x=x,
        padded=padded,
        n_topics=results[0].n_topics,
        buckets_mapped=float(np.mean([r.buckets_mapped for r in results])),
        adv_hist=ah,
        adv_post=ap,
        delta=float(np.mean([r.delta for r in results])),
        ratio=(ah / ap) if ap > 0 else 0.0,
        adv_hist_lo=boot.adv_hist.lo,
        adv_hist_hi=boot.adv_hist.hi,
        adv_post_lo=boot.adv_post.lo,
        adv_post_hi=boot.adv_post.hi,
        delta_lo=boot.delta.lo,
        delta_hi=boot.delta.hi,
        ratio_lo=boot.ratio.lo,
        ratio_hi=boot.ratio.hi,
        mfc_hist=boot.mfc_hist,
        mfc_post=boot.mfc_post,
        adv_hist_vs_mfc=boot.adv_hist_vs_mfc.point,
        adv_hist_vs_mfc_lo=boot.adv_hist_vs_mfc.lo,
        adv_hist_vs_mfc_hi=boot.adv_hist_vs_mfc.hi,
        adv_post_vs_mfc=boot.adv_post_vs_mfc.point,
        n_boot=boot.n_boot,
    )


def main(
    out: Path | None = None, backend: str = "auto", corpus_path: Path | None = None
) -> list[Row]:
    """Run the A4 sweep.

    ``corpus_path`` selects the corpus. D3 (the default) is synthetic and
    template-assembled; D2 is the BEIR multi-domain public corpus built by
    ``data/fetch_d2.py``. Both are reported, because a structure-transfer result measured only
    on generated text is partly a measurement of the generator --- template prose separates
    more cleanly in embedding space than the real thing.
    """
    corpus = C.load(corpus_path) if corpus_path else C.load()
    n_topics = len({e.topic for e in corpus.all_entries})
    print(
        f"corpus: {corpus_path.name if corpus_path else 'D3 (default)'} --- "
        f"{len(corpus.all_entries)} entries, {n_topics} topics, chance {1 / n_topics:.3f}"
    )
    part = B.partition(corpus, n_buckets=N_BUCKETS, seed=0, backend=backend)
    print(f"embedding backend: {part.embedding_backend}")

    rows: list[Row] = []

    # Axis 1: injection budget, at the default epoch schedule.
    for per in BUDGETS_PER_TOPIC:
        for padded in (False, True):
            rs = [A4.run(corpus, part, DEFAULT_EPOCH_WRITES, per, padded, s) for s in SEEDS]
            rows.append(_agg(rs, "budget", per * 10, padded))

    # Axis 2: epoch schedule, at a fixed budget. Theorem 1's knob.
    for ew in EPOCH_WRITES:
        for padded in (False, True):
            rs = [A4.run(corpus, part, ew, DEFAULT_BUDGET_PER_TOPIC, padded, s) for s in SEEDS]
            rows.append(_agg(rs, "epoch_writes", ew, padded))

    _report(rows)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        # NaN is not valid JSON. It appears legitimately: where the attack recovers nothing,
        # Adv_post is zero in every resample and the ratio is genuinely undefined. Emitting the
        # literal `NaN` produces a file that Python reads back happily and every strict parser
        # rejects, so undefined values are written as null instead of pretending to be zero --
        # "we could not measure this" and "we measured zero" are different claims.
        out.write_text(
            json.dumps([_json_safe(asdict(r)) for r in rows], indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return rows


def _json_safe(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}


def _report(rows: list[Row]) -> None:
    for axis, label in (("budget", "injection budget"), ("epoch_writes", "epoch_writes")):
        print(f"\n=== {label} ===")
        print(
            f"{'x':>7} {'pad':>5} {'mapped':>7} "
            f"{'adv_hist [95% CI]':>26} {'adv_post':>9} {'ratio [95% CI]':>24} {'vs MFC':>8}"
        )
        for r in rows:
            if r.axis != axis:
                continue
            ah = f"{r.adv_hist:.3f} [{r.adv_hist_lo:.3f}, {r.adv_hist_hi:.3f}]"
            rt = f"{r.ratio:.3f} [{r.ratio_lo:.3f}, {r.ratio_hi:.3f}]"
            print(
                f"{r.x:7.0f} {r.padded!s:>5} {r.buckets_mapped:7.1f} "
                f"{ah:>26} {r.adv_post:9.3f} {rt:>24} {r.adv_hist_vs_mfc:8.3f}"
            )

    if rows:
        print(
            # Derived from the corpus, never hardcoded. D3 has 10 topics and D2 has 8, so a
            # fixed 1/10 silently mislabels every D2 run -- while the advantage numbers, which
            # *are* computed from the real topic count, would disagree with their own baseline.
            f"\nBaselines: uniform chance = {1.0 / max(rows[0].n_topics, 1):.3f}; "
            f"most-frequent-class = {rows[0].mfc_hist:.3f} (cohort H), "
            f"{rows[0].mfc_post:.3f} (cohort P)"
        )

    # The headline is taken from the single best-resolved setting rather than averaged across
    # the sweep. Averaging ratios over budgets where the attack barely fires mixes a
    # well-estimated number with several ill-conditioned ones and reports the blend as though
    # it were one measurement.
    best = max(
        (r for r in rows if not r.padded and r.adv_post > 0),
        key=lambda r: r.adv_post,
        default=None,
    )
    if best is not None:
        print(
            f"\nHEADLINE (at the best-resolved setting, {best.axis}={best.x:.0f}):"
            f"\n  Adv_hist            = {best.adv_hist:.3f} "
            f"[{best.adv_hist_lo:.3f}, {best.adv_hist_hi:.3f}]"
            f"\n  Adv_hist vs MFC     = {best.adv_hist_vs_mfc:.3f} "
            f"[{best.adv_hist_vs_mfc_lo:.3f}, {best.adv_hist_vs_mfc_hi:.3f}]"
            f"\n  Adv_hist / Adv_post = {best.ratio:.3f} "
            f"[{best.ratio_lo:.3f}, {best.ratio_hi:.3f}]"
            f"\n  -> historicity buys {(1 - best.ratio) * 100:.1f}% "
            f"[{(1 - best.ratio_hi) * 100:.1f}%, {(1 - best.ratio_lo) * 100:.1f}%]"
        )

    padded_max = max((r.adv_hist_hi for r in rows if r.padded), default=0.0)
    print(
        f"\nMnemos (padded) worst-case Adv_hist UPPER CI bound across all settings: "
        f"{padded_max:.3f}"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "bge", "tfidf"),
        help="embedding backend; 'bge' is the real model the paper names",
    )
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument(
        "--corpus",
        default=None,
        help="corpus jsonl; omit for D3, or pass data/d2-beir/notes.jsonl for D2",
    )
    args = ap.parse_args()

    # Separate default output per backend. Overwriting one backend's results with another's
    # would make the two indistinguishable after the fact, which is the exact failure the
    # `embedding_backend` field exists to prevent.
    suffix = "" if args.backend == "tfidf" else "_" + args.backend
    if args.corpus:
        # A separate file per corpus, for the same reason there is one per backend: a result
        # silently overwritten by a different corpus is a result nobody can trust.
        suffix += "_" + Path(args.corpus).parent.name
    default = f"python/eval/results/e4_transfer{suffix}.json"
    main(
        C.repo_root() / (args.out or default),
        backend=args.backend,
        corpus_path=Path(args.corpus) if args.corpus else None,
    )
