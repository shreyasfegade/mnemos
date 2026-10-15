"""E9 -- drive attack A4 against the **live Mnemos engine**, not a model of it.

Why this experiment exists
==========================
The paper reports ``Adv_hist = 0.000`` for Mnemos at every injection budget. That number came
from ``eval/attacks/a4_injection.py`` with ``padded=True``, whose entire content is::

    if not padded:
        ... run the attack ...
    # padded: bucket_label stays empty, so every entry is scored wrong, so the answer is 0.000

That is not a measurement. It is the *design* asserting its own success: "partitions are padded
to constant size, so no growth is observable, so there is nothing to intersect." The claim is
about what an adversary can see on disk, and nothing in that branch ever looked at a disk.

This experiment closes the gap. It builds a real store through the compiled engine, induces
real writes, captures the ``LeakEvent`` stream the engine actually emits, and runs the
intersection attack on that. If padding works, the answer is chance. If it does not, the answer
is whatever it is.

What the adversary is allowed to see
------------------------------------
Only the trace, and only from tau onward:

* ``EpochFlush`` -- that a flush happened, and its epoch. Segments the trace.
* ``IndexWrite`` -- that a record was written into some partition, and which one.

and its own ground truth: which topic it induced, and when. It never reads corpus labels, never
reads the partition assignment, never touches the engine object. The historical cohort is
written **before tracing starts**, so the adversary genuinely never observed it.

Why ``IndexWrite`` is a fair thing to give the adversary
-------------------------------------------------------
It is not a courtesy. The flush path writes each record into its partition at file offset
``HEADER_LEN + bucket * capacity * slot_len``, so the partition of every index write is
recoverable from block-level write observation -- and the tau-arrival adversary is defined as
seeing every disk delta after tau. The event is the honest model of a channel
the implementation genuinely has. Finding F5 is what happens when that channel has no event:
nothing in the test suite, the leakage table, or the attack lab could see it, and it leaked
completely for the entire life of the project.

The distinction this experiment is really about
-----------------------------------------------
Padding to capacity defeats a **snapshot** adversary: in a final disk image every partition
occupies the same bytes whatever it holds, so occupancy is not recoverable. It says nothing
about a **persistent** adversary who watches writes as they happen, because *which* constant-
size region was written is still observable. Mnemos's threat model is explicitly the second
kind. Whether the defence survives it is the question, and it has never been asked.

Usage::

    cd python && python -m eval.experiments.e9_transfer_live
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval import stats as S
from eval.lab import corpus as C

#: Entries in the historical cohort -- written before tau, never observed being written.
N_HISTORICAL = 400
#: Entries available to draw induced writes from.
N_POST = 400
#: Induced writes per topic, swept.
BUDGETS = (2, 4, 8, 16)
#: Independent trials per budget, for the confidence interval.
N_TRIALS = 5
SEED = 20_260_827
TOP_K = 10


@dataclass
class LiveRow:
    """One (budget, trial) outcome measured against the real engine."""

    budget: int
    trial: int
    seed: int
    buckets_total: int
    buckets_mapped: int
    hist_correct: int
    hist_total: int
    post_correct: int
    post_total: int
    n_topics: int
    strategy: str
    index_write_padding: bool
    mapped_intersection: int
    mapped_frequency: int


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _to_toml(d: dict) -> str:
    """Minimal TOML writer -- the stdlib reads TOML but does not write it.

    Top-level scalars are emitted *before* any table header, because a key written after one
    would be parsed as belonging to that table. Dropping them instead (an earlier version of
    this function did) loses ``schema_version`` and the loader rejects the file.
    """
    lines: list[str] = []
    for k, v in d.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {_fmt(v)}")
    for section, body in d.items():
        if not isinstance(body, dict):
            continue
        flat = {k: v for k, v in body.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in body.items() if isinstance(v, dict)}
        if flat:
            lines.append(f"\n[{section}]")
            lines.extend(f"{k} = {_fmt(v)}" for k, v in flat.items())
        for sub, subbody in nested.items():
            lines.append(f"\n[{section}.{sub}]")
            lines.extend(f"{k} = {_fmt(v)}" for k, v in subbody.items())
    return "\n".join(lines) + "\n"


def _write_params(n: int, dim: int, index_write_padding: bool = True) -> tuple[Path, int]:
    """Parameter file sized so the index cannot saturate at `n` entries."""
    import tomllib

    base = C.repo_root() / "params.toml"
    with base.open("rb") as fh:
        p = tomllib.load(fh)

    buckets = max(16, 1 << int(np.ceil(np.log2(max(4.0, np.sqrt(n))))))
    capacity = max(32, 1 << int(np.ceil(np.log2(max(2.0, (n / buckets) * 2)))))
    n_max = max(1024, 1 << int(np.ceil(np.log2(n / 0.6 * 1.5))))

    assert buckets * capacity >= n, "index geometry cannot hold the corpus"
    assert n <= n_max * p["epoch"]["max_load_factor"], "corpus exceeds the safe load factor"

    p["embedding"]["dim"] = dim
    p["index"]["buckets"] = buckets
    p["index"]["bucket_capacity"] = capacity
    p["index"]["n_max"] = n_max
    p["index"]["top_k"] = TOP_K
    p["index"]["index_write_padding"] = index_write_padding
    # Pinned, not inherited. `params.toml` ships the query-equality mitigation on (`s = 2`),
    # which multiplies read volume; this experiment measures the **write** path, so the setting
    # cannot change its result -- but an experiment whose configuration moves when an unrelated
    # default moves is not reproducible, and "it happens not to matter" is a claim that has to
    # be re-derived every time someone touches params.toml.
    p["index"]["probe_spread"] = 1
    p["crypto"]["block_size_bytes"] = 1024
    p["crypto"]["max_entry_bytes"] = 1024
    p["memory"]["arena_bytes"] = 262_144

    tag = "padded" if index_write_padding else "f5"
    out = Path(tempfile.gettempdir()) / f"mnemos-e9-{tag}.toml"
    out.write_text(_to_toml(p), encoding="utf-8")
    return out, buckets


def read_flushes(trace: Path) -> list[list[int]]:
    """Segment the trace into flushes, each a list of partitions written.

    This is the adversary's whole view of the write path: ``index_write`` events accumulate
    until an ``epoch_flush`` closes the batch.

    The trace is one flat JSON object per line with a ``kind`` discriminator, e.g.
    ``{"kind":"index_write","bucket":24,"epoch":4}``. An earlier version of this function
    assumed an externally-tagged shape (``{"IndexWrite": {...}}``), matched nothing, and
    returned an empty list for every trial --- which the experiment then reported as
    ``Adv_hist = 0.000``, i.e. as a successful defence. A parser that silently matches nothing
    is indistinguishable from a store that leaks nothing, so the caller asserts that flushes
    were actually found.
    """
    flushes: list[list[int]] = []
    pending: list[int] = []
    with trace.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("kind")
            if kind == "index_write":
                pending.append(int(ev["bucket"]))
            elif kind == "epoch_flush":
                flushes.append(pending)
                pending = []
    if pending:
        flushes.append(pending)
    return flushes


def attack_intersection(by_topic: dict[str, list[list[int]]]) -> dict[int, str]:
    """Strategy 1: intersect the flushes carrying each topic's induced writes.

    This is the strategy the analytical model uses, and against a real store it is **the wrong
    one** -- which is worth keeping rather than quietly replacing, because the difference is
    instructive.

    The model groups injections by the partition they landed in, using ground truth it would
    not have. A real adversary can only group by the thing it controls: the topic. And a topic
    does not occupy one partition --- with `B` partitions and `T` topics each topic spans
    roughly `B/T` of them, so the induced writes for a topic land in *different* partitions and
    the intersection across its flushes is empty.

    Reported for contrast. An experiment that ran only this would conclude `Adv = 0.000` and
    congratulate the defence for surviving an attack that was never aimed at it.
    """
    labels: dict[int, str] = {}
    for topic, flushes in by_topic.items():
        if not flushes:
            continue
        surviving = set(flushes[0])
        for f in flushes[1:]:
            surviving &= set(f)
        if len(surviving) == 1:
            labels[surviving.pop()] = topic
    return labels


def attack_frequency(by_topic: dict[str, list[list[int]]], n_buckets: int) -> dict[int, str]:
    """Strategy 2: frequency analysis over which partitions each topic's flushes touch.

    Every flush is `1` real record plus `epoch_writes - 1` dummies in uniformly random
    partitions. So for topic `t` and partition `b`:

        P(b appears in a flush of t) = [b holds t] * 1 + uniform dummy baseline

    The dummy baseline is identical for every partition and every topic, so it cancels when
    partitions are compared *across* topics. A partition holding topic `t` is touched by `t`'s
    flushes at a measurably higher rate than by any other topic's, and that difference is the
    whole signal --- it survives dummy padding because padding adds a constant, and a constant
    does not hide a difference.

    Each partition is assigned the topic whose flushes touch it most, provided that topic's
    rate exceeds the mean rate across topics by a margin. Partitions that clear no margin are
    left unmapped rather than guessed, on the same principle as the intersection strategy: a
    real adversary would induce more writes, and guessing inflates the result.
    """
    topics = [t for t, f in by_topic.items() if f]
    if not topics:
        return {}

    rate: dict[str, np.ndarray] = {}
    for t in topics:
        counts = np.zeros(n_buckets, dtype=np.float64)
        for f in by_topic[t]:
            for b in set(f):
                if 0 <= b < n_buckets:
                    counts[b] += 1.0
        rate[t] = counts / len(by_topic[t])

    stacked = np.vstack([rate[t] for t in topics])
    mean_rate = stacked.mean(axis=0)
    # One real write per flush across `n` flushes lifts its partition's rate by ~1/(spread of
    # the topic). The margin only has to exceed dummy sampling noise, which for `n` flushes and
    # a per-flush dummy rate `q` has standard deviation sqrt(q(1-q)/n).
    n_min = min(len(by_topic[t]) for t in topics)
    q = float(mean_rate.mean())
    noise = float(np.sqrt(max(q * (1.0 - q), 1e-9) / max(n_min, 1)))
    margin = 2.0 * noise

    labels: dict[int, str] = {}
    for b in range(n_buckets):
        col = stacked[:, b]
        best = int(np.argmax(col))
        if col[best] - mean_rate[b] > margin:
            labels[b] = topics[best]
    return labels


@dataclass
class Setup:
    """Everything expensive, computed once and shared by every trial.

    Embedding 800 passages with `bge-base-en-v1.5` on CPU takes ~55 s. Doing it inside each
    trial made a 20-trial sweep spend twenty minutes re-deriving an identical array. It also
    made the trials differ in their partition table, which would have confounded attack
    variance with clustering variance.
    """

    hist: list
    post: list
    topics: list[str]
    vectors: np.ndarray
    centroid_list: list
    hist_bucket: np.ndarray
    post_bucket: np.ndarray
    params_padded: Path
    params_f5: Path
    buckets: int


def prepare(seed: int) -> Setup:
    """Embed the corpus, train one partition table, and record ground-truth assignments."""
    import mnemos_native

    from mnemos.embed import embed, require_semantic_backend

    corpus = C.load()
    hist = corpus.historical[:N_HISTORICAL]
    post = corpus.post[:N_POST]
    topics = sorted({e.topic for e in corpus.all_entries})

    texts = [e.text for e in hist + post]
    vectors = embed(texts)
    # Checked *after* embedding: the backend resolves lazily on first use. A4 is a security
    # metric and would survive the hashing fallback, but the partition structure it attacks is
    # only realistic under a real embedder.
    require_semantic_backend()
    dim = int(vectors.shape[1])

    # Sizing has to count *records written*, not entries induced. Every forced flush writes a
    # full padded batch of `epoch_writes` records -- that is the padding working -- so inducing
    # one write costs `epoch_writes` slab slots, not one. At the largest budget that is
    # 16 x 10 topics x 16 = 2,560 slots on top of the historical cohort, and sizing for 800
    # made the engine correctly refuse to flush partway through the sweep.
    import tomllib

    with (C.repo_root() / "params.toml").open("rb") as fh:
        epoch_writes = int(tomllib.load(fh)["epoch"]["epoch_writes"])
    worst_case = N_HISTORICAL + max(BUDGETS) * len(topics) * epoch_writes
    params_padded, buckets = _write_params(worst_case, dim, index_write_padding=True)
    params_f5, _ = _write_params(worst_case, dim, index_write_padding=False)
    # (sample, buckets, dim, seed) -- `dim` is third. Passing the seed there instead asks for
    # 20-million-dimensional centroids, which the FFI now rejects rather than trying to allocate.
    centroid_list = mnemos_native.train_centroids(
        [v.tolist() for v in vectors], buckets, dim, int(seed)
    )
    centroids = np.asarray(centroid_list, dtype=np.float64)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    # Ground truth, held by the harness only. Used to score the attack, never shown to it.
    assign = np.argmax(vectors @ centroids.T, axis=1)

    return Setup(
        hist=hist,
        post=post,
        topics=topics,
        vectors=vectors,
        centroid_list=centroid_list,
        hist_bucket=assign[: len(hist)],
        post_bucket=assign[len(hist) :],
        params_padded=params_padded,
        params_f5=params_f5,
        buckets=buckets,
    )


def run_trial(
    setup: Setup, budget: int, trial: int, seed: int, index_write_padding: bool = True
) -> LiveRow:
    """One end-to-end trial against a real store."""
    from mnemos.mcp_server import MnemosMemoryServer

    os.environ.setdefault("MNEMOS_I_ACCEPT_INSECURE", "1")
    rng = np.random.default_rng(seed)

    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        server = MnemosMemoryServer(
            scratch / "store",
            params_path=setup.params_padded if index_write_padding else setup.params_f5,
            allow_insecure_keystore=True,
        )
        server.open(create_if_missing=True, centroids=setup.centroid_list)

        # --- before tau: the historical cohort, written unobserved ---------------------
        for e in setup.hist:
            server.put(e.text[:400], {"topic": e.topic})
        server._require().flush()

        # --- tau: the adversary arrives ------------------------------------------------
        trace = scratch / "trace.jsonl"
        server.record_trace(trace)

        by_topic: dict[str, list[list[int]]] = {t: [] for t in setup.topics}
        post_by_topic: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(setup.post):
            post_by_topic[e.topic].append(i)

        # Each induced write is issued and then flushed on its own, so the adversary knows
        # which flush carries it. That is the strongest realistic attribution: it controls the
        # stimulus, so it controls the timing.
        flush_topics: list[str] = []
        for topic in setup.topics:
            pool = post_by_topic.get(topic, [])
            if not pool:
                continue
            picks = rng.choice(len(pool), size=min(budget, len(pool)), replace=False)
            for j in picks:
                e = setup.post[pool[int(j)]]
                server.put(e.text[:400], {"topic": e.topic})
                server._require().flush()
                flush_topics.append(topic)

        server.close()

        flushes = [f for f in read_flushes(trace) if f]
        assert flushes, (
            f"no index_write events parsed from {trace}; the attack would report a perfect "
            "defence for what is actually a broken parser"
        )

    for topic, f in zip(flush_topics, flushes, strict=False):
        by_topic[topic].append(f)

    # A real adversary runs both and keeps whichever worked. Reporting only the weaker one
    # would be the same error finding F5 was: mistaking "our attack found nothing" for
    # "there is nothing to find".
    labels_i = attack_intersection(by_topic)
    labels_f = attack_frequency(by_topic, setup.buckets)

    def score(labels: dict[int, str]) -> int:
        return sum(
            1 for i, e in enumerate(setup.hist) if labels.get(int(setup.hist_bucket[i])) == e.topic
        )

    labels = labels_f if score(labels_f) >= score(labels_i) else labels_i
    strategy = "frequency" if labels is labels_f else "intersection"

    hist_correct = sum(
        1 for i, e in enumerate(setup.hist) if labels.get(int(setup.hist_bucket[i])) == e.topic
    )
    post_correct = sum(
        1 for i, e in enumerate(setup.post) if labels.get(int(setup.post_bucket[i])) == e.topic
    )

    return LiveRow(
        budget=budget,
        trial=trial,
        seed=seed,
        buckets_total=setup.buckets,
        buckets_mapped=len(labels),
        hist_correct=hist_correct,
        hist_total=len(setup.hist),
        post_correct=post_correct,
        post_total=len(setup.post),
        n_topics=len(setup.topics),
        strategy=strategy,
        index_write_padding=index_write_padding,
        mapped_intersection=len(labels_i),
        mapped_frequency=len(labels_f),
    )


def main() -> None:
    setup = prepare(SEED)
    print(
        f"corpus {len(setup.hist)}+{len(setup.post)} entries, {len(setup.topics)} topics, "
        f"{setup.buckets} partitions",
        flush=True,
    )
    chance = 1.0 / max(len(setup.topics), 1)
    print(f"chance floor {chance:.3f}; adv = max(0, accuracy - chance)\n", flush=True)

    rows: list[LiveRow] = []
    # Both arms of the ablation. `padding=False` reproduces finding F5, so the two columns of
    # the output table are "what the attack achieves against the shipped engine" and "what it
    # achieved against the engine as it was for most of this project's life".
    for padding in (True, False):
        label = "padded" if padding else "F5 (no index padding)"
        print(f"--- index_write_padding = {padding}  [{label}] ---", flush=True)
        for budget in BUDGETS:
            for t in range(N_TRIALS):
                row = run_trial(
                    setup, budget, t, SEED + 1000 * budget + t, index_write_padding=padding
                )
                rows.append(row)
                acc = row.hist_correct / row.hist_total
                print(
                    f"  budget={budget:<3} trial={t}  "
                    f"mapped i={row.mapped_intersection:>3} f={row.mapped_frequency:>3}"
                    f"/{row.buckets_total}  [{row.strategy:>12}]  "
                    f"acc_hist={acc:.3f}  adv_hist={max(0.0, acc - chance):.3f}",
                    flush=True,
                )

    out = C.repo_root() / "python" / "eval" / "results" / "e9_transfer_live.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "experiment": "E9",
                "description": "A4 structure transfer against the live Mnemos engine",
                "observables": ["epoch_flush", "index_write"],
                "n_historical": N_HISTORICAL,
                "n_post": N_POST,
                "chance": chance,
                "rows": [asdict(r) for r in rows],
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    print("\n=== E9: A4 against the LIVE engine ===\n")
    n_topics = rows[0].n_topics if rows else 0
    print(f"  topics {n_topics}, chance floor {chance:.3f}\n")
    header = (
        f"  {'padding':>8}  {'budget':>6}  {'mapped':>7}  "
        f"{'acc_hist':>9}  {'adv_hist [95% CI]':>24}"
    )
    print(header)
    for padding in (True, False):
        for budget in BUDGETS:
            group = [r for r in rows if r.budget == budget and r.index_write_padding == padding]
            if not group:
                continue
            hist_hits = np.vstack(
                [
                    np.concatenate(
                        [
                            np.ones(r.hist_correct, bool),
                            np.zeros(r.hist_total - r.hist_correct, bool),
                        ]
                    )
                    for r in group
                ]
            )
            post_hits = np.vstack(
                [
                    np.concatenate(
                        [
                            np.ones(r.post_correct, bool),
                            np.zeros(r.post_total - r.post_correct, bool),
                        ]
                    )
                    for r in group
                ]
            )
            labels_h = np.zeros(hist_hits.shape[1], dtype=np.int64)
            labels_p = np.zeros(post_hits.shape[1], dtype=np.int64)
            pair = S.bootstrap_pair(hist_hits, post_hits, labels_h, labels_p, n_topics)
            mapped = np.mean([r.buckets_mapped for r in group])
            acc = float(hist_hits.mean())
            print(
                f"  {padding!s:>8}  {budget:>6}  {mapped:>5.1f}/{group[0].buckets_total:<3}"
                f"  {acc:>9.3f}"
                f"  {pair.adv_hist.point:>7.3f} [{pair.adv_hist.lo:.3f}, {pair.adv_hist.hi:.3f}]"
            )
    print(f"\nwrote {out.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
