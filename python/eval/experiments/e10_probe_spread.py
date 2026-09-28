"""E10 -- the query-equality mitigation, and what it costs.

A5 established that query equality is recoverable at 1.000 against Mnemos as shipped, intra-
and inter-epoch, because the probe set is a pure function of the query vector. The paper
reported that as a limitation and stopped there, which leaves the obvious question unasked:
**can it be closed, and what does closing it cost?**

This experiment answers both by sweeping the two knobs added to `EncryptedIvf::probe_set` and
measuring, on a real store, what each setting buys and what it charges.

The two knobs, and why one is not enough
========================================
``probe_spread`` (`s`)
    Read `s * k_probe` partitions, of which `k_probe` are the genuine top-`k` and the rest are
    drawn with *fresh* randomness on every call. Two searches for the same query stop being
    identical, so pairwise signature matching collapses.

    On its own this is not sufficient. The real probes are present in *every* observation of a
    query, so an adversary able to group repetitions by some other means intersects them and
    the real probes fall out.

``probe_retention`` (`p`)
    Read each genuine probe only with probability `p`, substituting a random partition when it
    is dropped. A real probe now survives `r` intersected repetitions with probability `p^r`,
    so the intersection attack degrades geometrically too.

What each costs
---------------
Spread costs **read volume**, linearly and exactly: `s x` the partitions per search. Retention
costs **recall**, because a dropped probe is a partition genuinely not searched. Neither costs
correctness in the sense of returning wrong answers -- results are still ranked by true
similarity; retention only means a relevant entry may not be among the candidates considered.

Both are reported here rather than argued, which is the point: "randomise the probe set" is a
sentence anyone can write in a limitations section, and it is worth very little until someone
says what it does to Recall@10.

Usage::

    cd python && python -m eval.experiments.e10_probe_spread
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval.attacks import a5_volumetric as A5
from eval.lab import corpus as C

#: Distinct queries in the leakage workload.
N_QUERIES = 12
#: Searches issued, sampled from those queries with repeats.
N_SEARCHES = 60
#: Corpus size for the leakage measurement. The channel is structural; it does not need scale.
N_ENTRIES = 240
#: Corpus size for the utility measurement, large enough for Recall@10 to be meaningful.
N_UTILITY = 400
TOP_K = 10
SEED = 20_260_827

#: The design space. `(spread, retention)`; `(1, 1.0)` is Mnemos as shipped and as measured by
#: A5, carried through so the table has its own control.
GRID: tuple[tuple[int, float], ...] = (
    (1, 1.0),
    (2, 1.0),
    (4, 1.0),
    (8, 1.0),
    (4, 0.9),
    (4, 0.75),
    (4, 0.5),
)

#: Partitions in the leakage store, fixed so `s = 8` does not degenerate into reading every
#: partition. At `B = 32` and `k_probe = 4`, `s = 8` reads the whole index -- which is genuinely
#: oblivious and therefore leaks nothing, but makes the sweep's top row a special case rather
#: than a point on a curve. Fixed at 64 so every setting reads a strict subset.
LEAK_BUCKETS = 64


@dataclass
class SpreadRow:
    """One point in the mitigation design space, measured on a real store."""

    spread: int
    retention: float
    query_equality_intra: float
    query_equality_inter: float
    false_positive_rate: float
    advantage: float
    """Youden's J: true-positive rate minus false-positive rate.

    Recovery alone is recall, and recall alone is gameable: at `s = 8` the probe set is every
    partition in the store, so every pair of searches looks identical, an adversary who always
    answers "same query" scores recovery 1.000 -- and learns nothing whatsoever. J is 0 for
    that adversary and 1 for a perfect fingerprint, so it is the number the figure plots.
    """
    intersection_recovery: float
    volume_variance: float
    bucket_reads_per_search: int
    recall_at_10: float
    recall_delta_pp: float
    partitions_read_multiple: float


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _to_toml(d: dict) -> str:
    """Minimal TOML writer. Top-level scalars first -- see e9's note on why."""
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


def _write_params(
    n: int, dim: int, spread: int, retention: float, tag: str, force_buckets: int = 0
) -> Path:
    """Parameter file for one point in the sweep, sized so the index cannot saturate."""
    import tomllib

    base = C.repo_root() / "params.toml"
    with base.open("rb") as fh:
        p = tomllib.load(fh)

    buckets = max(16, 1 << int(np.ceil(np.log2(max(4.0, np.sqrt(n))))))
    if force_buckets:
        buckets = force_buckets
    capacity = max(32, 1 << int(np.ceil(np.log2(max(2.0, (n / buckets) * 2)))))
    # Sized against **padded** occupancy, not the number of real records.
    #
    # Every flush writes `slot_write_budget` slots for `epoch_writes` real records -- a 4x write
    # amplification, deliberate, and documented in the paper as the price of a data-independent
    # per-flush write count. A store sized as `n / max_load_factor` therefore runs out at a
    # quarter of the corpus it was sized for: 400 entries filled 1,216 of the 1,228 slots that
    # `n_max = 2048` allows, and the next flush was refused.
    #
    # This is the saturated-index bug (`crates/mnemos-bench/benches/search.rs`) one layer up,
    # and it fails in the *safe* direction only because `Engine::flush` refuses rather than
    # silently dropping. The 1.5 factor on top is headroom for compaction and rounding.
    amplification = p["epoch"]["slot_write_budget"] / p["epoch"]["epoch_writes"]
    needed = n * amplification / p["epoch"]["max_load_factor"] * 1.5
    n_max = max(1024, 1 << int(np.ceil(np.log2(needed))))
    assert buckets * capacity >= n, "index geometry cannot hold the corpus"
    assert p["index"]["probe_width"] * spread <= buckets, (
        f"probe_width * spread = {p['index']['probe_width'] * spread} exceeds {buckets} "
        "partitions; the search would read the whole index"
    )

    p["embedding"]["dim"] = dim
    p["index"]["buckets"] = buckets
    p["index"]["bucket_capacity"] = capacity
    p["index"]["n_max"] = n_max
    p["index"]["top_k"] = TOP_K
    p["index"]["decoy_buckets"] = 0
    p["index"]["probe_spread"] = spread
    p["index"]["probe_retention"] = retention
    p["crypto"]["block_size_bytes"] = 1024
    p["crypto"]["max_entry_bytes"] = 1024
    p["memory"]["arena_bytes"] = 262_144

    out = Path(tempfile.gettempdir()) / f"mnemos-e10-{tag}.toml"
    out.write_text(_to_toml(p), encoding="utf-8")
    return out


def _intersection_recovery(trace: Path, truth: list[int]) -> float:
    """Can an adversary who already knows which searches repeat still find the real probes?

    This is the *stronger* attack that spread alone does not stop. It hands the adversary the
    grouping for free -- it is told which searches were the same query -- and asks whether
    intersecting those read sets isolates a small candidate set.

    Scored as the fraction of query groups whose intersection is small enough to be actionable
    (at most `k_probe` partitions). Handing over the grouping is generous, and deliberately so:
    a mitigation that only works because the adversary cannot group repetitions is relying on
    the wrong thing.
    """
    searches = A5.split_searches(A5.load_events(trace))
    if not searches:
        return 0.0
    by_query: dict[int, list[set]] = {}
    for q, s in zip(truth, searches, strict=False):
        by_query.setdefault(q, []).append(set(s.get("buckets", [])))

    actionable = 0
    total = 0
    for sets in by_query.values():
        if len(sets) < 2:
            continue
        total += 1
        inter = set(sets[0])
        for nxt in sets[1:]:
            inter &= nxt
        if 0 < len(inter) <= 4:
            actionable += 1
    return actionable / total if total else 0.0


def measure_leakage(
    spread: int, retention: float, seed: int
) -> tuple[A5.VolumetricResult, float, float]:
    """Drive a real store and attack the trace it emits."""
    from mnemos.mcp_server import MnemosMemoryServer

    os.environ.setdefault("MNEMOS_I_ACCEPT_INSECURE", "1")
    rng = random.Random(seed)
    corpus = C.load()
    entries = corpus.all_entries[:N_ENTRIES]
    params = _write_params(
        N_ENTRIES * 2,
        768,
        spread,
        retention,
        f"leak-{spread}-{retention}",
        force_buckets=LEAK_BUCKETS,
    )

    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        trace = scratch / "trace.jsonl"
        server = MnemosMemoryServer(
            scratch / "store", params_path=params, allow_insecure_keystore=True
        )
        server.open(create_if_missing=True)
        for e in entries:
            server.put(e.text[:200], {"topic": e.topic})

        queries = [entries[rng.randrange(len(entries))].text[:120] for _ in range(N_QUERIES)]
        server.record_trace(trace)

        truth: list[int] = []
        boundaries: list[int] = []
        for i in range(N_SEARCHES):
            q = rng.randrange(N_QUERIES)
            server.search(queries[q])
            truth.append(q)
            if i == N_SEARCHES // 3:
                server.put("a write to force an epoch boundary")
                boundaries.append(len(truth))
        server.close()

        result = A5.analyse(trace, truth, boundaries, seed=seed, epoch_ms=500)
        inter = _intersection_recovery(trace, truth)
        fpr = A5.false_positive_rate(trace, truth)
    return result, inter, fpr


def measure_recall(spread: int, retention: float, seed: int) -> float:
    """Recall@10 against exact nearest neighbours, on a real store at this setting."""
    from mnemos.embed import embed, require_semantic_backend
    from mnemos.mcp_server import MnemosMemoryServer

    os.environ.setdefault("MNEMOS_I_ACCEPT_INSECURE", "1")
    corpus = C.load()
    entries = corpus.all_entries[:N_UTILITY]
    texts = [e.text[:400] for e in entries]
    vectors = embed(texts)
    require_semantic_backend()
    dim = int(vectors.shape[1])

    rng = np.random.default_rng(seed)
    q_idx = rng.choice(len(entries), size=100, replace=False)
    sims = vectors @ vectors.T
    ideal = {int(q): set(np.argsort(-sims[q])[:TOP_K].tolist()) for q in q_idx}

    params = _write_params(N_UTILITY * 2, dim, spread, retention, f"util-{spread}-{retention}")
    with tempfile.TemporaryDirectory() as tmp:
        server = MnemosMemoryServer(
            Path(tmp) / "store", params_path=params, allow_insecure_keystore=True
        )
        server.open(create_if_missing=True)
        ids = [server.put(t, {"i": i}) for i, t in enumerate(texts)]
        assert len(ids) == len(texts)

        hits = 0
        total = 0
        for q in q_idx:
            got = server.search(texts[int(q)], top_k=TOP_K)
            returned = {int(h["meta"].get("i", -1)) for h in got}
            hits += len(returned & ideal[int(q)])
            total += TOP_K
        server.close()
    return hits / total if total else 0.0


def main() -> None:
    rows: list[SpreadRow] = []
    baseline_recall: float | None = None

    for spread, retention in GRID:
        leak, inter, fpr = measure_leakage(spread, retention, SEED)
        recall = measure_recall(spread, retention, SEED)
        if baseline_recall is None:
            baseline_recall = recall
        rows.append(
            SpreadRow(
                spread=spread,
                retention=retention,
                query_equality_intra=leak.query_equality_intra_epoch,
                query_equality_inter=leak.query_equality_inter_epoch,
                false_positive_rate=fpr,
                advantage=max(0.0, leak.query_equality_intra_epoch - fpr),
                intersection_recovery=inter,
                volume_variance=leak.volume_variance,
                bucket_reads_per_search=leak.bucket_reads_per_search,
                recall_at_10=recall,
                recall_delta_pp=100.0 * (recall - baseline_recall),
                partitions_read_multiple=float(spread),
            )
        )
        print(
            f"  s={spread} p={retention:<5} "
            f"eq_intra={leak.query_equality_intra_epoch:.3f} "
            f"eq_inter={leak.query_equality_inter_epoch:.3f} "
            f"fpr={fpr:.3f} J={max(0.0, leak.query_equality_intra_epoch - fpr):.3f} "
            f"inter={inter:.3f} reads={leak.bucket_reads_per_search} "
            f"recall={recall:.3f}",
            flush=True,
        )

    out = C.repo_root() / "python" / "eval" / "results" / "e10_probe_spread.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "experiment": "E10",
                "description": "query-equality mitigation: recovery and cost vs probe spread",
                "n_queries": N_QUERIES,
                "n_searches": N_SEARCHES,
                "rows": [asdict(r) for r in rows],
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    print("\n=== E10: query-equality mitigation ===\n")
    print(
        f"  {'s':>2} {'p':>5}  {'eq intra':>9} {'FPR':>6} {'J':>6} {'intersect':>10}"
        f"  {'reads':>6}  {'Recall@10':>10}"
    )
    for r in rows:
        print(
            f"  {r.spread:>2} {r.retention:>5}  {r.query_equality_intra:>9.3f} "
            f"{r.false_positive_rate:>6.3f} {r.advantage:>6.3f} "
            f"{r.intersection_recovery:>10.3f}"
            f"  {r.bucket_reads_per_search:>6}  {r.recall_at_10:>10.3f}"
        )
    print(f"\nwrote {out.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
