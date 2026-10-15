"""E10: query-equality leakage under smarter adversaries, and bucketed probes as the alternative.

With probe_spread s = 2 the shipped v1 config reports pairwise recovery 0.000,
but that pairwise adversary tests *exact* equality of read sets, which is a strawman. Two searches
for the same query share all k genuine probes; two unrelated queries share about one partition by
chance. An adversary that thresholds the overlap, or clusters the overlap graph, should recover
query groups anyway.

This experiment runs, against the live v2 engine, four adversaries on a stream of repeated queries:

- ``exact``           equality of the read set (the old strawman).
- ``overlap_thresh``  "equal" iff the read sets share >= tau partitions, tau swept.
- ``graph_cluster``   connected components of the graph joining searches with overlap >= tau.

for two probe modes:

- ``spread``  s genuine-plus-random superset (v1's mitigation, ported to v2).
- ``grouped`` fixed disjoint groups; a search reads exactly one whole group (the
  bucketed-probes alternative, whose equality leaks only at group granularity but with a
  deterministic, provable anonymity set).

Metrics: pairwise TPR - FPR (Youden's J) for the pairwise adversaries, and adjusted Rand index
against the true query grouping for the clustering adversary.

    cd python && python -m eval.experiments.e10_query_equality
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from eval.lab import corpus as C
from eval.lab.embcache import embed_cached
from eval.lab.stores import create_v2

OUT = C.repo_root() / "python" / "eval" / "results" / "e10_query_equality.json"
B = 64
Cap = 16
DIM = 768
N_QUERIES = 12
REPEATS = 20
K = 4


def read_regions_of_search(trace_lines: list[dict]) -> list[int]:
    return [int(e["region"]) for e in trace_lines if e["kind"] == "region_read"]


def collect_searches(
    eng, trace_path: Path, queries: np.ndarray, repeats: int, rng
) -> tuple[list[set[int]], np.ndarray]:
    """Return one read-set per search and the true query id behind each, in random order."""
    order = [(q, r) for q in range(len(queries)) for r in range(repeats)]
    rng.shuffle(order)
    read_sets: list[set[int]] = []
    truth: list[int] = []
    for q, _ in order:
        before = trace_path.stat().st_size if trace_path.exists() else 0
        eng.search(queries[q].tolist())
        lines = [json.loads(ln) for ln in _tail(trace_path, before)]
        read_sets.append(set(read_regions_of_search(lines)))
        truth.append(q)
    return read_sets, np.array(truth)


def _tail(path: Path, offset: int) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        fh.seek(offset)
        return [ln for ln in fh if ln.strip()]


def pairwise_scores(read_sets: list[set[int]], truth: np.ndarray, predicate) -> dict:
    n = len(read_sets)
    tp = fp = tn = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same = truth[i] == truth[j]
            pred = predicate(read_sets[i], read_sets[j])
            tp += same and pred
            fp += (not same) and pred
            tn += (not same) and (not pred)
            fn += same and (not pred)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tpr": tpr, "fpr": fpr, "youden_j": tpr - fpr}


def graph_cluster(read_sets: list[set[int]], tau: int) -> np.ndarray:
    n = len(read_sets)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if len(read_sets[i] & read_sets[j]) >= tau:
                parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


def run_mode(mode: str, queries: np.ndarray, cents, rng) -> dict:
    tmp = Path(tempfile.mkdtemp())
    over = {
        "embedding": {"dim": DIM},
        "index": {
            "buckets": B,
            "bucket_capacity": Cap,
            "top_k": 10,
            "probe_width": K,
            "probe_spread": 2 if mode == "spread" else 1,
            "probe_retention": 1.0,
        },
        "crypto": {"block_size_bytes": 1024, "max_entry_bytes": 1024},
        "epoch": {"epoch_writes": 16, "epoch_ms": 3_600_000},
        "memory": {"arena_bytes": 1 << 20},
        "v2": {
            "durable": False,
            "probe_mode": "grouped" if mode == "grouped" else "spread",
            "group_size": 8,
            "flush_policy": "on_demand",
        },
    }
    eng, h, _ = create_v2(tmp / "s", cents, over)
    eng.set_trace_sink(str(tmp / "t.jsonl"))
    # Seed the store so regions are non-empty (does not affect which regions a search reads).
    for i in range(200):
        eng.put(f"seed {i}", cents[i % B].tolist(), "{}")
    eng.flush()
    read_sets, truth = collect_searches(eng, tmp / "t.jsonl", queries, REPEATS, rng)
    eng.close()

    out = {
        "mode": mode,
        "params_hash": h,
        "regions_per_search": len(read_sets[0]) if read_sets else 0,
    }
    out["exact"] = pairwise_scores(read_sets, truth, lambda a, b: a == b)
    best_j = None
    for tau in range(1, K * 2 + 1):
        s = pairwise_scores(read_sets, truth, lambda a, b, t=tau: len(a & b) >= t)
        s["tau"] = tau
        if best_j is None or s["youden_j"] > best_j["youden_j"]:
            best_j = s
        ari = adjusted_rand_score(truth, graph_cluster(read_sets, tau))
        s["cluster_ari"] = ari
    out["overlap_threshold_best"] = best_j
    out["cluster_ari_by_tau"] = {
        tau: adjusted_rand_score(truth, graph_cluster(read_sets, tau))
        for tau in range(1, K * 2 + 1)
    }
    out["cluster_ari_best"] = max(out["cluster_ari_by_tau"].values())
    return out


def main() -> None:
    rows = [
        json.loads(ln)
        for ln in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if ln.strip()
    ][:2000]
    vecs = embed_cached([r["text"] for r in rows])
    km = KMeans(n_clusters=B, random_state=0, n_init=2).fit(vecs)
    cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    rng = np.random.default_rng(20260915)
    qidx = rng.choice(len(vecs), size=N_QUERIES, replace=False)
    queries = vecs[qidx]

    results = {
        "experiment": "E10",
        "buckets": B,
        "probe_width": K,
        "n_queries": N_QUERIES,
        "repeats": REPEATS,
        "modes": [],
    }
    for mode in ("spread", "grouped"):
        r = run_mode(mode, queries, cents, np.random.default_rng(1))
        results["modes"].append(r)
        print(
            f"[{mode}] regions/search={r['regions_per_search']} "
            f"exact J={r['exact']['youden_j']:.3f} "
            f"overlap-thresh best J={r['overlap_threshold_best']['youden_j']:.3f} "
            f"(tau={r['overlap_threshold_best']['tau']}) "
            f"cluster ARI best={r['cluster_ari_best']:.3f}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
