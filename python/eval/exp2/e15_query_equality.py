"""E15: query equality under an overlap-threshold adversary, with an interval.

The paper's query-equality numbers came from an earlier harness (``eval/experiments/
e10_query_equality.py``): one store, 12 queries x 20 repeats, and the adversary's threshold chosen
on the same pairs it was then scored on. That is a single draw with an in-sample threshold, so it
has no interval and an optimistic point estimate. This version fixes both.

* **Independent repeats.** Each repeat builds a fresh store (fresh keys, so a fresh permutation
  and fresh probe randomness) and draws fresh queries. The interval is over repeats.
* **Held-out threshold.** Within a repeat the queries are split in half. The adversary picks the
  overlap threshold tau that maximises Youden's J on pairs among the *training* queries, then is
  scored on pairs among the *held-out* queries. A tau fixed in advance is a legitimate adversary;
  one fitted on the answers is not.

Four adversaries on the stream of read sets, as before: exact equality of read sets, overlap
threshold, and connected components of the overlap graph (adjusted Rand index against the true
query grouping, tau again fixed on the training half). Probe mechanisms: the randomised superset
at s = 1, 2 (shipped) and 4, and grouped probes.

This is a read-observer (RW) channel. A W adversary -- sync, backup, imaging -- sees no searches.

    cd python && python -m eval.exp2.e15_query_equality
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e15_query_equality.json"

B = 64
CAP = 16
DIM = 768
BLOCK = 256
KPROBE = 4
N_QUERIES = 16  # per repeat; half train the threshold, half are scored
REPEATS_PER_QUERY = 10
TAUS = tuple(range(1, 2 * KPROBE * 2 + 1))

MODES = {
    "spread_s1": {"probe_spread": 1, "grouped": False},
    "spread_s2": {"probe_spread": 2, "grouped": False},
    "spread_s4": {"probe_spread": 4, "grouped": False},
    "grouped": {"probe_spread": 1, "grouped": True},
}


def read_sets_for(eng, trace: Path, queries: np.ndarray, order: list[int]) -> list[frozenset[int]]:
    out: list[frozenset[int]] = []
    offset = trace.stat().st_size if trace.exists() else 0
    for q in order:
        eng.search(queries[q].tolist())
        with trace.open("r", encoding="utf-8") as fh:
            fh.seek(offset)
            lines = [line for line in fh if line.strip()]
        offset = trace.stat().st_size
        regs = set()
        for line in lines:
            e = json.loads(line)
            if e["kind"] == "region_read":
                regs.add(int(e["region"]))
        out.append(frozenset(regs))
    return out


def youden(sets: list[frozenset[int]], truth: np.ndarray, pred) -> float:
    tp = fp = tn = fn = 0
    n = len(sets)
    for i in range(n):
        for j in range(i + 1, n):
            same = truth[i] == truth[j]
            p = pred(sets[i], sets[j])
            tp += same and p
            fp += (not same) and p
            tn += (not same) and (not p)
            fn += same and (not p)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return tpr - fpr


def components(sets: list[frozenset[int]], tau: int) -> np.ndarray:
    parent = list(range(len(sets)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if len(sets[i] & sets[j]) >= tau:
                parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(sets))])


def run_repeat(mode: str, vecs: np.ndarray, cents: np.ndarray, seed: int) -> dict:
    cfg = MODES[mode]
    rng = np.random.default_rng(seed)
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.base_overrides(
            dim=DIM,
            buckets=B,
            capacity=CAP,
            block=BLOCK,
            epoch_writes=16,
            probe_width=KPROBE,
            probe_spread=cfg["probe_spread"],
        )
        if cfg["grouped"]:
            over = lab.merge(over, {"v2": {"probe_mode": "grouped", "group_size": 8}})
        eng, _, _ = lab.create("p", tmp / "s", cents, over, {})
        for i in rng.choice(len(vecs), size=B * 4, replace=False):
            with contextlib.suppress(Exception):
                eng.put(f"seed {int(i)}", vecs[int(i)].tolist(), "{}")
        lab.drain(eng)
        eng.flush()

        trace = tmp / "t.jsonl"
        trace.write_text("", encoding="utf-8")
        eng.set_trace_sink(str(trace))
        queries = vecs[rng.choice(len(vecs), size=N_QUERIES, replace=False)]
        order = [q for q in range(N_QUERIES) for _ in range(REPEATS_PER_QUERY)]
        rng.shuffle(order)
        sets = read_sets_for(eng, trace, queries, order)
        with contextlib.suppress(Exception):
            eng.close()

        truth = np.array(order)
        train_q = set(rng.choice(N_QUERIES, size=N_QUERIES // 2, replace=False).tolist())
        tr = [i for i, q in enumerate(order) if q in train_q]
        te = [i for i, q in enumerate(order) if q not in train_q]
        s_tr, t_tr = [sets[i] for i in tr], truth[tr]
        s_te, t_te = [sets[i] for i in te], truth[te]

        j_train = {t: youden(s_tr, t_tr, lambda a, b, t=t: len(a & b) >= t) for t in TAUS}
        tau = max(j_train, key=lambda t: (j_train[t], -t))
        ari_train = {t: adjusted_rand_score(t_tr, components(s_tr, t)) for t in TAUS}
        tau_c = max(ari_train, key=lambda t: (ari_train[t], -t))
        return {
            "mode": mode,
            "seed": seed,
            "regions_per_search": float(np.mean([len(s) for s in sets])),
            "exact_j": youden(s_te, t_te, lambda a, b: a == b),
            "threshold_tau": tau,
            "threshold_j": youden(s_te, t_te, lambda a, b, t=tau: len(a & b) >= t),
            "cluster_tau": tau_c,
            "cluster_ari": float(adjusted_rand_score(t_te, components(s_te, tau_c))),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    args = ap.parse_args()
    t0 = time.time()

    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ][:4000]
    vecs = embed_cached([r["text"] for r in rows])
    cents = lab.kmeans_centroids(vecs, B, seed=0)

    results = {
        "experiment": "E15",
        "buckets": B,
        "probe_width": KPROBE,
        "n_queries_per_repeat": N_QUERIES,
        "searches_per_query": REPEATS_PER_QUERY,
        "threshold_selection": (
            "tau fitted on pairs among half the queries, scored on the other half"
        ),
        "adversary_class": "RW (observes reads). A W adversary sees no searches.",
        "modes": [],
    }
    for mode in args.modes:
        reps = [run_repeat(mode, vecs, cents, seed=1300 + r) for r in range(args.repeats)]
        summary: dict = {
            "mode": mode,
            "repeats": reps,
            "regions_per_search": float(np.mean([r["regions_per_search"] for r in reps])),
        }
        for key in ("exact_j", "threshold_j", "cluster_ari"):
            m, lo, hi = lab.boot_ci([r[key] for r in reps])
            summary[key] = {"mean": m, "lo": lo, "hi": hi}
        taus = [r["threshold_tau"] for r in reps]
        summary["threshold_tau_mode"] = int(np.bincount(taus).argmax())
        results["modes"].append(summary)
        print(
            f"[{mode:>10}] regions/search {summary['regions_per_search']:.1f}  "
            f"exact J {summary['exact_j']['mean']:.3f}  "
            f"threshold J {summary['threshold_j']['mean']:.3f} "
            f"[{summary['threshold_j']['lo']:.3f}, {summary['threshold_j']['hi']:.3f}]  "
            f"cluster ARI {summary['cluster_ari']['mean']:.3f}",
            flush=True,
        )
    results["elapsed_s"] = time.time() - t0
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
