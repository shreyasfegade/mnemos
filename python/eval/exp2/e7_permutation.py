"""E7: recovering the keyed partition permutation from probe co-occurrence alone.

Mnemos-P hides which physical region holds which logical partition behind a
keyed permutation, and the paper's stated reason for it is that a public partition table would
otherwise let an adversary label regions with no induced writes at all. But a read-observing
adversary does not need induced writes either: a genuine probe set is the ``kprobe`` *nearest*
partitions to a query, so the regions in one probe set are geometric neighbours. Collect enough
probe sets, build the region co-occurrence graph, and match it against the graph the public
partition table predicts. If the match succeeds, the permutation falls out of ordinary search
traffic.

This bears on what the permutation buys, not on cohort H directly: the write-observing adversary
of the anchoring instantiations sees no reads at all, so none of this reaches it. The result is
reported against the read-observing class and labelled as such.

Both graphs are built from the **same public query pool**, which is what makes them comparable:
the adversary embeds public documents, computes their nearest partitions from the public table to
get the predicted graph, and observes the real read sets to get the measured one. Matching is
seeded graph matching, solved as a quadratic assignment problem.

    cd python && python -m eval.exp2.e7_permutation
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
from scipy.optimize import quadratic_assignment

from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e7_permutation.json"

B = 64
CAP = 16
DIM = 768
BLOCK = 256
KPROBE = 4
N_QUERIES = 800
QAP_SEEDS = 6

MODES = {
    # s = 1: the read set is exactly the genuine probes, which is the cleanest case for the
    # adversary and the configuration the paper shipped before the query-equality mitigation.
    "spread_s1": {"probe_spread": 1, "grouped": False},
    # s = 2: the shipped default, half genuine and half fresh random.
    "spread_s2": {"probe_spread": 2, "grouped": False},
    # grouped: one fixed group per search, so co-occurrence exists only at group granularity.
    "grouped": {"probe_spread": 1, "grouped": True},
}


def normalise(g: np.ndarray) -> np.ndarray:
    m = g.max()
    return g / m if m > 0 else g


def predicted_graph(cents: np.ndarray, queries: np.ndarray, kprobe: int) -> np.ndarray:
    """What the public table says: how often two logical partitions are the query's neighbours."""
    g = np.zeros((len(cents), len(cents)))
    sims = queries @ cents.T
    for row in sims:
        near = np.argsort(-row)[:kprobe]
        for i, a in enumerate(near):
            for b in near[i + 1 :]:
                g[a, b] += 1
                g[b, a] += 1
    return normalise(g)


def qap_value(pred: np.ndarray, obs: np.ndarray, est: np.ndarray) -> float:
    return float(np.sum(pred * obs[np.ix_(est, est)]))


def two_opt(pred: np.ndarray, obs: np.ndarray, est: np.ndarray, max_sweeps: int = 50) -> np.ndarray:
    """Pairwise-swap local search on the QAP objective, from a FAQ solution.

    FAQ solves a continuous relaxation and projects back to a permutation, which leaves it at a
    point no single swap improves only by accident. An adversary has no reason to stop there: a
    swap of two logical partitions' regions changes the objective by

        2 * sum_{k != a,b} (pred[a,k] - pred[b,k]) * (obs[p_b, p_k] - obs[p_a, p_k])

    for symmetric, zero-diagonal graphs, which costs O(B) per pair. Sweep all pairs, take every
    improving swap, stop when a sweep finds none. The objective never decreases, so this is a
    strictly stronger matcher than FAQ alone, not a different one.
    """
    p = est.copy()
    n = len(p)
    for _ in range(max_sweeps):
        improved = False
        for a in range(n):
            for b in range(a + 1, n):
                dp = pred[a] - pred[b]
                do = obs[p[b], p] - obs[p[a], p]
                terms = dp * do
                delta = 2.0 * (terms.sum() - terms[a] - terms[b])
                if delta > 1e-12:
                    p[a], p[b] = p[b], p[a]
                    improved = True
        if not improved:
            break
    return p


def observed_graph(eng, trace: Path, queries: np.ndarray, n_regions: int) -> np.ndarray:
    """What the disk shows: how often two physical regions are read together."""
    g = np.zeros((n_regions, n_regions))
    offset = trace.stat().st_size if trace.exists() else 0
    for q in queries:
        eng.search(q.tolist())
        with trace.open("r", encoding="utf-8") as fh:
            fh.seek(offset)
            lines = [line for line in fh if line.strip()]
        offset = trace.stat().st_size
        regs = []
        for line in lines:
            e = json.loads(line)
            if e["kind"] == "region_read":
                regs.append(int(e["region"]))
        for i, a in enumerate(regs):
            for b in regs[i + 1 :]:
                g[a, b] += 1
                g[b, a] += 1
    return normalise(g)


def run_mode(mode: str, vecs, cents, seed: int) -> dict:
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

        for i in range(B * 4):
            with contextlib.suppress(Exception):
                eng.put(f"seed {i}", vecs[i % len(vecs)].tolist(), "{}")
        lab.drain(eng)
        eng.flush()

        perm_true = np.array(eng.ground_truth_permutation())  # region_of[logical]
        trace = tmp / "t.jsonl"
        trace.write_text("", encoding="utf-8")
        eng.set_trace_sink(str(trace))

        qidx = rng.choice(len(vecs), size=N_QUERIES, replace=False)
        queries = vecs[qidx]
        obs = observed_graph(eng, trace, queries, B)
        with contextlib.suppress(Exception):
            eng.close()

        pred = predicted_graph(cents, queries, KPROBE)

        # Two matchers. FAQ alone is what an earlier harness ran; FAQ followed by 2-opt from each
        # restart is strictly stronger, and it is the adversary reported, because a leakage
        # measurement is a lower bound and should be taken with the best attacker available.
        best, best_ref, best_ref_val = None, None, -np.inf
        for t in range(QAP_SEEDS):
            res = quadratic_assignment(
                pred,
                obs,
                method="faq",
                options={"maximize": True, "rng": np.random.default_rng(t), "maxiter": 80},
            )
            if best is None or res.fun > best.fun:
                best = res
            ref = two_opt(pred, obs, np.asarray(res.col_ind))
            val = qap_value(pred, obs, ref)
            if val > best_ref_val:
                best_ref, best_ref_val = ref, val
        faq = np.asarray(best.col_ind)

        # The adversary can also settle for less: does it place a partition in the right
        # neighbourhood, even when the exact region is wrong? Measured as the fraction whose
        # estimated region is a true neighbour of the right one in the observed graph.
        def neighbourhood(est: np.ndarray) -> float:
            hits = 0
            for logical in range(B):
                true_r, got = perm_true[logical], est[logical]
                if got == true_r or obs[true_r, got] > 0:
                    hits += 1
            return hits / B

        rate = float(np.mean(perm_true == best_ref))
        return {
            "mode": mode,
            "seed": seed,
            "exact_recovery": rate,
            "neighbourhood_recovery": neighbourhood(best_ref),
            "faq_exact_recovery": float(np.mean(perm_true == faq)),
            "faq_neighbourhood_recovery": neighbourhood(faq),
            "null": 1.0 / B,
            "advantage": rate - 1.0 / B,
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
        "experiment": "E7",
        "buckets": B,
        "kprobe": KPROBE,
        "n_queries": N_QUERIES,
        "adversary_class": (
            "RW (observes reads). A W adversary -- sync, backup, imaging -- sees no searches, "
            "so this channel does not reach it."
        ),
        "modes": [],
    }
    for mode in args.modes:
        exact, neigh, faq_exact = [], [], []
        for rep in range(args.repeats):
            r = run_mode(mode, vecs, cents, seed=700 + rep)
            exact.append(r["exact_recovery"])
            neigh.append(r["neighbourhood_recovery"])
            faq_exact.append(r["faq_exact_recovery"])
            print(
                f"[{mode:>10}] rep {rep}: exact {r['exact_recovery']:.3f} "
                f"(FAQ alone {r['faq_exact_recovery']:.3f})  "
                f"neighbourhood {r['neighbourhood_recovery']:.3f}  (null {r['null']:.3f})",
                flush=True,
            )
        m, lo, hi = lab.boot_ci(exact)
        nm, nlo, nhi = lab.boot_ci(neigh)
        fm, flo, fhi = lab.boot_ci(faq_exact)
        results["modes"].append(
            {
                "mode": mode,
                "matcher": "FAQ from 6 restarts, each refined by pairwise-swap local search",
                "exact_mean": m,
                "exact_lo": lo,
                "exact_hi": hi,
                "neighbourhood_mean": nm,
                "neighbourhood_lo": nlo,
                "neighbourhood_hi": nhi,
                "faq_exact_mean": fm,
                "faq_exact_lo": flo,
                "faq_exact_hi": fhi,
                "null": 1.0 / B,
                "repeats": exact,
                "faq_repeats": faq_exact,
            }
        )
        print(
            f"[{mode:>10}] exact {m:.3f} [{lo:.3f},{hi:.3f}]  "
            f"neighbourhood {nm:.3f}  null {1.0 / B:.3f}",
            flush=True,
        )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"wrote {OUT.relative_to(C.repo_root())} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
