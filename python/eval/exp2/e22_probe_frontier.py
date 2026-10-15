"""E22: the probe-design privacy/utility frontier.

The paper measured query equality at one operating point each for the spread (s = 1, 2, 4) and
grouped (group size 8) mechanisms. The whole curve is measured: how read amplification
trades against query-equality leakage as the knob moves. This sweeps the spread s in
{1, 2, 3, 4, 6, 8} and the grouped group size in {2, 4, 8, 16, 32}, reporting the read amplification
(regions read per search) and the held-out query-equality Youden J, over independent stores. It
reuses E15's measurement (fresh store per repeat, threshold fixed on a held-out query split) so
the numbers match the shipped points.

Retrieval degradation, the third axis, is measured by E11 at the shipped points and by E26 across
the grid; this experiment is the security axis (amplification vs equality leakage).

    cd python && python -m eval.exp2.e22_probe_frontier --repeats 20
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
from eval.exp2.e15_query_equality import (
    BLOCK,
    CAP,
    DIM,
    KPROBE,
    N_QUERIES,
    REPEATS_PER_QUERY,
    TAUS,
    components,
    read_sets_for,
    youden,
)
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e22_probe_frontier.json"
B = 64

SPREADS = (1, 2, 3, 4, 6, 8)
GROUP_SIZES = (2, 4, 8, 16, 32)


def run_repeat(spread: int, group_size: int | None, vecs, cents, seed: int) -> dict:
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
            probe_spread=spread,
        )
        if group_size is not None:
            over = lab.merge(over, {"v2": {"probe_mode": "grouped", "group_size": group_size}})
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
            "read_amplification": float(np.mean([len(s) for s in sets])),
            "threshold_j": youden(s_te, t_te, lambda a, b, t=tau: len(a & b) >= t),
            "cluster_ari": float(adjusted_rand_score(t_te, components(s_te, tau_c))),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sweep(label: str, spread: int, group_size: int | None, vecs, cents, repeats: int) -> dict:
    runs = [run_repeat(spread, group_size, vecs, cents, 4200 + i) for i in range(repeats)]
    jm, jlo, jhi = lab.boot_ci([r["threshold_j"] for r in runs])
    amp = float(np.mean([r["read_amplification"] for r in runs]))
    ari = float(np.mean([r["cluster_ari"] for r in runs]))
    print(
        f"{label:>14}: read x{amp / KPROBE:.1f} ({amp:.1f} regions)  "
        f"query-eq J {jm:.3f} [{jlo:.3f}, {jhi:.3f}]  cluster ARI {ari:.3f}",
        flush=True,
    )
    return {
        "mechanism": "grouped" if group_size is not None else "spread",
        "spread": spread,
        "group_size": group_size,
        "read_amplification": amp,
        "read_amplification_x": amp / KPROBE,
        "threshold_j": jm,
        "threshold_j_lo": jlo,
        "threshold_j_hi": jhi,
        "cluster_ari": ari,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()

    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ][:4000]
    vecs = embed_cached([r["text"] for r in rows])
    cents = lab.kmeans_centroids(vecs, B, seed=0)
    print(f"B={B} C={CAP} k={KPROBE}; query-equality frontier over independent stores", flush=True)

    out = {"experiment": "E22", "buckets": B, "kprobe": KPROBE, "spread": [], "grouped": []}
    for s in SPREADS:
        out["spread"].append(sweep(f"spread s={s}", s, None, vecs, cents, args.repeats))
    for g in GROUP_SIZES:
        out["grouped"].append(sweep(f"grouped g={g}", 1, g, vecs, cents, args.repeats))
    out["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())} in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
