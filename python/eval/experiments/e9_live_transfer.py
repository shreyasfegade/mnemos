"""E9: structure transfer against the live engines, v1 and v2, from their real event traces.

The adversary sees only the post-tau trace (region/index writes segmented by flush) and its own
induced-topic ground truth, and runs the realistic frequency estimator (never the oracle). It is
scored on cohort H, written before the trace begins.

Three engines under the same corpus and budget:

- ``v1_padded``   the shipped v1 engine (index-write padding on).
- ``v1_f5``       v1 with padding off (finding F5), as an upper bound on what the channel gives.
- ``v2``          the v2 engine, whose flush rewrites w + 1 whole regions with the real records'
  regions hidden among random ones and a maintenance region.

Because v2's physical regions are a keyed permutation of logical partitions, an attacker scoring
against the *logical* partition of cohort-H records must first invert the permutation, which it
cannot; we give it the strongest fair footing by scoring against the physical region (ground
truth from the engine), so any residual signal is visible.

    cd python && python -m eval.experiments.e9_live_transfer
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.lab import corpus as C
from eval.lab.embcache import embed_cached
from eval.lab.stores import create_v1, create_v2

OUT = C.repo_root() / "python" / "eval" / "results" / "e9_live_transfer.json"
B = 64
Cap = 64
DIM = 768
N_HIST = 400
BUDGETS = (2, 4, 8, 16, 32, 64)
N_TRIALS = 8
W = 16


def read_flushes(trace: Path, kind: str, field: str) -> list[list[int]]:
    flushes, pending = [], []
    for line in trace.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e["kind"] == kind:
            pending.append(int(e[field]))
        elif e["kind"] == "epoch_flush":
            flushes.append(pending)
            pending = []
    if pending:
        flushes.append(pending)
    return flushes


def score_against(
    labels: dict[int, str], gt_region: np.ndarray, topics: np.ndarray, names: list[str]
) -> float:
    hits = sum(
        1 for r, t in zip(gt_region, topics, strict=True) if labels.get(int(r)) == names[int(t)]
    )
    return hits / len(topics)


def frequency_labels(by_topic: dict[str, list[list[int]]], n_regions: int) -> dict[int, str]:
    topics = [t for t, f in by_topic.items() if f]
    if not topics:
        return {}
    rate = np.zeros((len(topics), n_regions))
    for i, t in enumerate(topics):
        for f in by_topic[t]:
            for r in set(f):
                if 0 <= r < n_regions:
                    rate[i, r] += 1
        rate[i] /= max(len(by_topic[t]), 1)
    mean = rate.mean(0)
    n_min = min(len(by_topic[t]) for t in topics)
    q = float(mean.mean())
    margin = 2.0 * float(np.sqrt(max(q * (1 - q), 1e-9) / max(n_min, 1)))
    labels = {}
    for r in range(n_regions):
        best = int(np.argmax(rate[:, r]))
        if rate[best, r] - mean[r] > margin:
            labels[r] = topics[best]
    return labels


def run_engine(kind: str, vecs, topics, names, cents, budget, trial, rng):
    tmp = Path(tempfile.mkdtemp())
    # v1 stores a dummy for every padded slot, so its index must hold real + dummy records:
    # ~N_HIST + budget*n_topics*W. Size it generously; v2 stores no dummies and B*Cap suffices.
    v1_cap = 512
    over = {
        "embedding": {"dim": DIM},
        "index": {
            "buckets": B,
            "bucket_capacity": Cap,
            "top_k": 10,
            "probe_width": 4,
            "probe_spread": 1,
            "n_max": B * Cap,
            "overflow_depth": 0,
        },
        "crypto": {"block_size_bytes": 1024, "max_entry_bytes": 1024},
        "epoch": {
            "epoch_writes": W,
            "epoch_ms": 3_600_000,
            "compaction_slots": 64,
            "slot_write_budget": 64,
            "max_load_factor": 0.95,
        },
        "memory": {"arena_bytes": 1 << 20},
        "v2": {"durable": False, "flush_policy": "on_demand"},
    }
    hist = np.arange(N_HIST)
    post = np.arange(N_HIST, len(vecs))
    if kind == "v2":
        eng, _, _ = create_v2(tmp / "s", cents, over)
        gt_region = np.array([eng.ground_truth_region(vecs[i].tolist()) for i in hist])
        write_kind, field = "region_write", "region"
    else:
        over = json.loads(json.dumps(over))
        over["index"]["bucket_capacity"] = v1_cap
        over["index"]["n_max"] = B * v1_cap
        over["index"]["index_write_padding"] = kind == "v1_padded"
        eng, _, _ = create_v1(tmp / "s", cents, over)
        assign = np.argmax(vecs @ cents.T, axis=1)
        gt_region = assign[hist]
        write_kind, field = "index_write", "bucket"

    for i in hist:
        with contextlib.suppress(Exception):
            eng.put(f"h{i}", vecs[i].tolist(), "{}")
    eng.flush()

    trace = tmp / "t.jsonl"
    eng.set_trace_sink(str(trace))
    by_topic: dict[str, list[list[int]]] = {t: [] for t in names}
    post_by_topic = defaultdict(list)
    for i in post:
        post_by_topic[names[int(topics[i])]].append(i)
    for tname in names:
        pool = post_by_topic.get(tname, [])
        if not pool:
            continue
        picks = rng.choice(len(pool), size=min(budget, len(pool)), replace=False)
        for j in picks:
            with contextlib.suppress(Exception):
                eng.put(f"p{pool[int(j)]}", vecs[pool[int(j)]].tolist(), "{}")
            eng.flush()
            by_topic[tname].append([])
    # reassign flush -> topic in order
    flushes = read_flushes(trace, write_kind, field)
    order = [t for tname in names for t in [tname] * len(post_by_topic.get(tname, [])[:budget])]
    grouped: dict[str, list[list[int]]] = {t: [] for t in names}
    for tname, f in zip(order, flushes, strict=False):
        grouped[tname].append(f)
    with contextlib.suppress(Exception):
        eng.close()

    labels = frequency_labels(grouped, B)
    acc = score_against(labels, gt_region, topics[hist], names)
    return acc, len(labels)


def main() -> None:
    rows = [
        json.loads(ln)
        for ln in (C.repo_root() / "data" / "d3-swebench" / "notes.jsonl").open(encoding="utf-8")
        if ln.strip()
    ]
    import random

    random.Random(0).shuffle(rows)
    rows = rows[:1200]
    names = sorted({r["topic"] for r in rows})
    idx = {t: i for i, t in enumerate(names)}
    vecs = embed_cached([r["text"] for r in rows])
    topics = np.array([idx[r["topic"]] for r in rows])
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=B, random_state=0, n_init=2).fit(vecs)
    cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    chance = 1.0 / len(names)
    print(f"D3: {len(rows)} records, {len(names)} topics, chance {chance:.3f}", flush=True)

    results = {"experiment": "E9", "buckets": B, "chance": chance, "n_hist": N_HIST, "rows": []}
    for kind in ("v1_padded", "v1_f5", "v2"):
        for budget in BUDGETS:
            accs = []
            for trial in range(N_TRIALS):
                acc, _mapped = run_engine(
                    kind,
                    vecs,
                    topics,
                    names,
                    cents,
                    budget,
                    trial,
                    np.random.default_rng(1000 * budget + trial),
                )
                accs.append(acc)
            a = np.array(accs)
            # 95% t-interval over trials
            se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
            row = {
                "engine": kind,
                "budget": budget,
                "acc_mean": float(a.mean()),
                "acc_lo": float(a.mean() - 1.96 * se),
                "acc_hi": float(a.mean() + 1.96 * se),
                "adv": float(max(0.0, a.mean() - chance)),
            }
            results["rows"].append(row)
            print(
                f"{kind:>10} b={budget:>3}  acc {a.mean():.3f} [{row['acc_lo']:.3f}, "
                f"{row['acc_hi']:.3f}]  adv {row['adv']:.3f}",
                flush=True,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
