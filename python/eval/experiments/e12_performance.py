"""E12: search and write latency of Mnemos v2 against a plaintext IVF baseline.

Measured through the FFI at the shipped configuration, at two corpus sizes, reporting p50/p95/p99
per operation. The plaintext baseline is a numpy IVF over the same centroids: assign to nearest
centroid, scan the probed partitions, return top-k. Writes are timed as full batches of
``epoch_writes`` incremental puts on an already-built store, so the flush that a batch triggers is
inside the timed region and the store is not rebuilt each sample.

    cd python && python -m eval.experiments.e12_performance
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from eval.lab import corpus as C
from eval.lab.embcache import device, embed_cached
from eval.lab.stores import create_v2

OUT = C.repo_root() / "python" / "eval" / "results" / "e12_performance.json"
SIZES = (1000, 10000)
DIM = 768
N_SEARCH = 200
N_BATCH = 60  # batches of epoch_writes puts to time


def geometry(n: int) -> tuple[int, int]:
    b = max(16, 1 << int(np.ceil(np.log2(max(4.0, np.sqrt(n))))))
    c = max(64, 1 << int(np.ceil(np.log2(max(2.0, (n / b) * 3)))))
    return b, c


def pct(xs: list[float]) -> dict:
    a = np.array(xs) * 1e6  # to microseconds
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
    }


class PlaintextIVF:
    def __init__(self, cents, k, top_k):
        self.cents = cents
        self.k = k
        self.top_k = top_k
        self.vecs = np.zeros((0, cents.shape[1]), dtype=np.float32)
        self.part = np.zeros(0, dtype=np.int64)

    def add(self, v):
        p = int(np.argmax(self.cents @ v))
        self.vecs = np.vstack([self.vecs, v[None, :]])
        self.part = np.append(self.part, p)

    def search(self, q):
        probes = np.argsort(-(self.cents @ q))[: self.k]
        mask = np.isin(self.part, probes)
        idx = np.where(mask)[0]
        if idx.size == 0:
            return []
        sims = self.vecs[idx] @ q
        order = idx[np.argsort(-sims)[: self.top_k]]
        return order.tolist()


def run(n: int, vecs: np.ndarray, cents: np.ndarray) -> dict:
    B, Cap = geometry(n)
    over = {
        "embedding": {"dim": DIM},
        "index": {
            "buckets": B,
            "bucket_capacity": Cap,
            "top_k": 10,
            "probe_width": 4,
            "probe_spread": 2,
        },
        "crypto": {"block_size_bytes": 4096, "max_entry_bytes": 4096},
        "epoch": {"epoch_writes": 16, "epoch_ms": 3_600_000},
        "memory": {"arena_bytes": 4 << 20},
        "v2": {"durable": False, "flush_policy": "on_demand"},
    }
    tmp = Path(tempfile.mkdtemp())
    eng, _, _ = create_v2(tmp / "s", cents, over)
    plain = PlaintextIVF(cents, k=4, top_k=10)
    added = 0
    for i in range(n):
        try:
            eng.put(f"r{i}", vecs[i].tolist(), "{}")
            plain.add(vecs[i])
            added += 1
        except Exception:
            break
    eng.flush()

    rng = np.random.default_rng(0)
    qidx = rng.integers(0, added, N_SEARCH)
    v2_s, pl_s = [], []
    for qi in qidx:
        q = vecs[qi]
        t = time.perf_counter()
        eng.search(q.tolist())
        v2_s.append(time.perf_counter() - t)
        t = time.perf_counter()
        plain.search(q)
        pl_s.append(time.perf_counter() - t)

    # amortized writes: time batches of 16 incremental puts (the last triggers a flush).
    v2_w, pl_w = [], []
    nxt = added
    for _ in range(N_BATCH):
        t = time.perf_counter()
        for _ in range(16):
            with contextlib.suppress(Exception):
                eng.put(f"w{nxt}", vecs[(nxt) % len(vecs)].tolist(), "{}")
            nxt += 1
        v2_w.append((time.perf_counter() - t) / 16)
        t = time.perf_counter()
        for j in range(16):
            plain.add(vecs[(nxt + j) % len(vecs)])
        pl_w.append((time.perf_counter() - t) / 16)
    eng.close()

    return {
        "n": n,
        "buckets": B,
        "capacity": Cap,
        "added": added,
        "search_v2_us": pct(v2_s),
        "search_plain_us": pct(pl_s),
        "search_overhead_x": pct(v2_s)["p50"] / max(pct(pl_s)["p50"], 1e-9),
        "put_v2_us": pct(v2_w),
        "put_plain_us": pct(pl_w),
        "put_overhead_x": pct(v2_w)["p50"] / max(pct(pl_w)["p50"], 1e-9),
    }


def main() -> None:
    rows = [
        json.loads(ln)
        for ln in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if ln.strip()
    ]
    texts = [r["text"] for r in rows]
    # tile embeddings up to 10k
    base = embed_cached(texts)
    reps = int(np.ceil(10000 / len(base)))
    vecs = np.tile(base, (reps, 1))[:10000].astype(np.float32)
    from sklearn.cluster import KMeans

    results = {"experiment": "E12", "device": device(), "sizes": SIZES, "rows": []}
    for n in SIZES:
        B = geometry(n)[0]
        km_n = KMeans(n_clusters=B, random_state=0, n_init=2).fit(base)
        cents = (
            km_n.cluster_centers_ / np.linalg.norm(km_n.cluster_centers_, axis=1, keepdims=True)
        ).astype(np.float32)
        r = run(n, vecs, cents)
        results["rows"].append(r)
        print(
            f"n={n}: search v2 p50 {r['search_v2_us']['p50']:.0f}us plain "
            f"{r['search_plain_us']['p50']:.0f}us "
            f"({r['search_overhead_x']:.1f}x) | put v2 p50 {r['put_v2_us']['p50']:.1f}us plain "
            f"{r['put_plain_us']['p50']:.1f}us "
            f"({r['put_overhead_x']:.1f}x)",
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
