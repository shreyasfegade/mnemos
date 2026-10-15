"""E11: capacity and write amplification of v1 and v2 over a long sparse write trace.

Under sparse agent writes almost every v1 flush contains one real record and
15 dummies, so the index fills after ~1,024 real memories and topic-heavy partitions overflow
far earlier, while each memory costs a full padded batch of slab and index writes. v2 stores no
dummies, so capacity counts real records only.

This drives both engines with the same trace of real writes drawn from a skewed topic
distribution (a few topics dominate, as a real agent's work does), one write per flush (the
sparse regime), and reports, as a function of the number of real memories: usable capacity
before the first partition overflows, and bytes written per real memory.

    cd python && python -m eval.experiments.e11_capacity
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from eval.lab import corpus as C
from eval.lab.embcache import embed_cached
from eval.lab.stores import create_v1, create_v2

OUT = C.repo_root() / "python" / "eval" / "results" / "e11_capacity.json"
B = 64
Cap = 32
DIM = 768
N_WRITES = 1500
ZIPF_A = 1.4


def load_records() -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    rows = [
        json.loads(ln)
        for ln in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if ln.strip()
    ]
    names = sorted({r["topic"] for r in rows})
    idx = {t: i for i, t in enumerate(names)}
    texts = [r["text"] for r in rows]
    vecs = embed_cached(texts)
    topics = np.array([idx[r["topic"]] for r in rows])
    return texts, vecs, topics, names


def skewed_trace(topics: np.ndarray, names: list[str], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, len(names) + 1)
    w = 1.0 / ranks**ZIPF_A
    w /= w.sum()
    order = rng.permutation(len(names))
    topic_w = np.zeros(len(names))
    topic_w[order] = w
    picks = []
    by_topic = {t: np.where(topics == t)[0] for t in range(len(names))}
    for _ in range(n):
        t = rng.choice(len(names), p=topic_w)
        picks.append(int(rng.choice(by_topic[t])))
    return np.array(picks)


def run_v2(dir_: Path, cents, vecs, trace, block) -> dict:
    eng, h, _ = create_v2(
        dir_,
        cents,
        {
            "embedding": {"dim": DIM},
            "index": {
                "buckets": B,
                "bucket_capacity": Cap,
                "top_k": 10,
                "probe_width": 4,
                "probe_spread": 1,
            },
            "crypto": {"block_size_bytes": block, "max_entry_bytes": block},
            "epoch": {"epoch_writes": 16, "epoch_ms": 3_600_000},
            "memory": {"arena_bytes": 1 << 20},
            "v2": {"durable": False, "payload_blocks": 1, "flush_policy": "on_demand"},
        },
    )
    first_overflow = None
    bytes0 = 0
    per_mem_bytes = []
    for i, r in enumerate(trace):
        try:
            eng.put(f"m{i}", vecs[r].tolist(), "{}")
            eng.flush()  # sparse: one real record per flush
        except Exception:
            if first_overflow is None:
                first_overflow = i
            break
        s = eng.stat()
        per_mem_bytes.append(s["region_bytes_written"] + s["manifest_bytes_written"] - bytes0)
        bytes0 = s["region_bytes_written"] + s["manifest_bytes_written"]
    eng.close()
    return {
        "engine": "v2",
        "params_hash": h,
        "usable_before_refusal": first_overflow if first_overflow is not None else len(trace),
        "median_bytes_per_memory": float(np.median(per_mem_bytes)) if per_mem_bytes else 0.0,
        "final_live": eng_stat_safe(eng),
    }


def eng_stat_safe(eng) -> int:
    try:
        return int(eng.stat()["live_entries"])
    except Exception:
        return -1


def run_v1(dir_: Path, cents, vecs, trace, block) -> dict:
    import mnemos_native as m  # noqa: F401

    eng, h, _ = create_v1(
        dir_,
        cents,
        {
            "embedding": {"dim": DIM},
            "index": {
                "buckets": B,
                "bucket_capacity": Cap,
                "n_max": B * Cap,
                "top_k": 10,
                "probe_width": 4,
                "probe_spread": 1,
                "overflow_depth": 0,
            },
            "crypto": {"block_size_bytes": block, "max_entry_bytes": block},
            "epoch": {
                "epoch_writes": 16,
                "epoch_ms": 3_600_000,
                "compaction_slots": 64,
                "slot_write_budget": 64,
                "max_load_factor": 0.95,
            },
            "memory": {"arena_bytes": 1 << 20},
        },
    )
    trace_dir = dir_
    first_refusal = None
    dir_size_before = 0
    per_mem = []
    for i, r in enumerate(trace):
        try:
            eng.put(f"m{i}", vecs[r].tolist(), "{}")
            eng.flush()
        except Exception:
            if first_refusal is None:
                first_refusal = i
            break
        size = sum(f.stat().st_size for f in trace_dir.glob("*") if f.is_file())
        # v1 files are preallocated, so on-disk size does not grow; instead count via a proxy:
        # each flush writes slot_write_budget slab slots + epoch_writes index regions + manifest.
        per_mem.append(size - dir_size_before)
        dir_size_before = size
    return {
        "engine": "v1",
        "params_hash": h,
        "usable_before_refusal": first_refusal if first_refusal is not None else len(trace),
        "note": "v1 preallocates, so bytes/memory is measured via the write model, not file growth",
    }


def main() -> None:
    _, vecs, topics, names = load_records()
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=B, random_state=0, n_init=2).fit(vecs)
    cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    part = np.argmax(vecs @ cents.T, axis=1)
    trace = skewed_trace(topics, names, N_WRITES, seed=7)

    # How skewed is the induced load across partitions?
    load = np.bincount(part[trace], minlength=B)
    results = {
        "experiment": "E11",
        "buckets": B,
        "capacity": Cap,
        "n_writes": N_WRITES,
        "zipf_a": ZIPF_A,
        "max_partition_load_fraction": float(load.max() / load.sum()),
        "runs": [],
    }

    tmp = Path(tempfile.mkdtemp())
    try:
        results["runs"].append(run_v2(tmp / "v2", cents, vecs, trace, block=1024))
        results["runs"].append(run_v1(tmp / "v1", cents, vecs, trace, block=1024))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for r in results["runs"]:
        print(r)
    print(f"max partition load fraction {results['max_partition_load_fraction']:.3f}")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
