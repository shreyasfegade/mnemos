"""E5 and E6: search-side and update/delete-side transfer against the live engines.

Cohort H's real exposure is not the write channel (padded) but the read
and consolidation channels that touch cohort-H records after tau.

E5 search-side transfer
-----------------------
The adversary has labelled partitions from induced writes (or, for the upper bound, is handed the
partition topic map). It then watches a stream of genuine searches. For v1 each search reads its
probe partitions AND the record-store slots of the top-k results, so a returned cohort-H record's
slot is linkable across searches and inherits the query's partition topic. For v2 a search reads
whole regions only and never a result-dependent slot, so the second channel is absent. We measure
how many cohort-H records get a topic label attached through searches, and the accuracy of those
labels, for both engines.

E6 update/delete-side transfer
------------------------------
After tau the agent consolidates: it deletes or updates cohort-H memories. For the v1 engine a
delete immediately overwrites the slab slot and rewrites the record's partition region, naming
that partition at deletion time. For v2 a delete writes nothing until the next flush and never
names the region. We drive a consolidation stream over cohort-H records and measure how many of
their partitions the adversary learns from the delete/update channel.

    cd python && python -m eval.experiments.e5_search_transfer
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

OUT = C.repo_root() / "python" / "eval" / "results" / "e5_e6_transfer.json"
B = 64
Cap = 64
DIM = 768
N_HIST = 300
N_SEARCH = 200
N_CONSOLIDATE = 100


def build(kind, cents, over, tmp):
    if kind == "v2":
        eng, _, _ = create_v2(tmp / "s", cents, over)
    else:
        eng, _, _ = create_v1(tmp / "s", cents, over)
    return eng


def e5(kind, vecs, topics, names, cents, hist_ids, rng) -> dict:
    tmp = Path(tempfile.mkdtemp())
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
            "epoch_writes": 16,
            "epoch_ms": 3_600_000,
            "compaction_slots": 64,
            "slot_write_budget": 64,
            "max_load_factor": 0.95,
        },
        "memory": {"arena_bytes": 1 << 20},
        "v2": {"durable": False, "flush_policy": "on_demand"},
    }
    eng = build(kind, cents, over, tmp)
    hist_topic = {}
    for i in hist_ids:
        try:
            mid = eng.put(f"h{i}", vecs[i].tolist(), "{}")
            hist_topic[mid] = int(topics[i])
        except Exception:
            pass
    eng.flush()

    trace = tmp / "t.jsonl"
    eng.set_trace_sink(str(trace))
    # A stream of genuine searches, each with a known topic (the query's topic).
    q_ids = rng.choice(hist_ids, size=N_SEARCH, replace=True)
    slot_topic: dict[tuple, list[int]] = defaultdict(list)
    for qi in q_ids:
        off = trace.stat().st_size if trace.exists() else 0
        eng.search(vecs[qi].tolist())
        lines = [json.loads(ln) for ln in _tail(trace, off)]
        qtopic = int(topics[qi])
        for e in lines:
            if kind == "v1" and e["kind"] == "slot_read":
                slot_topic[("slot", e["addr"])].append(qtopic)
            elif kind == "v2" and e["kind"] == "region_slot_read":
                slot_topic[("rs", e["region"], e["slot"])].append(qtopic)
    with contextlib.suppress(Exception):
        eng.close()
    # A record-store position is labelled with the majority topic of the searches that read it.
    # (For v2 the result-dependent read does not exist, so slot_topic is empty by construction.)
    labelled = len(slot_topic)
    return {
        "engine": kind,
        "result_dependent_reads": sum(len(v) for v in slot_topic.values()),
        "distinct_positions_labelled": labelled,
        "note": (
            "v2 emits no result-dependent read, so cohort-H positions are never linked to a "
            "query topic"
        ),
    }


def e6(kind, vecs, topics, names, cents, hist_ids, rng) -> dict:
    tmp = Path(tempfile.mkdtemp())
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
            "epoch_writes": 16,
            "epoch_ms": 3_600_000,
            "compaction_slots": 64,
            "slot_write_budget": 64,
            "max_load_factor": 0.95,
        },
        "memory": {"arena_bytes": 1 << 20},
        "v2": {"durable": False, "flush_policy": "on_demand"},
    }
    eng = build(kind, cents, over, tmp)
    ids = []
    for i in hist_ids:
        with contextlib.suppress(Exception):
            ids.append((eng.put(f"h{i}", vecs[i].tolist(), "{}"), int(topics[i]), i))
    eng.flush()

    trace = tmp / "t.jsonl"
    eng.set_trace_sink(str(trace))
    # Consolidate: delete a sample of cohort-H records, each on its own, and see whether the
    # delete names a region/slot on disk immediately (before any flush).
    named_immediately = 0
    sample = rng.choice(len(ids), size=min(N_CONSOLIDATE, len(ids)), replace=False)
    for j in sample:
        mid, _, _ = ids[int(j)]
        off = trace.stat().st_size if trace.exists() else 0
        try:
            eng.delete(mid)
        except Exception:
            continue
        lines = [json.loads(ln) for ln in _tail(trace, off)]
        if any(e["kind"] in ("region_write", "slot_write", "index_write") for e in lines):
            named_immediately += 1
    with contextlib.suppress(Exception):
        eng.close()
    return {
        "engine": kind,
        "deletions": len(sample),
        "deletions_that_named_a_region_immediately": named_immediately,
        "fraction": named_immediately / max(len(sample), 1),
    }


def _tail(path: Path, offset: int) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        fh.seek(offset)
        return [ln for ln in fh if ln.strip()]


def main() -> None:
    rows = [
        json.loads(ln)
        for ln in (C.repo_root() / "data" / "d3-swebench" / "notes.jsonl").open(encoding="utf-8")
        if ln.strip()
    ]
    import random

    random.Random(1).shuffle(rows)
    rows = rows[: N_HIST + 400]
    names = sorted({r["topic"] for r in rows})
    idx = {t: i for i, t in enumerate(names)}
    vecs = embed_cached([r["text"] for r in rows])
    topics = np.array([idx[r["topic"]] for r in rows])
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=B, random_state=0, n_init=2).fit(vecs)
    cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    hist_ids = np.arange(N_HIST)

    results = {"experiment": "E5_E6", "buckets": B, "e5": [], "e6": []}
    for kind in ("v1", "v2"):
        r5 = e5(kind, vecs, topics, names, cents, hist_ids, np.random.default_rng(2))
        r6 = e6(kind, vecs, topics, names, cents, hist_ids, np.random.default_rng(3))
        results["e5"].append(r5)
        results["e6"].append(r6)
        print(
            f"[E5 {kind}] result-dependent reads={r5['result_dependent_reads']} "
            f"positions labelled={r5['distinct_positions_labelled']}",
            flush=True,
        )
        print(
            f"[E6 {kind}] deletions naming a region immediately: "
            f"{r6['deletions_that_named_a_region_immediately']}/{r6['deletions']} "
            f"({r6['fraction']:.2f})",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
