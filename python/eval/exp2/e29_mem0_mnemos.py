"""E29: the unmodified mem0 library on a Mnemos backend -- does the clocked store close the oracle?

E18 measured the deduplication membership oracle in mem0 2.2.1 over its default Qdrant store. This
experiment keeps mem0 unmodified and replaces only its vector store, through mem0's own
``VectorStoreBase`` interface, with a Mnemos engine. Both of mem0's collections go to the backend:
the memories and the entity index its linking phase maintains. The adversary is E18's: it supplies a
fact and observes a directory. Arms differ in the backend and in what the adversary observes.

* ``qdrant``   -- mem0 as shipped; observes the Qdrant directory (E18 re-run in this protocol).
* ``m_store``  -- Mnemos-M (sweep flush on a 1 s clock); observes the Mnemos store directory.
* ``m_all``    -- Mnemos-M; observes the store **and** mem0's SQLite history, which mem0 writes on
  every call and appends to on every ADD. This is the scope caveat of Theorem 2 measured: the store
  can be clock-only while the framework's other state is not.
* ``p_store``  -- Mnemos-P (content placement, flushed when the agent writes); observes its store.

The observation window is fixed, not the call's duration: the adversary snapshots when it submits
the fact and again ``T`` seconds later, with ``T`` fixed per arm from the latencies of the seeding
calls (which are the adversary's own induced writes, so calibrating on them is within its power).
The statistic is the number of 4 KiB blocks that changed in the window, oriented a priori (more
writing means non-member); the binary "anything changed" is reported beside it.

    cd python && ..\\.venv-fw\\Scripts\\python -m eval.exp2.e29_mem0_mnemos --arm m_store
    cd python && ..\\.venv-fw\\Scripts\\python -m eval.exp2.e29_mem0_mnemos --aggregate
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "False")

import numpy as np

from eval.exp2 import lab
from eval.exp2.oracle_common import (
    Clock,
    DirView,
    auc_ci,
    changed_blocks,
    facts,
    ollama_embed,
    ollama_ready,
    write_json,
)
from eval.lab import corpus as C

RES = C.repo_root() / "python" / "eval" / "results"
OUT_DIR = RES / "e29"
OUT = RES / "e29_mem0_mnemos.json"
ARMS = ("qdrant", "m_store", "m_all", "p_store")
EMBED = "nomic-embed-text"
DIM = 768
DELTA_S = 1.0


# --------------------------------------------------------------------------- the backend


def _norm(v) -> list[float]:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return (a / n if n > 0 else a).tolist()


class MnemosBackend:
    """One Mnemos engine holding every mem0 collection, with a plaintext mirror in process memory.

    The mirror holds what the locked arena would: the records' payloads and vectors, which mem0's
    ``get``, ``list`` and ``update`` need by id. It never touches disk; the engine is the only
    writer of the observed directory.
    """

    def __init__(self, kind: str, root: Path, centroids: np.ndarray) -> None:
        self.kind = kind
        self.lock = threading.Lock()
        over = lab.base_overrides(
            dim=DIM,
            buckets=64,
            # Room for every seeded, probed and entity record even if topics skew, so Mnemos-P's
            # per-partition bound never refuses one of mem0's writes mid-experiment.
            capacity=128,
            top_k=64,
            probe_width=8,
            probe_spread=1,
            block=4096,
            epoch_writes=4,
            epoch_ms=int(DELTA_S * 1000),
        )
        if kind == "m":
            over = lab.merge(
                over,
                {
                    "m": {
                        "durable": True,
                        "payload_blocks": 1,
                        "flush_ms": int(DELTA_S * 1000),
                        "flush_mode": "sweep",
                        "sweep_slots": 32,
                        "journal_slots": 16,
                        "backlog": "drain",
                        "search": "exact",
                    }
                },
            )
        else:
            over = lab.merge(over, {"v2": {"durable": True, "flush_policy": "on_demand"}})
        self.eng, _, _ = lab.create(kind, root, centroids, over, {})
        self.mirror: dict[str, dict] = {}  # mem0 id -> {mid, vec, payload, col}
        self.clock: Clock | None = None
        if kind == "m":
            self.clock = Clock(self.eng.flush, DELTA_S, self.lock).start()

    def _after_write(self) -> None:
        # Mnemos-P flushes when the agent writes; Mnemos-M waits for its clock.
        if self.kind == "p":
            self.eng.flush()

    def put(self, col: str, mid0: str, vec, payload: dict) -> None:
        v = _norm(vec)
        meta = json.dumps({"c": col, "i": mid0, "p": payload}, separators=(",", ":"))
        with self.lock:
            mid = self.eng.put(str(payload.get("data", ""))[:400], v, meta)
            self._after_write()
        self.mirror[mid0] = {"mid": mid, "vec": v, "payload": payload, "col": col}

    def update(self, mid0: str, vec, payload: dict | None) -> None:
        rec = self.mirror[mid0]
        v = _norm(vec) if vec is not None else rec["vec"]
        p = payload if payload is not None else rec["payload"]
        meta = json.dumps({"c": rec["col"], "i": mid0, "p": p}, separators=(",", ":"))
        with self.lock:
            new = self.eng.update(rec["mid"], str(p.get("data", ""))[:400], v, meta)
            self._after_write()
        rec.update(mid=new, vec=v, payload=p)

    def delete(self, mid0: str) -> None:
        rec = self.mirror.pop(mid0, None)
        if rec is None:
            return
        with self.lock:
            self.eng.delete(rec["mid"])
            self._after_write()

    def search(self, col: str, vec) -> list[tuple[str, float]]:
        with self.lock:
            hits = self.eng.search(_norm(vec))
        out = []
        for h in hits:
            meta, score = h[2], h[3]
            try:
                m = json.loads(meta)
            except Exception:
                continue
            if m.get("c") == col and m.get("i") in self.mirror:
                out.append((m["i"], float(score)))
        return out

    def close(self) -> None:
        if self.clock is not None:
            self.clock.stop()
        with self.lock, contextlib.suppress(Exception):
            self.eng.close()


def _matches(payload: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    for k, want in filters.items():
        if k in ("AND", "OR", "NOT") or isinstance(want, dict | list):
            continue  # operator filters are not used by the add path; accept rather than guess
        if payload.get(k) != want:
            return False
    return True


def make_store_class():
    from mem0.vector_stores.base import VectorStoreBase

    class OutputData:
        def __init__(self, id, score, payload):
            self.id = id
            self.score = score
            self.payload = payload

    class MnemosVectorStore(VectorStoreBase):
        def __init__(self, backend: MnemosBackend, collection: str) -> None:
            self.b = backend
            self.collection_name = collection

        def create_col(self, *a, **k):
            return None

        def insert(self, vectors, payloads=None, ids=None):
            payloads = payloads or [{} for _ in vectors]
            for v, p, i in zip(vectors, payloads, ids, strict=True):
                self.b.put(self.collection_name, str(i), v, dict(p))

        def search(self, query, vectors, top_k=5, filters=None):
            out = []
            for i, s in self.b.search(self.collection_name, vectors):
                p = self.b.mirror[i]["payload"]
                if _matches(p, filters):
                    out.append(OutputData(i, s, p))
                if len(out) >= top_k:
                    break
            return out

        def search_batch(self, queries, vectors_list, top_k=1, filters=None):
            return [
                self.search(q, v, top_k=top_k, filters=filters)
                for q, v in zip(queries, vectors_list, strict=True)
            ]

        def delete(self, vector_id):
            self.b.delete(str(vector_id))

        def update(self, vector_id, vector=None, payload=None):
            self.b.update(str(vector_id), vector, dict(payload) if payload is not None else None)

        def get(self, vector_id):
            rec = self.b.mirror.get(str(vector_id))
            return None if rec is None else OutputData(str(vector_id), None, rec["payload"])

        def list_cols(self):
            return [self.collection_name]

        def delete_col(self):
            for i in [k for k, r in self.b.mirror.items() if r["col"] == self.collection_name]:
                self.b.delete(i)

        def col_info(self):
            return {"name": self.collection_name}

        def list(self, filters=None, top_k=None):
            rows = [
                OutputData(i, None, r["payload"])
                for i, r in self.b.mirror.items()
                if r["col"] == self.collection_name and _matches(r["payload"], filters)
            ]
            return [rows[: top_k or len(rows)]]

        def reset(self):
            self.delete_col()

    return MnemosVectorStore


# --------------------------------------------------------------------------- the experiment


def build_memory(arm: str, work: Path, model: str, centroids: np.ndarray | None):
    from mem0 import Memory

    observed = work / "observed"
    hidden = work / "hidden"
    observed.mkdir(parents=True)
    hidden.mkdir(parents=True)
    qdrant_dir = (observed if arm == "qdrant" else hidden) / "qdrant"
    history = (observed if arm == "m_all" else hidden) / "history.db"
    cfg = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": str(qdrant_dir),
                "collection_name": "m",
                "embedding_model_dims": DIM,
                "on_disk": True,
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {"model": model, "ollama_base_url": "http://127.0.0.1:11434"},
        },
        "embedder": {
            "provider": "ollama",
            "config": {"model": EMBED, "ollama_base_url": "http://127.0.0.1:11434"},
        },
        "history_db_path": str(history),
    }
    mem = Memory.from_config(cfg)
    backend = None
    if arm == "qdrant":
        view = DirView([qdrant_dir])
    else:
        backend = MnemosBackend(
            "m" if arm.startswith("m_") else "p", observed / "mnemos", centroids
        )
        cls = make_store_class()
        mem.vector_store = cls(backend, "m")
        mem._entity_store = cls(backend, "m_entities")
        view = DirView([observed], lock=backend.lock)
    return mem, backend, view


def run_arm(arm: str, model: str, n_probes: int, seed_size: int, window: float | None) -> dict:
    ollama_ready([model, EMBED])
    pool = facts()
    rng = np.random.default_rng(29_000 + ARMS.index(arm))
    centroids = None
    if arm != "qdrant":
        # A public table: nomic embeddings of facts that are never seeded or probed.
        cvecs = np.asarray(ollama_embed(pool[-2000:], EMBED), dtype=np.float32)
        cvecs /= np.linalg.norm(cvecs, axis=1, keepdims=True)
        centroids = lab.kmeans_centroids(cvecs, 64, seed=0)

    work = Path(tempfile.mkdtemp(prefix=f"e29-{arm}-"))
    t0 = time.time()
    backend = None
    try:
        mem, backend, view = build_memory(arm, work, model, centroids)

        # Cohort H: seed member facts before tau; their latencies calibrate the window.
        members = pool[:seed_size]
        lat = []
        for f in members:
            s = time.time()
            try:
                mem.add(f, user_id="u")
            except Exception as e:
                print("seed add failed:", type(e).__name__, e, flush=True)
            lat.append(time.time() - s)
        T = window or float(min(45.0, max(6.0, math.ceil(1.25 * max(lat)) + 2)))
        print(
            f"{arm}: seeded {len(members)} facts, max latency {max(lat):.1f}s, window {T:.0f}s",
            flush=True,
        )
        if backend is not None:
            time.sleep(3 * DELTA_S)  # let the sweep settle the seed backlog before probing

        probes = [(f, 1) for f in members[:n_probes]] + [
            (f, 0) for f in pool[seed_size : seed_size + n_probes]
        ]
        rng.shuffle(probes)
        rows = []
        overruns = 0
        for k, (f, is_member) in enumerate(probes):
            before = view.snapshot()
            s = time.time()
            ev = "?"
            try:
                r = mem.add(f, user_id="u")
                res = r.get("results", []) if isinstance(r, dict) else []
                ev = ",".join(sorted({str(x.get("event", "?")) for x in res})) or "NONE"
            except Exception as e:
                ev = f"ERR:{type(e).__name__}"
            took = time.time() - s
            if took > T:
                overruns += 1
            time.sleep(max(0.0, s + T - time.time()))
            after = view.snapshot()
            n = changed_blocks(before, after)
            rows.append(
                {
                    "member": is_member,
                    "changed_blocks": n,
                    "any": int(n > 0),
                    "latency": took,
                    "event": ev,
                }
            )
            if (k + 1) % 10 == 0:
                print(f"  {arm}: {k + 1}/{len(probes)} probes", flush=True)

        y = [1 - r["member"] for r in rows]
        auc_n, lo_n, hi_n = auc_ci(y, [r["changed_blocks"] for r in rows], rng)
        auc_b, lo_b, hi_b = auc_ci(y, [r["any"] for r in rows], rng)
        mean = lambda key, m: float(np.mean([r[key] for r in rows if r["member"] == m]))  # noqa: E731
        out = {
            "experiment": "E29",
            "arm": arm,
            "library": "mem0 2.2.1",
            "model": model,
            "embedder": EMBED,
            "seed_size": seed_size,
            "probes_each": n_probes,
            "window_s": T,
            "window_overruns": overruns,
            "clock_s": DELTA_S if arm.startswith("m_") else None,
            "auc_blocks": auc_n,
            "lo_blocks": lo_n,
            "hi_blocks": hi_n,
            "auc_any": auc_b,
            "lo_any": lo_b,
            "hi_any": hi_b,
            "member_blocks": mean("changed_blocks", 1),
            "nonmember_blocks": mean("changed_blocks", 0),
            "member_any": mean("any", 1),
            "nonmember_any": mean("any", 0),
            "event_counts": {
                e: sum(r["event"] == e for r in rows) for e in sorted({r["event"] for r in rows})
            },
            "clock_errors": (backend.clock.errors[:5] if backend and backend.clock else []),
            "rows": rows,
            "seconds": time.time() - t0,
        }
        write_json(OUT_DIR / f"{arm}.json", out)
        print(
            f"{arm}: AUC(blocks) {auc_n:.3f} [{lo_n:.3f},{hi_n:.3f}]  AUC(any) {auc_b:.3f}  "
            f"in {out['seconds'] / 60:.1f} min",
            flush=True,
        )
        return out
    finally:
        if backend is not None:
            backend.close()
        shutil.rmtree(work, ignore_errors=True)


def aggregate() -> None:
    arms = {}
    for a in ARMS:
        f = OUT_DIR / f"{a}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            d.pop("rows", None)
            arms[a] = d
    write_json(OUT, {"experiment": "E29", "arms": arms})
    print(f"wrote {OUT.relative_to(C.repo_root())} with arms {sorted(arms)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--probes", type=int, default=40, help="member and non-member probes each")
    ap.add_argument("--seed-size", type=int, default=40)
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--window", type=float, default=None, help="fix T instead of calibrating")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
        return
    if not args.arm:
        ap.error("--arm or --aggregate")
    run_arm(args.arm, args.model, args.probes, args.seed_size, args.window)
    aggregate()


if __name__ == "__main__":
    main()
