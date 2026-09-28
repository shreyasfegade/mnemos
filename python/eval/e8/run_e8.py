"""E8: structure transfer against real vector stores under whole-file encryption.

Runs inside WSL, where a gocryptfs FUSE mount provides a size-preserving, block-aligned encryption
layer (like Cryptomator): the ciphertext directory mirrors the plaintext file structure, each file
the same size up to a small per-file header, and a write to plaintext offset X produces a write to
roughly ciphertext offset X. A sync/imaging adversary sees exactly which ciphertext blocks change.

For each store we build cohort H, snapshot the ciphertext directory, induce writes per topic while
snapshotting after each, and diff the ciphertext to learn which blocks each induced write touched.
We then map blocks to topics by frequency and score cohort-H recovery, the same structure-transfer
attack as the Mnemos experiments but against an off-the-shelf store an adversary would actually
meet:

- ``faiss``      IVF flat index persisted to one file; per-list storage is contiguous.
- ``chroma``     persistent HNSW; ``data_level0.bin`` / ``link_lists.bin`` updated incrementally.
- ``lancedb``    IVF-PQ index in a Lance dataset.
- ``sqlitevec``  sqlite-vec virtual table (page-level change granularity).

Embeddings are precomputed by the Windows side (GPU) and passed in as a .npy plus labels, because
torch-cuda lives in the Windows venv; this script only needs numpy and the store libraries.

    # invoked by run_all.sh inside WSL
    python3 run_e8.py --emb emb.npy --labels labels.npy --store faiss --out faiss.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

BLOCK = 4096


def block_hashes(root: Path) -> dict[str, list[str]]:
    """Per-file list of block hashes over the ciphertext tree."""
    out: dict[str, list[str]] = {}
    for f in sorted(root.rglob("*")):
        if f.is_file():
            data = f.read_bytes()
            out[str(f.relative_to(root))] = [
                hashlib.sha1(data[i : i + BLOCK]).hexdigest() for i in range(0, len(data), BLOCK)
            ]
    return out


def changed_blocks(a: dict[str, list[str]], b: dict[str, list[str]]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for name in set(a) | set(b):
        xa, xb = a.get(name, []), b.get(name, [])
        for i in range(max(len(xa), len(xb))):
            if i >= len(xa) or i >= len(xb) or xa[i] != xb[i]:
                out.add((name, i))
    return out


class FaissStore:
    def __init__(self, path: Path, dim: int):
        import faiss

        self.faiss = faiss
        self.path = path
        quant = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quant, dim, 64, faiss.METRIC_INNER_PRODUCT)
        self.dim = dim
        self._trained = False

    def train(self, x):
        self.index.train(x)
        self._trained = True

    def add(self, x):
        self.index.add(x)
        self.faiss.write_index(self.index, str(self.path / "index.faiss"))


class ChromaStore:
    def __init__(self, path: Path, dim: int):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(path))
        self.col = self.client.get_or_create_collection("mem", metadata={"hnsw:space": "ip"})
        self.n = 0

    def train(self, x):
        pass

    def add(self, x):
        ids = [str(self.n + i) for i in range(len(x))]
        self.col.add(ids=ids, embeddings=x.tolist())
        self.n += len(x)


class LanceStore:
    def __init__(self, path: Path, dim: int):
        import lancedb

        self.db = lancedb.connect(str(path))
        self.dim = dim
        self.tbl = None

    def train(self, x):

        data = [{"vector": v.tolist(), "id": i} for i, v in enumerate(x)]
        self.tbl = self.db.create_table("m", data=data)

    def add(self, x):
        if self.tbl is None:
            self.train(x)
            return
        self.tbl.add([{"vector": v.tolist(), "id": -1} for v in x])


class SqliteVecStore:
    def __init__(self, path: Path, dim: int):
        import sqlite3

        import sqlite_vec

        self.db = sqlite3.connect(str(path / "vec.db"))
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.execute(f"CREATE VIRTUAL TABLE m USING vec0(embedding float[{dim}])")
        self.n = 0

    def train(self, x):
        pass

    def add(self, x):
        for v in x:
            self.db.execute(
                "INSERT INTO m(rowid, embedding) VALUES (?, ?)", (self.n, json.dumps(v.tolist()))
            )
            self.n += 1
        self.db.commit()


STORES = {
    "faiss": FaissStore,
    "chroma": ChromaStore,
    "lancedb": LanceStore,
    "sqlitevec": SqliteVecStore,
}


def frequency_labels(by_topic, n_topics, block_index):
    n_blocks = len(block_index)
    rate = np.zeros((n_topics, n_blocks))
    for t, flushes in by_topic.items():
        for blocks in flushes:
            for b in blocks:
                rate[t, block_index[b]] += 1
        if flushes:
            rate[t] /= len(flushes)
    mean = rate.mean(0)
    labels = {}
    for j in range(n_blocks):
        best = int(np.argmax(rate[:, j]))
        if rate[best, j] - mean[j] > 2 * mean[j] and rate[best, j] > 0:
            labels[j] = best
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--store", required=True, choices=list(STORES))
    ap.add_argument("--cipher", required=True, help="mounted plaintext dir inside gocryptfs")
    ap.add_argument("--ciphertext", required=True, help="the ciphertext dir gocryptfs encrypts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=40)
    args = ap.parse_args()

    emb = np.load(args.emb).astype(np.float32)
    labels = np.load(args.labels)
    n_topics = int(labels.max()) + 1
    dim = emb.shape[1]
    n = len(emb)
    hist, post = np.arange(n // 2), np.arange(n // 2, n)

    plain = Path(args.cipher)
    plain.mkdir(parents=True, exist_ok=True)
    cipher = Path(args.ciphertext)
    store = STORES[args.store](plain, dim)
    rng = np.random.default_rng(0)

    # cohort H
    train_x = emb[rng.choice(n, size=min(2000, n), replace=False)]
    store.train(train_x)
    store.add(emb[hist])
    _sync()
    snap0 = block_hashes(cipher)

    # induced writes, one flush per write, snapshot after each
    by_topic: dict[int, list[list]] = {t: [] for t in range(n_topics)}
    all_blocks: set = set()
    prev = snap0
    t0 = time.time()
    for t in range(n_topics):
        pool = post[labels[post] == t]
        picks = (
            rng.choice(pool, size=min(args.budget, len(pool)), replace=False) if len(pool) else []
        )
        for r in picks:
            store.add(emb[r : r + 1])
            _sync()
            snap = block_hashes(cipher)
            ch = changed_blocks(prev, snap)
            by_topic[t].append(list(ch))
            all_blocks |= ch
            prev = snap

    block_index = {b: i for i, b in enumerate(sorted(all_blocks))}
    lab = frequency_labels(by_topic, n_topics, block_index)

    # Score cohort H: a record's "block" is the changed block(s) most associated with it. Since we
    # cannot see which block holds which cohort-H record without inducing, we instead measure the
    # attack's *reach*: fraction of the changed-block map that is confidently labelled, and the
    # label purity against the induced writes themselves (an upper bound on cohort-H recovery for
    # a content-placed store).
    induced_hits = induced_total = 0
    for t, flushes in by_topic.items():
        for blocks in flushes:
            idxs = [block_index[b] for b in blocks if b in block_index]
            votes = [lab[i] for i in idxs if i in lab]
            if votes:
                induced_total += 1
                if max(set(votes), key=votes.count) == t:
                    induced_hits += 1

    result = {
        "store": args.store,
        "dim": dim,
        "n_hist": len(hist),
        "n_topics": n_topics,
        "budget_per_topic": args.budget,
        "block_size": BLOCK,
        "distinct_changed_blocks": len(all_blocks),
        "blocks_confidently_labelled": len(lab),
        "induced_localisation_accuracy": induced_hits / induced_total if induced_total else 0.0,
        "mean_blocks_per_write": float(np.mean([len(b) for f in by_topic.values() for b in f]))
        if all_blocks
        else 0.0,
        "seconds": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def _sync():
    os.sync() if hasattr(os, "sync") else None
    subprocess.run(["sync"], check=False)
    time.sleep(0.05)


if __name__ == "__main__":
    main()
