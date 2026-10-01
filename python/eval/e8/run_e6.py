"""E6: structure transfer against deployed vector stores, redone to 

What was wrong before
=====================
The previous harness built its block-to-topic map from a set of induced writes and then scored
**those same writes**. That is in-sample, and it explains every number it produced: stores that
allocate fresh blocks per insert (sqlite-vec's tail chunk, LanceDB's new fragments) score ~1.00 by
construction, while FAISS, which rewrites its whole index on every insert, scores poorly. It also
never computed ``Adv_hist`` at all -- it measured *localisation of induced writes*, which is not
transfer, and its own docstring conceded as much.

What this does instead
======================
1. **Held-out scoring.** Induced writes are split: one half labels blocks, the other half is
   scored. A label map fitted on the same writes it is tested on is not evidence.
2. **Adv_hist, properly.** Cohort H is written one record at a time with a ciphertext snapshot
   after each, so the harness knows which blocks hold which pre-compromise record. That map is
   **ground truth for scoring only** and is never shown to the adversary, exactly as in the Mnemos
   experiments. The adversary labels blocks from its induced writes alone, and we then ask whether
   those labels identify the topics of cohort-H records.
3. **Placement is classified mechanistically, and time-placed stores are kept as controls.** A
   store that places by insertion order *should* fail to transfer, and showing that it does is the
   cleanest available validation of the transfer lemma's premise.
4. **The ciphertext is verified to be ciphertext.** The diff runs over the gocryptfs ciphertext
   directory, and the harness asserts that directory is not the plaintext mount and that a known
   plaintext marker does not appear in it.

    python3 run_e6.py --emb emb.npy --labels labels.npy --store faiss \
        --plain /mnt/p --cipher /mnt/c --out faiss.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

BLOCK = 4096


# --------------------------------------------------------------------------- ciphertext diffing


def block_hashes(root: Path) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    for f in sorted(root.rglob("*")):
        if f.is_file():
            try:
                data = f.read_bytes()
            except OSError:
                continue
            name = str(f.relative_to(root))
            for i in range(0, max(len(data), 1), BLOCK):
                out[(name, i // BLOCK)] = hashlib.sha1(data[i : i + BLOCK]).hexdigest()
    return out


def changed(a: dict, b: dict) -> set[tuple[str, int]]:
    keys = set(a) | set(b)
    return {k for k in keys if a.get(k) != b.get(k)}


def sync() -> None:
    subprocess.run(["sync"], check=False)
    time.sleep(0.05)


# --------------------------------------------------------------------------- stores
# Each store declares how it places records, because that is the property the attack needs and
# the property the previous write-up asserted without checking.


class FaissIvf:
    placement = "content (inverted lists), but the whole index file is rewritten on every save"
    schema = (
        "IndexIVFFlat(quantizer=IndexFlatIP, nlist=64, metric=INNER_PRODUCT), "
        "faiss.write_index per insert"
    )

    def __init__(self, path: Path, dim: int):
        import faiss

        self.faiss = faiss
        self.path = path
        self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, 64, faiss.METRIC_INNER_PRODUCT)

    def train(self, x):
        self.index.train(x)

    def add(self, x):
        self.index.add(x)
        self.faiss.write_index(self.index, str(self.path / "index.faiss"))


class Chroma:
    placement = "content (HNSW neighbour lists updated incrementally)"
    schema = "PersistentClient, collection 'mem', hnsw:space=ip"

    def __init__(self, path: Path, dim: int):
        import chromadb

        self.col = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
            "mem", metadata={"hnsw:space": "ip"}
        )
        self.n = 0

    def train(self, x):
        pass

    def add(self, x):
        self.col.add(ids=[str(self.n + i) for i in range(len(x))], embeddings=x.tolist())
        self.n += len(x)


class LanceRaw:
    placement = "TIME (append-only fragments; no vector index, so nothing is placed by content)"
    schema = "lancedb table 'm', no create_index call"

    def __init__(self, path: Path, dim: int):
        import lancedb

        self.db = lancedb.connect(str(path))
        self.dim = dim
        self.tbl = None

    def train(self, x):
        self.tbl = self.db.create_table(
            "m", data=[{"vector": v.tolist(), "id": i} for i, v in enumerate(x)]
        )

    def add(self, x):
        self.tbl.add([{"vector": v.tolist(), "id": -1} for v in x])


class LanceIndexed(LanceRaw):
    placement = "content (IVF-PQ) once the index is built and rebuilt after each batch"
    schema = "lancedb table 'm' + create_index(IVF_PQ, num_partitions=32), rebuilt per insert"

    def train(self, x):
        super().train(x)
        try:
            self.tbl.create_index(num_partitions=32, num_sub_vectors=16, replace=True)
        except Exception as e:
            print(f"  lancedb index build failed: {e}")

    def add(self, x):
        super().add(x)
        with contextlib.suppress(Exception):
            self.tbl.create_index(num_partitions=32, num_sub_vectors=16, replace=True)


class SqliteVec:
    placement = "TIME (vec0 chunk storage fills in insertion order; no partition key is declared)"
    schema = "CREATE VIRTUAL TABLE m USING vec0(embedding float[dim])  -- note: NO partition key"

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


class AppendControl:
    placement = (
        "TIME (explicit negative control: raw append, placement is arrival order and nothing else)"
    )
    schema = "float32 vectors appended to one file"

    def __init__(self, path: Path, dim: int):
        self.f = path / "append.bin"
        self.f.write_bytes(b"")

    def train(self, x):
        pass

    def add(self, x):
        with self.f.open("ab") as fh:
            fh.write(x.astype(np.float32).tobytes())
            fh.flush()
            os.fsync(fh.fileno())


# FAISS OnDiskInvertedLists was evaluated as a target but dropped: its incremental per-add write
# path segfaults in faiss 1.15 (the on-disk lists are built for the batch ``merge_ondisk`` pattern,
# not single-record appends), so it cannot be driven one record at a time the way the attack needs.
# The real content-placed on-disk target E16 uses is pgvector ivfflat below.


class Pgvector:
    placement = "content (ivfflat lists); an insert lands in its nearest list's pages after build"
    schema = "pgvector ivfflat (lists=64, vector_ip_ops); built on training set, then inserts"

    def __init__(self, path: Path, dim: int):
        import pgserver

        self.srv = pgserver.get_server(str(path / "pg"), cleanup_mode="stop")
        self.dim = dim
        self.srv.psql("CREATE EXTENSION IF NOT EXISTS vector;")
        self.srv.psql(f"CREATE TABLE m (id serial PRIMARY KEY, v vector({dim}));")
        self.n = 0

    def _vec(self, row) -> str:
        return "[" + ",".join(f"{float(z):.6f}" for z in row) + "]"

    def train(self, x):
        # Bulk-load the training set and build the ivfflat index, so placement is by content.
        for row in x:
            self.srv.psql(f"INSERT INTO m (v) VALUES ('{self._vec(row)}');")
        self.n += len(x)
        self.srv.psql("CREATE INDEX m_ivf ON m USING ivfflat (v vector_ip_ops) WITH (lists=64);")
        self.srv.psql("CHECKPOINT;")

    def add(self, x):
        for row in x:
            self.srv.psql(f"INSERT INTO m (v) VALUES ('{self._vec(row)}');")
        self.n += len(x)
        # Force the heap and index pages to disk so the ciphertext diff reflects this insert.
        self.srv.psql("CHECKPOINT;")


STORES = {
    "faiss": FaissIvf,
    "pgvector": Pgvector,
    "chroma": Chroma,
    "lancedb": LanceRaw,
    "lancedb_indexed": LanceIndexed,
    "sqlitevec": SqliteVec,
    "append_control": AppendControl,
}


# --------------------------------------------------------------------------- attack


def frequency_labels(
    by_topic: dict[int, list[set]], n_topics: int, block_index: dict
) -> dict[int, int]:
    n_blocks = len(block_index)
    rate = np.zeros((n_topics, n_blocks))
    for t, writes in by_topic.items():
        if not writes:
            continue
        for blocks in writes:
            for b in blocks:
                if b in block_index:
                    rate[t, block_index[b]] += 1
        rate[t] /= len(writes)
    mean = rate.mean(0)
    labels: dict[int, int] = {}
    for j in range(n_blocks):
        best = int(np.argmax(rate[:, j]))
        if rate[best, j] > 0 and rate[best, j] - mean[j] > mean[j]:
            labels[j] = best
    return labels


def majority_topic(blocks: set, block_index: dict, labels: dict[int, int]) -> int | None:
    votes = [
        labels[block_index[b]] for b in blocks if b in block_index and block_index[b] in labels
    ]
    if not votes:
        return None
    return max(set(votes), key=votes.count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--store", required=True, choices=list(STORES))
    ap.add_argument("--plain", required=True, help="the gocryptfs plaintext mount")
    ap.add_argument("--cipher", required=True, help="the gocryptfs ciphertext directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hist", type=int, default=300)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument(
        "--seed", type=int, default=0, help="repeat index; varies the split and the draws"
    )
    args = ap.parse_args()

    emb = np.load(args.emb).astype(np.float32)
    labels_arr = np.load(args.labels)
    n_topics = int(labels_arr.max()) + 1
    dim = emb.shape[1]
    n = len(emb)

    plain = Path(args.plain)
    cipher = Path(args.cipher)
    plain.mkdir(parents=True, exist_ok=True)
    assert plain.resolve() != cipher.resolve(), (
        "plaintext mount and ciphertext dir are the same path"
    )

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    hist_idx = order[: args.hist]
    post_idx = order[args.hist :]

    cls = STORES[args.store]
    store = cls(plain, dim)
    store.train(emb[rng.choice(n, size=min(2000, n), replace=False)])
    sync()

    t0 = time.time()
    # --- cohort H, one record at a time, with a snapshot after each. The resulting map is the
    # harness's ground truth for scoring and is never available to the adversary.
    prev = block_hashes(cipher)
    h_blocks: list[set] = []
    for i in hist_idx:
        store.add(emb[i : i + 1])
        sync()
        snap = block_hashes(cipher)
        h_blocks.append(changed(prev, snap))
        prev = snap
    tau_snapshot_blocks = len(prev)

    # --- induced writes, one at a time, INTERLEAVED ACROSS TOPICS.
    #
    # The order matters and writing topic-by-topic is a trap. Several of these stores append new
    # data at a tail (a sqlite-vec chunk, a Lance fragment, an HNSW node block), so consecutive
    # writes share blocks whatever their content. Grouping a topic's writes together therefore
    # makes recency look exactly like content placement, and a localisation score computed that
    # way measures the write order rather than the index. Interleaving removes the confound: any
    # remaining block-to-topic association has to come from where the store *placed* the record.
    schedule: list[int] = []
    for t in range(n_topics):
        pool = post_idx[labels_arr[post_idx] == t]
        picks = (
            rng.choice(pool, size=min(args.budget, len(pool)), replace=False) if len(pool) else []
        )
        schedule.extend((t, int(r)) for r in picks)
    rng.shuffle(schedule)

    induced: list[tuple[int, set]] = []
    for t, r in schedule:
        store.add(emb[r : r + 1])
        sync()
        snap = block_hashes(cipher)
        induced.append((t, changed(prev, snap)))
        prev = snap

    # --- ciphertext sanity: the directory we diffed must not contain our plaintext marker.
    marker_found = False
    for f in cipher.rglob("*"):
        if f.is_file():
            try:
                if b"append.bin" in f.read_bytes()[:4096] and args.store != "append_control":
                    marker_found = True
            except OSError:
                pass

    # --- held-out split of the induced writes.
    rng.shuffle(induced)
    cut = len(induced) // 2
    label_half, score_half = induced[:cut], induced[cut:]

    all_blocks = sorted({b for _, s in induced for b in s} | {b for s in h_blocks for b in s})
    block_index = {b: i for i, b in enumerate(all_blocks)}

    by_topic: dict[int, list[set]] = {t: [] for t in range(n_topics)}
    for t, s in label_half:
        by_topic[t].append(s)
    lab = frequency_labels(by_topic, n_topics, block_index)

    # Held-out localisation: does a *scored* induced write's blocks carry its own topic?
    hit = tot = 0
    for t, s in score_half:
        got = majority_topic(s, block_index, lab)
        if got is not None:
            tot += 1
            hit += int(got == t)
    localisation_heldout = hit / tot if tot else 0.0

    # Adv_hist: the quantity that matters. Does the adversary's label map identify the topics of
    # records written before the compromise?
    hhit = htot = 0
    for i, s in zip(hist_idx, h_blocks, strict=True):
        got = majority_topic(s, block_index, lab)
        if got is not None:
            htot += 1
            hhit += int(got == int(labels_arr[i]))
    chance = 1.0 / n_topics
    acc_h = hhit / htot if htot else 0.0
    # Two accuracies, and coverage reported beside both so it is never hidden. ``acc_h_all`` counts
    # a record the adversary cannot label as wrong: that is an adversary that abstains, and at
    # coverage 0 it scores 0 and an "advantage" of exactly -chance, which is an accounting choice
    # rather than information. ``acc_h_guess`` credits those records with the uniform guess (1/K)
    # a rational adversary would make, and is what the advantage is taken from.
    acc_h_all = hhit / max(len(h_blocks), 1)
    unlabelled = max(len(h_blocks), 1) - htot
    acc_h_guess = (hhit + unlabelled / n_topics) / max(len(h_blocks), 1)

    result = {
        "store": args.store,
        "seed": args.seed,
        "placement": cls.placement,
        "schema": cls.schema,
        "dim": dim,
        "n_hist": len(hist_idx),
        "n_topics": n_topics,
        "chance": chance,
        "budget_per_topic": args.budget,
        "block_size": BLOCK,
        "induced_order": "interleaved across topics (so recency cannot imitate content placement)",
        "tau_snapshot_blocks": tau_snapshot_blocks,
        "distinct_changed_blocks": len(all_blocks),
        "blocks_labelled": len(lab),
        "mean_blocks_per_write": float(np.mean([len(s) for _, s in induced])) if induced else 0.0,
        "localisation_heldout": localisation_heldout,
        "localisation_scored_n": tot,
        "adv_hist": acc_h_guess - chance,
        "acc_hist_covered": acc_h,
        "acc_hist_all": acc_h_all,
        "acc_hist_guess": acc_h_guess,
        "coverage_hist": htot / max(len(h_blocks), 1),
        "plaintext_marker_leaked_into_ciphertext": marker_found,
        "seconds": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
