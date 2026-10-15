"""E18: the deduplication membership oracle against the real mem0 library.

The derived-write attack (\\cref{prop:derived}) and E3 measured a mem0-*style* pipeline.
This runs real mem0 2.2.1, with a local on-disk Qdrant and a local
Ollama model making the add/skip decision, and measures the write-occurrence oracle: the adversary
supplies a fact and sees only whether the vector store's files changed.

Why Windows rather than gocryptfs here: gocryptfs is Linux-only and, as E6 and E16 established, its
ciphertext mirrors the plaintext file structure block for block, so whether a *write occurs* is
identical with or without the encryption layer. The oracle is the occurrence of a write, which the
encryption does not hide, so observing the plaintext Qdrant directory on Windows measures the
same signal a W adversary sees through gocryptfs. mem0 needs the Ollama server, reachable here on
localhost; the WSL path was blocked by the host firewall.

The signal is purely the occurrence of a write: nothing is decrypted. A member fact (already stored)
should be deduplicated by mem0's LLM add/skip stage and write less; a non-member should be added and
write. The AUC of ``the data files changed'' against ``the fact is a non-member'' is the
oracle's strength for the real library's realistic (LLM-decided) arm.

    cd python && python -m eval.exp2.e18_real_mem0 --probes 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from eval.lab import corpus as C

OUT = C.repo_root() / "python" / "eval" / "results" / "e18_real_mem0.json"
OLLAMA = "http://127.0.0.1:11434"


def collection_fingerprint(root: Path) -> int:
    """Number of distinct 4 KiB ciphertext-sized blocks in the vector store's data files.

    The adversary sees the store grow (or not) when mem0 writes (or dedups). We summarize the store
    by its set of block hashes; a write changes the set, a dedup/NOOP leaves it unchanged.
    """
    blocks = set()
    for f in sorted(root.rglob("*")):
        if f.is_file():
            try:
                data = f.read_bytes()
            except OSError:
                continue
            for i in range(0, max(len(data), 1), 4096):
                blocks.add(hashlib.sha1(data[i : i + 4096]).hexdigest())
    return len(blocks)


def make_memory(path: Path, model: str):
    from mem0 import Memory

    cfg = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": str(path / "qdrant"),
                "collection_name": "m",
                "embedding_model_dims": 768,
                "on_disk": True,
            },
        },
        "llm": {"provider": "ollama", "config": {"model": model, "ollama_base_url": OLLAMA}},
        "embedder": {
            "provider": "ollama",
            "config": {"model": "nomic-embed-text", "ollama_base_url": OLLAMA},
        },
        "history_db_path": str(path / "history.db"),
    }
    return Memory.from_config(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=int, default=30, help="member + non-member probes each")
    ap.add_argument("--seed-size", type=int, default=40, help="cohort-H facts seeded before tau")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    rng = np.random.default_rng(20261001)
    idx = rng.permutation(len(rows))
    facts = [rows[i]["text"].split("\n")[0][:200] for i in idx]  # one-line facts

    tmp = Path(tempfile.mkdtemp())
    t0 = time.time()
    try:
        mem = make_memory(tmp / "store", args.model)
        qdir = tmp / "store" / "qdrant"

        # Cohort H: seed member facts before tau.
        members = facts[: args.seed_size]
        for fct in members:
            try:
                mem.add(fct, user_id="u")
            except Exception as e:
                print("seed add failed:", e, flush=True)

        # Probes: members (exact repeats of seeded facts) and non-members (novel), interleaved.
        n = args.probes
        member_probes = [(f, 1) for f in members[:n]]
        nonmember_probes = [(f, 0) for f in facts[args.seed_size : args.seed_size + n]]
        probes = member_probes + nonmember_probes
        rng.shuffle(probes)

        y, wrote, events = [], [], []
        for fct, is_member in probes:
            before = collection_fingerprint(qdir)
            ev = "?"
            try:
                r = mem.add(fct, user_id="u")
                res = r.get("results", []) if isinstance(r, dict) else []
                ev = ",".join(sorted({str(x.get("event", "?")) for x in res})) or "NONE"
            except Exception as e:
                ev = f"ERR:{type(e).__name__}"
            after = collection_fingerprint(qdir)
            y.append(1 - is_member)  # adversary predicts "non-member"; write => non-member
            wrote.append(1 if after > before else 0)
            events.append(ev)

        y_arr = np.array(y)
        w_arr = np.array(wrote, dtype=float)
        auc = (
            float(roc_auc_score(y_arr, w_arr))
            if len(np.unique(y_arr)) > 1 and len(np.unique(w_arr)) > 1
            else 0.5
        )
        # Bootstrap CI over probes.
        boots = []
        for _ in range(2000):
            s = rng.integers(0, len(y_arr), len(y_arr))
            if len(np.unique(y_arr[s])) > 1 and len(np.unique(w_arr[s])) > 1:
                boots.append(roc_auc_score(y_arr[s], w_arr[s]))
        lo, hi = (
            (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
            if boots
            else (auc, auc)
        )

        member_write = float(
            np.mean([w for w, (_, m) in zip(wrote, probes, strict=True) if m == 1])
        )
        nonmember_write = float(
            np.mean([w for w, (_, m) in zip(wrote, probes, strict=True) if m == 0])
        )
        result = {
            "experiment": "E18",
            "library": "mem0 2.2.1",
            "model": args.model,
            "embedder": "nomic-embed-text",
            "vector_store": "qdrant (on-disk, local)",
            "note": "write-occurrence oracle; gocryptfs mirrors these files block-for-block",
            "seed_size": args.seed_size,
            "probes_each": n,
            "auc_write_vs_nonmember": auc,
            "lo": lo,
            "hi": hi,
            "member_write_rate": member_write,
            "nonmember_write_rate": nonmember_write,
            "event_counts": {e: events.count(e) for e in sorted(set(events))},
            "seconds": time.time() - t0,
        }
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps(result, indent=1))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
