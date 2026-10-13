"""E30: the deduplication membership oracle in a second framework, LangMem, run end to end.

The source census (supplementary material) records four frameworks whose writes depend on what is
already stored; E18 measured one of them, mem0, as shipped. This runs LangMem 0.0.30's memory store
manager -- the component the census cites (``langmem/knowledge/extraction.py``) -- with a local
Ollama model through its OpenAI-compatible endpoint and nomic embeddings. LangMem leaves the
backing store to the caller; this one is LangGraph's ``InMemoryStore`` with every ``put`` also
appended to a file on disk, which is what a persistent ``BaseStore`` does and what the adversary
observes. The manager searches the store, asks the model what to insert or update, and writes only
memories that are new or changed, so a fact it already holds should produce no write.

The adversary is E18's: it supplies a fact and sees whether, and how much of, the store file
changed. LangMem writes synchronously within the call, so the observation brackets the call.

    cd python && ..\\.venv-fw\\Scripts\\python -m eval.exp2.e30_langmem
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from eval.exp2.oracle_common import (
    OLLAMA,
    DirView,
    auc_ci,
    changed_blocks,
    facts,
    ollama_embed,
    ollama_ready,
    write_json,
)
from eval.lab import corpus as C

OUT = C.repo_root() / "python" / "eval" / "results" / "e30_langmem.json"
EMBED = "nomic-embed-text"


def make_store(path: Path):
    from langgraph.store.base import PutOp
    from langgraph.store.memory import InMemoryStore

    class DiskBackedStore(InMemoryStore):
        """InMemoryStore whose every put is also appended, durably, to one file on disk."""

        def __init__(self, file: Path, **kw):
            super().__init__(**kw)
            self.file = file
            self._io = threading.Lock()

        def batch(self, ops):
            ops = list(ops)
            res = super().batch(ops)
            puts = [op for op in ops if isinstance(op, PutOp)]
            if puts:
                with self._io, self.file.open("ab") as fh:
                    for op in puts:
                        rec = {"ns": list(op.namespace), "key": op.key, "value": op.value}
                        fh.write((json.dumps(rec, default=str) + "\n").encode())
                    fh.flush()
                    os.fsync(fh.fileno())
            return res

    path.mkdir(parents=True, exist_ok=True)
    return DiskBackedStore(
        path / "memories.jsonl",
        index={"dims": 768, "embed": lambda texts: ollama_embed(list(texts), EMBED)},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=int, default=40)
    ap.add_argument("--seed-size", type=int, default=40)
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    args = ap.parse_args()

    import langmem
    from langchain_openai import ChatOpenAI
    from langmem import create_memory_store_manager

    ollama_ready([args.model, EMBED])
    llm = ChatOpenAI(
        model=args.model, base_url=f"{OLLAMA}/v1", api_key="ollama", temperature=0, timeout=300
    )
    pool = facts()
    rng = np.random.default_rng(30_000)
    work = Path(tempfile.mkdtemp(prefix="e30-"))
    t0 = time.time()
    try:
        store = make_store(work / "observed")
        manager = create_memory_store_manager(llm, namespace=("memories",), store=store)
        view = DirView([work / "observed"])

        def remember(fact: str):
            return manager.invoke({"messages": [{"role": "user", "content": f"Remember: {fact}"}]})

        members = pool[: args.seed_size]
        seed_errors = 0
        for f in members:
            try:
                remember(f)
            except Exception as e:
                seed_errors += 1
                print("seed failed:", type(e).__name__, str(e)[:200], flush=True)
        print(f"seeded {len(members)} facts ({seed_errors} errors)", flush=True)

        probes = [(f, 1) for f in members[: args.probes]] + [
            (f, 0) for f in pool[args.seed_size : args.seed_size + args.probes]
        ]
        rng.shuffle(probes)
        rows = []
        log = work / "observed" / "memories.jsonl"

        def size() -> int:
            return log.stat().st_size if log.exists() else 0

        for k, (f, is_member) in enumerate(probes):
            before = view.snapshot()
            size0 = size()
            s = time.time()
            n_puts, err = -1, None
            try:
                n_puts = len(remember(f))
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:120]}"
            after = view.snapshot()
            n = changed_blocks(before, after)
            rows.append(
                {
                    "member": is_member,
                    "bytes_added": size() - size0,
                    "changed_blocks": n,
                    "any": int(n > 0),
                    "puts": n_puts,
                    "latency": time.time() - s,
                    "error": err,
                }
            )
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{len(probes)} probes", flush=True)

        ok = [r for r in rows if r["error"] is None]
        y = [1 - r["member"] for r in ok]
        auc_n, lo_n, hi_n = auc_ci(y, [r["changed_blocks"] for r in ok], rng)
        auc_b, lo_b, hi_b = auc_ci(y, [r["any"] for r in ok], rng)
        # The file's size, which gocryptfs preserves to within a block: a patch rewrites an
        # existing memory, an insertion adds one, and the two append different amounts.
        auc_s, lo_s, hi_s = auc_ci(y, [r["bytes_added"] for r in ok], rng)
        out = {
            "experiment": "E30",
            "library": f"langmem {getattr(langmem, '__version__', '0.0.30')}",
            "component": "create_memory_store_manager (langmem/knowledge/extraction.py)",
            "model": args.model,
            "embedder": EMBED,
            "store": "InMemoryStore with every put appended durably to one file",
            "seed_size": args.seed_size,
            "seed_errors": seed_errors,
            "probes_each": args.probes,
            "probe_errors": len(rows) - len(ok),
            "auc_blocks": auc_n,
            "lo_blocks": lo_n,
            "hi_blocks": hi_n,
            "auc_bytes": auc_s,
            "lo_bytes": lo_s,
            "hi_bytes": hi_s,
            "auc_any": auc_b,
            "lo_any": lo_b,
            "hi_any": hi_b,
            "member_write_rate": float(np.mean([r["any"] for r in ok if r["member"] == 1])),
            "nonmember_write_rate": float(np.mean([r["any"] for r in ok if r["member"] == 0])),
            "rows": rows,
            "seconds": time.time() - t0,
        }
        write_json(OUT, out)
        print(
            f"E30 LangMem: AUC(any) {auc_b:.3f} [{lo_b:.3f},{hi_b:.3f}], member write "
            f"{out['member_write_rate']:.2f} vs non-member {out['nonmember_write_rate']:.2f}",
            flush=True,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
