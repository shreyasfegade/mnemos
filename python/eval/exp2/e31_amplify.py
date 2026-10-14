"""E31: amplifying the mem0 oracle by re-probing, and a second model making the add/skip decision.

E18 probes each fact once and reads one bit, at AUC 0.80 with qwen2.5. Two questions remain.

**Is the oracle model-specific?** The decision is the model's, so E31 repeats the probe protocol
with a second, independently trained model, ``llama3.1:8b``, making it.

**Can the adversary amplify?** A probe is not free to repeat: a non-member probe usually writes,
and after that write the fact *is* a member. But a probe that writes nothing leaves the store
unchanged, so it can be repeated. The adversary therefore probes sequentially: the fact itself, then
up to ``K - 1`` paraphrases (written by the adversary's own model, cached), stopping at the first
write. Its statistic is the number of probes that wrote nothing before the first write; a member
should accumulate more of them. The single-probe AUC is the same protocol truncated at one probe, on
the same facts, so the comparison is paired.

Observation is E18's: the Qdrant directory, bracketing each call (mem0's Qdrant writes are
synchronous).

    cd python && ..\\.venv-fw\\Scripts\\python -m eval.exp2.e31_amplify --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "False")

import numpy as np

from eval.exp2.oracle_common import (
    DirView,
    auc_ci,
    changed_blocks,
    facts,
    ollama_chat,
    ollama_ready,
    write_json,
)
from eval.lab import corpus as C

RES = C.repo_root() / "python" / "eval" / "results"
OUT_DIR = RES / "e31"
OUT = RES / "e31_amplify.json"
PARA = OUT_DIR / "paraphrases.json"
EMBED = "nomic-embed-text"
PARA_MODEL = "qwen2.5:7b-instruct"


def paraphrases(fs: list[str], k: int) -> dict[str, list[str]]:
    """``k`` paraphrases per fact from the adversary's model, cached across runs and models."""
    cache: dict[str, list[str]] = (
        json.loads(PARA.read_text(encoding="utf-8")) if PARA.exists() else {}
    )
    todo = [f for f in fs if len(cache.get(f, [])) < k]
    if todo:
        ollama_ready([PARA_MODEL])  # loaded only when something is missing from the cache
    for n, f in enumerate(todo):
        got = cache.get(f, [])
        while len(got) < k:
            p = ollama_chat(
                "Rewrite this fact in different words, keeping its meaning exactly. Reply with "
                f"the rewritten sentence only.\n\nFact: {f}",
                PARA_MODEL,
                temperature=0.8,
                seed=len(got),
            ).strip()
            got.append(p.splitlines()[0][:300] if p else f)
        cache[f] = got
        if (n + 1) % 20 == 0:
            write_json(PARA, cache)
    write_json(PARA, cache)
    return cache


def make_memory(path: Path, model: str):
    from mem0 import Memory

    return Memory.from_config(
        {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": str(path / "qdrant"),
                    "collection_name": "m",
                    "embedding_model_dims": 768,
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
            "history_db_path": str(path / "history.db"),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--probes", type=int, default=40)
    ap.add_argument("--seed-size", type=int, default=40)
    ap.add_argument("--k", type=int, default=3, help="maximum probes per fact")
    ap.add_argument("--paraphrase-only", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
        return

    pool = facts()
    members = pool[: args.seed_size]
    nonmembers = pool[args.seed_size : args.seed_size + args.probes]
    para = paraphrases(members[: args.probes] + nonmembers, args.k - 1)
    if args.paraphrase_only:
        return
    ollama_ready([args.model, EMBED])

    rng = np.random.default_rng(31_000)
    work = Path(tempfile.mkdtemp(prefix="e31-"))
    t0 = time.time()
    try:
        mem = make_memory(work / "store", args.model)
        view = DirView([work / "store" / "qdrant"])
        for f in members:
            try:
                mem.add(f, user_id="u")
            except Exception as e:
                print("seed add failed:", type(e).__name__, e, flush=True)
        probes = [(f, 1) for f in members[: args.probes]] + [(f, 0) for f in nonmembers]
        rng.shuffle(probes)
        rows = []
        for n, (f, is_member) in enumerate(probes):
            seq = [f, *para.get(f, [])[: args.k - 1]]
            outcomes = []
            for text in seq:
                before = view.snapshot()
                try:
                    mem.add(text, user_id="u")
                except Exception as e:
                    print("probe failed:", type(e).__name__, e, flush=True)
                wrote = changed_blocks(before, view.snapshot()) > 0
                outcomes.append(int(wrote))
                if wrote:
                    break
            quiet = 0
            for w in outcomes:
                if w:
                    break
                quiet += 1
            rows.append({"member": is_member, "outcomes": outcomes, "quiet_before_write": quiet})
            if (n + 1) % 10 == 0:
                print(f"  {args.model}: {n + 1}/{len(probes)} facts", flush=True)

        y = [1 - r["member"] for r in rows]
        # More writing means non-member: score a write on the first probe, and fewer quiet probes.
        s1 = [r["outcomes"][0] for r in rows]
        sk = [-r["quiet_before_write"] for r in rows]
        a1 = auc_ci(y, s1, rng)
        ak = auc_ci(y, sk, rng)
        # Paired difference, bootstrapped over facts.
        from sklearn.metrics import roc_auc_score

        diffs = []
        ya, s1a, ska = np.array(y), np.array(s1, float), np.array(sk, float)
        for _ in range(2000):
            i = rng.integers(0, len(ya), len(ya))
            if len(np.unique(ya[i])) < 2:
                continue
            f1 = roc_auc_score(ya[i], s1a[i]) if len(np.unique(s1a[i])) > 1 else 0.5
            fk = roc_auc_score(ya[i], ska[i]) if len(np.unique(ska[i])) > 1 else 0.5
            diffs.append(fk - f1)
        out = {
            "experiment": "E31",
            "library": "mem0 2.2.1",
            "model": args.model,
            "paraphrase_model": PARA_MODEL,
            "k": args.k,
            "seed_size": args.seed_size,
            "probes_each": args.probes,
            "auc_k1": a1[0],
            "lo_k1": a1[1],
            "hi_k1": a1[2],
            "auc_k": ak[0],
            "lo_k": ak[1],
            "hi_k": ak[2],
            "auc_gain": ak[0] - a1[0],
            "auc_gain_lo": float(np.percentile(diffs, 2.5)),
            "auc_gain_hi": float(np.percentile(diffs, 97.5)),
            "member_write_first": float(np.mean([r["outcomes"][0] for r in rows if r["member"]])),
            "nonmember_write_first": float(
                np.mean([r["outcomes"][0] for r in rows if not r["member"]])
            ),
            "mean_probes_per_fact": float(np.mean([len(r["outcomes"]) for r in rows])),
            "rows": rows,
            "seconds": time.time() - t0,
        }
        tag = args.model.replace(":", "_").replace("/", "_")
        write_json(OUT_DIR / f"{tag}.json", out)
        print(
            f"E31 {args.model}: AUC k=1 {a1[0]:.3f} [{a1[1]:.3f},{a1[2]:.3f}], "
            f"k={args.k} {ak[0]:.3f} [{ak[1]:.3f},{ak[2]:.3f}]",
            flush=True,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    aggregate()


def aggregate() -> None:
    models = {}
    for f in sorted(OUT_DIR.glob("*.json")):
        if f.name == "paraphrases.json":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        d.pop("rows", None)
        models[d["model"]] = d
    write_json(OUT, {"experiment": "E31", "models": models})
    print(f"wrote {OUT.relative_to(C.repo_root())} with {sorted(models)}")


if __name__ == "__main__":
    main()
