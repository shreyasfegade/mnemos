"""E3: deduplication-based membership inference against a self-writing agent.

This attack makes the derived-write proposition
concrete instead of circular. The earlier measurement -- "50 of 50 follow-ups landed in the
retrieved memory's partition" -- held by construction, because the agent had been told to write
about the retrieved subject. This one does not.

The attack
==========
A mem0-style memory pipeline does not write a fact it already holds: the update stage returns NOOP.
So an adversary who supplies a document and can see *whether a write happened* learns whether a
matching fact was already in memory. Since the adversary chooses the document, it chooses the fact,
and a fact that was in memory before τ is a **cohort-H membership** disclosure. Nothing about the
record's content is decrypted; the signal is purely the occurrence of a write.

Why it bites Mnemos-P and not Mnemos-M
======================================
Under an on-demand flush, a real write produces a flush and a NOOP produces nothing, so the
occurrence of a flush *is* the answer. Under Mnemos-M's clocked flush, a flush happens every Δ
whether or not the agent wrote, so the observable is identical in both cases and the channel is
closed. This is the separation Proposition 3 should be stating: it is not that no storage design
can close the derived-write channel, but that no storage design whose operations depend on the
content or the *occurrence* of operations can.

Two dedup mechanisms are measured, because production pipelines use both:

``llm``        the real mem0-style update stage, a local model deciding ADD / UPDATE / NOOP.
``threshold``  cosine-similarity dedup above a fixed threshold, which many pipelines use instead.

    cd python && python -m eval.exp2.e3_dedup --probes 80
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from eval.agent.llm import available
from eval.agent.memory_agent import UPDATE_SYSTEM
from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e3_dedup.json"

B = 32
CAP = 32
DIM = 768
BLOCK = 1024
N_HIST = 120
DEDUP_THRESHOLD = 0.95
TOP_K_NEIGHBOURS = 5


def trace_events(trace: Path, offset: int) -> tuple[int, int, int]:
    """(flushes, region writes, new offset) since ``offset``."""
    if not trace.exists():
        return 0, 0, offset
    with trace.open("r", encoding="utf-8") as fh:
        fh.seek(offset)
        lines = [line for line in fh if line.strip()]
    flushes = writes = 0
    for line in lines:
        e = json.loads(line)
        if e["kind"] == "epoch_flush":
            flushes += 1
        elif e["kind"] == "region_write":
            writes += 1
    return flushes, writes, trace.stat().st_size


def decide_threshold(vec, stored: np.ndarray) -> bool:
    """True = write. A fact too similar to something stored is deduplicated away."""
    if stored.size == 0:
        return True
    return float(np.max(stored @ vec)) < DEDUP_THRESHOLD


def decide_llm(llm, text: str, neighbours: list[str]) -> bool:
    if not neighbours:
        return True
    listing = "\n".join(f"{i}: {t}" for i, t in enumerate(neighbours))
    obj = llm.chat_json(UPDATE_SYSTEM, f"Candidate fact:\n{text}\n\nExisting memories:\n{listing}")
    if not isinstance(obj, dict):
        return True
    return str(obj.get("action", "ADD")).upper() != "NOOP"


def run(engine_kind: str, dedup: str, llm, texts, vecs, cents, probes, rng) -> dict:
    """Returns the adversary's per-probe observation and the ground-truth membership."""
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.base_overrides(
            dim=DIM,
            buckets=B,
            capacity=CAP,
            block=BLOCK,
            # One record per flush is the sparse regime a real agent produces, and it is the
            # regime in which the occurrence of a flush is most informative.
            epoch_writes=1,
            probe_spread=1,
        )
        eng, _, _ = lab.create(engine_kind, tmp / "s", cents, over, {})

        # Cohort H: the facts the agent already holds.
        member_idx = probes["member"]
        stored_vecs = []
        stored_texts = []
        hist = list(member_idx) + list(probes["filler"])
        rng.shuffle(hist)
        for i in hist:
            try:
                eng.put(texts[i][:900], vecs[i].tolist(), "{}")
                stored_vecs.append(vecs[i])
                stored_texts.append(texts[i][:900])
            except Exception:
                pass
        lab.drain(eng)
        eng.flush()
        stored = (
            np.array(stored_vecs, dtype=np.float32)
            if stored_vecs
            else np.zeros((0, DIM), np.float32)
        )

        trace = tmp / "t.jsonl"
        trace.write_text("", encoding="utf-8")
        eng.set_trace_sink(str(trace))
        offset = 0

        obs, truth, wrote_flags = [], [], []
        order = [(i, 1) for i in probes["member"]] + [(i, 0) for i in probes["nonmember"]]
        rng.shuffle(order)
        for i, is_member in order:
            text = texts[i][:900]
            vec = vecs[i]
            if dedup == "threshold":
                write = decide_threshold(vec, stored)
            else:
                sims = stored @ vec if stored.size else np.zeros(0)
                nb = (
                    [stored_texts[j] for j in np.argsort(-sims)[:TOP_K_NEIGHBOURS]]
                    if sims.size
                    else []
                )
                write = decide_llm(llm, text, nb)
            if write:
                try:
                    eng.put(text, vec.tolist(), "{}")
                    stored = np.vstack([stored, vec[None, :]]) if stored.size else vec[None, :]
                    stored_texts.append(text)
                except Exception:
                    pass
            # The clock fires regardless of whether the agent wrote. For Mnemos-M that is the
            # design; for Mnemos-P the flush only happens because a record was staged.
            if engine_kind == "m":
                eng.flush()
            flushes, _region_writes, offset = trace_events(trace, offset)
            obs.append(float(flushes))
            truth.append(is_member)
            wrote_flags.append(1 if write else 0)

        with contextlib.suppress(Exception):
            eng.close()

        o = np.array(obs)
        t = np.array(truth)
        wf = np.array(wrote_flags)
        # A member should be deduplicated, so a member should produce *fewer* observed flushes.
        auc = (
            float(roc_auc_score(t, -o)) if len(np.unique(t)) > 1 and len(np.unique(o)) > 1 else 0.5
        )
        # The interval resamples probes. Where the observation cannot vary at all -- Mnemos-M's
        # clock flushes exactly once per probe -- every resample returns 0.5 and the interval is
        # a point; that is the channel being closed, and the verifier and the ledger treat a
        # zero-width interval as such rather than as infinite precision.
        boot_rng = np.random.default_rng(31)
        boots = []
        for _ in range(4000):
            ix = boot_rng.integers(0, len(t), len(t))
            if len(np.unique(t[ix])) < 2 or len(np.unique(o[ix])) < 2:
                boots.append(0.5)
                continue
            boots.append(float(roc_auc_score(t[ix], -o[ix])))
        lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
        # The accuracy of the adversary's fixed rule, "no write happened, so the fact was already
        # there". Nothing is fitted, so nothing is optimistic: against a clocked flush every probe
        # sees a write and the rule is at chance by construction.
        pred_member = o == 0
        acc = float(0.5 * ((pred_member[t == 1]).mean() + (~pred_member[t == 0]).mean()))
        return {
            "engine": engine_kind,
            "dedup": dedup,
            "auc": auc,
            "lo": lo,
            "hi": hi,
            "accuracy": acc,
            "advantage": abs(auc - 0.5),
            "write_rate_member": float(wf[t == 1].mean()) if (t == 1).any() else float("nan"),
            "write_rate_nonmember": float(wf[t == 0].mean()) if (t == 0).any() else float("nan"),
            "observed_flushes_member": float(o[t == 1].mean()) if (t == 1).any() else float("nan"),
            "observed_flushes_nonmember": float(o[t == 0].mean())
            if (t == 0).any()
            else float("nan"),
            "n_probes": len(t),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=int, default=80, help="probes per class")
    ap.add_argument("--dedup", nargs="+", default=["threshold", "llm"])
    args = ap.parse_args()
    t0 = time.time()

    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "e3" / "github.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    rng0 = np.random.default_rng(31337)
    rng0.shuffle(rows)
    rows = rows[: N_HIST + 4 * args.probes]
    texts = [r["text"] for r in rows]
    vecs = embed_cached([t[:900] for t in texts])
    cents = lab.kmeans_centroids(vecs, B, seed=0)

    n = len(rows)
    idx = np.arange(n)
    rng0.shuffle(idx)
    member = idx[: args.probes]
    nonmember = idx[args.probes : 2 * args.probes]
    filler = idx[2 * args.probes : 2 * args.probes + N_HIST]
    probes = {"member": member, "nonmember": nonmember, "filler": filler}
    print(
        f"github: {n} records; {len(member)} member probes, {len(nonmember)} non-member, "
        f"{len(filler)} filler in cohort H",
        flush=True,
    )

    llms = available()
    results = {
        "experiment": "E3",
        "attack": "deduplication membership inference",
        "n_probes_per_class": args.probes,
        "dedup_threshold": DEDUP_THRESHOLD,
        "models": [m.name for m in llms],
        "rows": [],
    }

    for dedup in args.dedup:
        if dedup == "llm" and not llms:
            print("no LLM backend available; skipping the llm arm", flush=True)
            continue
        llm = llms[0] if dedup == "llm" else None
        for kind in ("p", "m"):
            r = run(kind, dedup, llm, texts, vecs, cents, probes, np.random.default_rng(7))
            r["model"] = llm.name if llm else None
            results["rows"].append(r)
            print(
                f"[{dedup:>9} / {kind}] AUC {r['auc']:.3f} adv {r['advantage']:.3f} | "
                f"write rate member {r['write_rate_member']:.2f} non-member "
                f"{r['write_rate_nonmember']:.2f} | observed flushes "
                f"{r['observed_flushes_member']:.2f} vs {r['observed_flushes_nonmember']:.2f}",
                flush=True,
            )
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"wrote {OUT.relative_to(C.repo_root())} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
