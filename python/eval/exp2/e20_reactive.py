"""E20: the reactive (feedback-agent) game.

Definition 1's game has the adversary choose every post-tau operation. A self-writing agent is a
second source of operations: it retrieves a cohort-H record and, in the same turn, writes or
declines to write as a function of what it retrieved. Proposition 4 (derived-write separation) says
any store whose operation leakage depends on the content -- or the mere occurrence -- of an
operation is broken by such an agent, and Theorem 2 (the composition theorem this round adds) says
a store whose disk behaviour is a function of the clock alone is not.

This experiment is the empirical form of both. A feedback agent deduplicates: it writes a supplied
fact only if a similar one is not already stored. The adversary supplies facts, watches the storage
trace, and guesses whether each fact was a member of cohort H. The distinguishing power is reported
as the AUC of the best trace statistic, per arm:

* ``M sweep``            -- the shipped store: clocked sweep flush. Prediction: at chance.
* ``M regions``          -- clocked whole-region flush (the earlier design). Prediction: at chance.
* ``M flush on write``   -- the clock removed (``on_demand_flush``). Prediction: broken.
* ``M content placement``-- clocked, but placement by content. Prediction: broken (the region a
                            real write lands in is content-derived even though occurrence is hidden).
* ``P shipped``          -- Mnemos-P: on-demand flush, content placement, permutation. Prediction:
                            broken, which is what makes P dual-epoch secure (Thm 1) yet not
                            reactively secure -- the separation RDE => DE, DE =/=> RDE.

Dedup here is a fixed cosine threshold, so the experiment is deterministic and needs no model;
E18 runs the same attack against real mem0/LangMem/CrewAI end to end. The attacker statistics are
held out: a threshold on the trace is fixed on a training half of the probes and scored on the
other, so the AUC is not an in-sample artifact.

    cd python && python -m eval.exp2.e20_reactive --probes 400
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

from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e20_reactive.json"

B = 32
CAP = 32
DIM = 768
BLOCK = 1024
N_HIST = 200
DEDUP_THRESHOLD = 0.9

# Each arm: the engine, the ablation, and the parameter overrides that define the flush schedule.
ARMS = [
    {"name": "M sweep", "kind": "m", "abl": {}, "over": {"m": {"flush_mode": "sweep", "sweep_slots": 8, "journal_slots": 4}}},
    {"name": "M regions", "kind": "m", "abl": {}, "over": {"m": {"flush_mode": "regions"}}},
    {"name": "M flush on write", "kind": "m", "abl": {"on_demand_flush": True}, "over": {"m": {"flush_mode": "regions"}}},
    {"name": "M content placement", "kind": "m", "abl": {"content_placement": True}, "over": {"m": {"flush_mode": "regions"}}},
    {"name": "P shipped", "kind": "p", "abl": {}, "over": {}},
]


def read_trace(trace: Path, offset: int) -> tuple[dict, int]:
    """Counts of each event kind, and the set of regions written, since ``offset``."""
    counts = {"epoch_flush": 0, "region_write": 0}
    regions: list[int] = []
    if not trace.exists():
        return {"counts": counts, "regions": regions}, offset
    with trace.open("r", encoding="utf-8") as fh:
        fh.seek(offset)
        lines = [ln for ln in fh if ln.strip()]
        new_off = fh.tell()
    for ln in lines:
        e = json.loads(ln)
        k = e["kind"]
        counts[k] = counts.get(k, 0) + 1
        if k == "region_write":
            regions.append(int(e["region"]))
    return {"counts": counts, "regions": regions}, new_off


def _auc_ci(y, col, sign, te, rng) -> tuple[float, float, float]:
    if len(np.unique(y[te])) < 2:
        return 0.5, 0.5, 0.5
    a_te = float(roc_auc_score(y[te], sign * col[te]))
    boots = []
    for _ in range(2000):
        s = rng.integers(0, len(te), len(te))
        yt, ct = y[te][s], sign * col[te][s]
        if len(np.unique(yt)) > 1 and len(np.unique(ct)) > 1:
            boots.append(roc_auc_score(yt, ct))
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (a_te, a_te)
    return a_te, lo, hi


def heldout_aucs(y: np.ndarray, feats: dict[str, np.ndarray], rng) -> dict[str, dict]:
    """Held-out AUC of each named statistic, oriented on the training half only.

    Returns a dict per statistic with auc/lo/hi. The caller picks the strongest as the "learned"
    attacker and ``flushes_per_turn`` as the interpretable hand-built one.
    """
    n = len(y)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2 :]
    out: dict[str, dict] = {}
    for name, col in feats.items():
        if len(np.unique(col[tr])) < 2 or len(np.unique(y[tr])) < 2:
            out[name] = {"auc": 0.5, "lo": 0.5, "hi": 0.5, "sign": 1.0}
            continue
        sign = 1.0 if roc_auc_score(y[tr], col[tr]) >= 0.5 else -1.0
        auc, lo, hi = _auc_ci(y, col, sign, te, rng)
        out[name] = {"auc": auc, "lo": lo, "hi": hi, "sign": sign}
    return out


def play(arm: dict, texts, vecs, cents, assign, probes, rng) -> dict:
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.merge(
            lab.base_overrides(dim=DIM, buckets=B, capacity=CAP, block=BLOCK, epoch_writes=1, probe_spread=1),
            arm["over"],
        )
        eng, _, _ = lab.create(arm["kind"], tmp / "s", cents, over, arm["abl"])

        # Cohort H: the facts the agent already holds at tau.
        stored = []
        hist = list(probes["member"]) + list(probes["filler"])
        rng.shuffle(hist)
        for i in hist:
            with contextlib.suppress(Exception):
                eng.put(texts[i][:900], vecs[i].tolist(), "{}")
                stored.append(vecs[i])
        lab.drain(eng)
        eng.flush()
        smat = np.array(stored, dtype=np.float32) if stored else np.zeros((0, DIM), np.float32)

        trace = tmp / "t.jsonl"
        trace.write_text("", encoding="utf-8")
        eng.set_trace_sink(str(trace))
        offset = 0

        order = [(i, 1) for i in probes["member"]] + [(i, 0) for i in probes["nonmember"]]
        rng.shuffle(order)
        y, f_flush, f_regionhit, f_writes = [], [], [], []
        for i, is_member in order:
            vec = vecs[i]
            write = float(np.max(smat @ vec)) < DEDUP_THRESHOLD if smat.size else True
            # The clock fires every turn for the clocked arms; the on-demand arm and P flush only
            # when a record is staged.
            if write:
                with contextlib.suppress(Exception):
                    eng.put(texts[i][:900], vec.tolist(), "{}")
                    smat = np.vstack([smat, vec[None, :]]) if smat.size else vec[None, :]
            clocked = arm["kind"] == "m" and not arm["abl"].get("on_demand_flush")
            if clocked:
                eng.flush()
            elif arm["kind"] == "p":
                eng.flush()  # drains staged; on-demand means nothing to drain if no write
            ev, offset = read_trace(trace, offset)
            y.append(is_member)
            f_flush.append(ev["counts"].get("epoch_flush", 0))
            f_writes.append(ev["counts"].get("region_write", 0))
            # The region the adversary predicts this probe's content maps to, via the public table.
            want = int(assign[i])
            f_regionhit.append(sum(1 for r in ev["regions"] if r == want))

        with contextlib.suppress(Exception):
            eng.close()

        y = np.array(y)
        feats = {
            "flushes_per_turn": np.array(f_flush, float),
            "region_writes_per_turn": np.array(f_writes, float),
            "writes_to_content_region": np.array(f_regionhit, float),
        }
        return heldout_aucs(y, feats, rng)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=int, default=400, help="member + nonmember probes total")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ][:4000]
    texts = [r["text"] for r in rows]
    vecs = embed_cached(texts)
    cents = lab.kmeans_centroids(vecs, B, seed=0)
    assign = np.argmax(vecs @ cents.T, axis=1)

    rng = np.random.default_rng(20260928)
    n_each = args.probes // 2
    pool = rng.permutation(len(texts))
    member = pool[:n_each]
    filler = pool[n_each : N_HIST]
    nonmember = pool[N_HIST : N_HIST + n_each]
    probes = {"member": member, "nonmember": nonmember, "filler": filler}

    results = {"experiment": "E20", "buckets": B, "capacity": CAP, "dedup_threshold": DEDUP_THRESHOLD,
               "probes": args.probes, "note": "held-out AUC of the best trace statistic per arm", "rows": []}

    for arm in ARMS:
        t0 = time.time()
        aucs = play(arm, texts, vecs, cents, assign, probes, np.random.default_rng(7))
        # The learned attacker is the strongest of the three named statistics; the hand-built one
        # is flushes-per-turn, the derived-write occurrence signal Proposition 4 is about.
        learned_stat = max(aucs, key=lambda k: abs(aucs[k]["auc"] - 0.5))
        learned = {**aucs[learned_stat], "statistic": learned_stat}
        hand = {**aucs["flushes_per_turn"], "statistic": "flushes_per_turn"}
        row = {
            "arm": arm["name"],
            "engine": arm["kind"],
            "learned": learned,
            "hand_built": hand,
            "per_statistic": aucs,
            "seconds": time.time() - t0,
        }
        results["rows"].append(row)
        print(f"{arm['name']:>22}: AUC {learned['auc']:.3f} [{learned['lo']:.3f},{learned['hi']:.3f}] "
              f"via {learned_stat}  ({row['seconds']:.0f}s)", flush=True)
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
