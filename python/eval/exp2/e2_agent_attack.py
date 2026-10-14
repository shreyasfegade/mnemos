"""E2: the structure-transfer attack run end to end on records a real agent wrote.

Every previous transfer measurement used corpus passages as if they were
memories. The agent study measured whether an agent writes at all, but the two were never joined:
the attack was never run against records the model actually authored. It is here.

Two phases, because the expensive part is not the attack
========================================================
``--generate`` drives a mem0-style pipeline over adversary-supplied ``kubernetes`` issues with a
local model, and records every memory the model wrote, in its own words, with the issue's topic and
turn id. That corpus is cached in ``data/e3/agent_memories.jsonl``.

The attack phase then runs against those agent-written records, in a live Mnemos-P store, with the
realistic frequency estimator, and is repeated over trials and budgets so every number carries an
interval.

Bursts (C12)
============
A writing turn yields 2.5-3.7 memories on the same subject within milliseconds of each other, so a
realistic flush carries a small burst of **same-topic** records rather than one. That helps the
attacker, because the regions of a burst co-occur. ``w_real`` sweeps 1, 2 and 4 to measure how much.

    cd python && python -m eval.exp2.e2_agent_attack --generate --turns 600
    cd python && python -m eval.exp2.e2_agent_attack
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.attacks import estimators as X
from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e2_agent_attack.json"
MEMS = C.repo_root() / "data" / "e3" / "agent_memories.jsonl"

B = 64
CAP = 48
DIM = 768
BLOCK = 1024
W = 16
BUDGETS = (4, 8, 16, 32)
W_REALS = (1, 2, 4)
TRIALS = 20


# --------------------------------------------------------------------------- generation


def generate(turns: int, model_filter: str | None) -> None:
    """Run the agent and cache every memory it writes."""
    from eval.agent.llm import available
    from eval.agent.memory_agent import MemoryAgent, Turn

    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "e3" / "github.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    import random

    # A contiguous chronological window, not a shuffle: E1 splits this same corpus at a real date,
    # so the stream has to keep its real order and its timestamps.
    rows.sort(key=lambda r: r["ts"])
    start = max(0, (len(rows) - turns) // 2)
    sample = rows[start : start + turns]
    random.Random(7)  # retained so the seed is recorded; the stream itself is not shuffled

    pool = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "e3" / "github.jsonl").open(encoding="utf-8")
        if line.strip()
    ][:4000]
    cents = lab.kmeans_centroids(embed_cached([r["text"][:1500] for r in pool]), B, seed=0)

    class Capture:
        def __init__(self):
            self.vecs: dict[str, np.ndarray] = {}
            self.n = 0

        def put(self, text, embedding, meta="{}"):
            mid = f"m{self.n}"
            self.n += 1
            self.vecs[mid] = np.asarray(embedding, dtype=np.float32)
            return mid

        def update(self, mid, text, embedding, meta="{}"):
            self.vecs.pop(mid, None)
            return self.put(text, embedding, meta)

        def search(self, embedding):
            return []

    models = [m for m in available() if model_filter is None or model_filter in m.name]
    if not models:
        raise SystemExit("no LLM backend available (start ollama, or set DEEPSEEK_API_KEY)")

    MEMS.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with MEMS.open("w", encoding="utf-8") as fh:
        for llm in models:
            print(f"--- generating with {llm.name} over {len(sample)} turns ---", flush=True)
            store = Capture()
            agent = MemoryAgent(
                llm,
                store,
                lambda ts: embed_cached(list(ts)),
                lambda v: int(np.argmax(cents @ np.asarray(v, dtype=np.float32))),
            )
            t0 = time.time()
            for i, r in enumerate(sample):
                turn = Turn(turn_id=str(r["id"]), topic=r["topic"], text=r["text"][:1500])
                before = set(agent.records)
                try:
                    agent.process(turn)
                except Exception as e:
                    print(f"  turn {turn.turn_id}: {e}", flush=True)
                for mid in set(agent.records) - before:
                    rec = agent.records[mid]
                    fh.write(
                        json.dumps(
                            {
                                "model": llm.name,
                                "turn_id": rec.turn_id,
                                "turn_index": i,
                                "ts": r["ts"],
                                "topic": rec.source_topic,
                                "text": rec.text,
                            }
                        )
                        + "\n"
                    )
                    written += 1
                if (i + 1) % 50 == 0:
                    print(
                        f"  {i + 1}/{len(sample)} turns, {written} memories, "
                        f"{time.time() - t0:.0f}s",
                        flush=True,
                    )
    print(
        f"wrote {written} agent-authored memories to {MEMS.relative_to(C.repo_root())}", flush=True
    )


# --------------------------------------------------------------------------- attack


def load_memories(model: str | None) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    rows = [json.loads(line) for line in MEMS.open(encoding="utf-8") if line.strip()]
    if model:
        rows = [r for r in rows if r["model"] == model]
    names = sorted({r["topic"] for r in rows})
    idx = {t: i for i, t in enumerate(names)}
    texts = [r["text"] for r in rows]
    topics = np.array([idx[r["topic"]] for r in rows])
    turn_index = np.array([r["turn_index"] for r in rows])
    return texts, topics, names, turn_index


def frequency_labels(
    by_topic: dict[int, list[list[int]]], n_regions: int, n_topics: int
) -> dict[int, int]:
    """Label each region with the topic whose induced flushes hit it most above the mean."""
    topics = [t for t, f in by_topic.items() if f]
    if not topics:
        return {}
    rate = np.zeros((len(topics), n_regions))
    for i, t in enumerate(topics):
        for f in by_topic[t]:
            for r in set(f):
                if 0 <= r < n_regions:
                    rate[i, r] += 1
        rate[i] /= max(len(by_topic[t]), 1)
    mean = rate.mean(0)
    n_min = min(len(by_topic[t]) for t in topics)
    q = float(mean.mean())
    margin = 2.0 * float(np.sqrt(max(q * (1 - q), 1e-9) / max(n_min, 1)))
    out: dict[int, int] = {}
    for r in range(n_regions):
        best = int(np.argmax(rate[:, r]))
        if rate[best, r] - mean[r] > margin:
            out[r] = topics[best]
    return out


def read_flushes(trace: Path, offset: int) -> tuple[list[list[int]], int]:
    if not trace.exists():
        return [], offset
    with trace.open("r", encoding="utf-8") as fh:
        fh.seek(offset)
        lines = [line for line in fh if line.strip()]
    new_offset = trace.stat().st_size
    flushes, pending = [], []
    for line in lines:
        e = json.loads(line)
        if e["kind"] == "region_write":
            pending.append(int(e["region"]))
        elif e["kind"] == "epoch_flush":
            flushes.append(pending)
            pending = []
    if pending:
        flushes.append(pending)
    return flushes, new_offset


def run_attack(vecs, topics, n_topics, cents, budget: int, w_real: int, trial: int) -> dict:
    rng = np.random.default_rng(900 + trial)
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.base_overrides(
            dim=DIM, buckets=B, capacity=CAP, block=BLOCK, epoch_writes=W, probe_spread=1
        )
        eng, _, _ = lab.create("p", tmp / "s", cents, over, {})

        # Each trial is an *independent induction sequence*: it draws its own cohort-H / cohort-P
        # split, not just its own induction order. The earlier harness fixed the
        # split at the median and varied only the order, so the bootstrap over trials captured
        # within-split noise rather than seed-to-seed variance. Re-drawing the split per trial makes
        # the resampling unit a whole independent sequence, which the interval should reflect.
        n = len(vecs)
        perm = rng.permutation(n)
        half = n // 2
        H = perm[:half]
        P = perm[half:]

        for i in H:
            with contextlib.suppress(Exception):
                eng.put(f"h{i}", vecs[i].tolist(), "{}")
        lab.drain(eng)
        eng.flush()
        gt_region = np.array([eng.ground_truth_region(vecs[i].tolist()) for i in H])

        trace = tmp / "t.jsonl"
        trace.write_text("", encoding="utf-8")
        eng.set_trace_sink(str(trace))
        offset = 0

        by_topic_pool: dict[int, list[int]] = defaultdict(list)
        for i in P:
            by_topic_pool[int(topics[i])].append(int(i))

        by_topic: dict[int, list[list[int]]] = {t: [] for t in range(n_topics)}
        for t in range(n_topics):
            pool = by_topic_pool.get(t, [])
            if len(pool) < w_real:
                continue
            rng.shuffle(pool)
            # Each induced "turn" contributes a burst of w_real same-topic memories, flushed
            # together -- the regime Table 6 of the paper actually shows.
            for k in range(min(budget, len(pool) // w_real)):
                burst = pool[k * w_real : (k + 1) * w_real]
                for i in burst:
                    with contextlib.suppress(Exception):
                        eng.put(f"p{i}", vecs[i].tolist(), "{}")
                eng.flush()
                fl, offset = read_flushes(trace, offset)
                for f in fl:
                    by_topic[t].append(f)

        labels = frequency_labels(by_topic, B, n_topics)
        with contextlib.suppress(Exception):
            eng.close()

        hits = sum(
            1 for r, t in zip(gt_region, topics[H], strict=True) if labels.get(int(r)) == int(t)
        )
        unlabelled = sum(1 for r in gt_region if int(r) not in labels)
        # The adversary guesses where it has no label (estimators.score explains why).
        acc = X.guess_accuracy(hits, unlabelled, len(H), n_topics)
        chance = 1.0 / n_topics
        bal = X.balanced_guess_accuracy(
            [labels.get(int(r), -1) for r in gt_region], topics[H], n_topics
        )
        return {
            "acc_H": acc,
            "adv_H": acc - chance,
            "adv_H_bal": bal - chance,
            "regions_labelled": len(labels),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_growing(vecs, topics, n_topics, cents, budget: int, w_real: int, trial: int) -> dict:
    """The positive control: the same agent-written memories under a growing layout.

    Without it, the Mnemos-P row cannot separate "agent-written memories do not carry the topic
    structure transfer relies on" from "this harness cannot find structure at all". A growing,
    unpadded layout discloses each record's partition in the snapshot and a flush names exactly
    the partitions its records went to, which is the setting the attack was built for; if transfer
    fails here too, the P row says nothing about P.
    """
    rng = np.random.default_rng(900 + trial)
    part = np.argmax(vecs @ cents.T, axis=1)
    n = len(vecs)
    # Independent split per trial, as in run_attack.
    perm = rng.permutation(n)
    half = n // 2
    H = perm[:half]
    P = perm[half:]
    by_topic_pool: dict[int, list[int]] = defaultdict(list)
    for i in P:
        by_topic_pool[int(topics[i])].append(int(i))
    by_topic: dict[int, list[list[int]]] = {t: [] for t in range(n_topics)}
    for t in range(n_topics):
        pool = by_topic_pool.get(t, [])
        if len(pool) < w_real:
            continue
        rng.shuffle(pool)
        for k in range(min(budget, len(pool) // w_real)):
            burst = pool[k * w_real : (k + 1) * w_real]
            by_topic[t].append(sorted({int(part[i]) for i in burst}))
    labels = frequency_labels(by_topic, B, n_topics)
    hits = sum(1 for i in H if labels.get(int(part[i])) == int(topics[i]))
    unlabelled = sum(1 for i in H if int(part[i]) not in labels)
    acc = X.guess_accuracy(hits, unlabelled, len(H), n_topics)
    bal = X.balanced_guess_accuracy([labels.get(int(part[i]), -1) for i in H], topics[H], n_topics)
    return {
        "acc_H": acc,
        "adv_H": acc - 1.0 / n_topics,
        "adv_H_bal": bal - 1.0 / n_topics,
        "regions_labelled": len(labels),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--turns", type=int, default=600)
    ap.add_argument("--model", default=None)
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args()

    if args.generate:
        generate(args.turns, args.model)
        return
    if not MEMS.exists():
        raise SystemExit(f"{MEMS} missing; run with --generate first")

    t0 = time.time()
    texts, topics, names, _ = load_memories(args.model)
    n_topics = len(names)
    vecs = embed_cached(texts)
    cents = lab.kmeans_centroids(vecs, B, seed=0)
    chance = 1.0 / n_topics
    print(
        f"{len(texts)} agent-authored memories, {n_topics} topics, chance {chance:.3f}; "
        f"B={B} C={CAP}",
        flush=True,
    )

    results = {
        "experiment": "E2",
        "source": "agent-authored memories",
        "n_memories": len(texts),
        "n_topics": n_topics,
        "chance": chance,
        "buckets": B,
        "rows": [],
    }
    for layout, fn in (("growing", run_growing), ("p", run_attack)):
        for w_real in W_REALS:
            for budget in BUDGETS:
                runs = [
                    fn(vecs, topics, n_topics, cents, budget, w_real, tr)
                    for tr in range(args.trials)
                ]
                m, lo, hi = lab.boot_ci([r["adv_H"] for r in runs])
                bm, blo, bhi = lab.boot_ci([r["adv_H_bal"] for r in runs])
                row = {
                    "layout": layout,
                    "w_real": w_real,
                    "budget_turns_per_topic": budget,
                    "adv_H": m,
                    "lo": lo,
                    "hi": hi,
                    # balanced accuracy (mean per-topic recall), which the prior cannot raise
                    "adv_H_bal": bm,
                    "lo_bal": blo,
                    "hi_bal": bhi,
                    "trials": args.trials,
                    "resample_unit": "independent induction sequence (own cohort split + order)",
                }
                results["rows"].append(row)
                print(
                    f"{layout:>7} w_real={w_real} budget={budget:>3}: "
                    f"adv_H {m:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                    flush=True,
                )
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"wrote {OUT.relative_to(C.repo_root())} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
