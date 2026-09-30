"""E17: the composed attack on Mnemos-P, end to end.

Figure 3 and Table 3 report Mnemos-P's cohort-H exposure with cohort-H membership *supplied by the
harness*. The paper motivates a chain the oracle short-circuits:

    retained file versions  ->  which region holds an earlier record  (membership)
    induced writes          ->  region -> topic labels
    compose                 ->  the earlier record's topic

At g = 1 the version diff names a record's region only to within the w + 1 regions a flush rewrote
(supplement Table 10: precision 0.208), so the composed attack must be weaker than the oracle. This
experiment runs the whole chain: it derives membership from actual diffs at g in {1, 2, 4, 16},
labels regions from induced writes exactly as E1 does, and reports the composed accuracy beside the
oracle accuracy that knows each record's region exactly.

    cd python && python -m eval.exp2.e17_composed --trials 20
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.exp2 import lab
from eval.exp2.e1_longitudinal import frequency_labels, load_stream, read_flushes
from eval.exp2.e8_multisnapshot import changed, region_digest
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e17_composed.json"

B = 64
CAP = 16
DIM = 768
BLOCK = 256
W = 4
N_HIST = 300
BUDGET = 64
GRANULARITIES = (1, 2, 4, 16)
TRIALS = 20


def guess_acc(hits: int, unlabelled: int, n: int, k: int) -> float:
    """Accuracy crediting an unlabelled record with the uniform 1/k guess an adversary makes."""
    return (hits + unlabelled / k) / max(n, 1)


def one_trial(vecs, topics, n_topics, cents, trial: int) -> dict:
    rng = np.random.default_rng(1700 + trial)
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.merge(
            lab.base_overrides(
                dim=DIM, buckets=B, capacity=CAP, block=BLOCK, epoch_writes=W, probe_spread=1
            ),
            {"v2": {"rewrite_at_lock": False}},
        )
        eng, _, _ = lab.create("p", tmp / "s", cents, over, {})
        st = eng.stat()
        slot_len = int(st["slot_len"])
        part_file = tmp / "s" / "partitions.bin"

        # --- cohort H, one record at a time, capturing a file version after each flush.
        idx = rng.choice(len(vecs), size=N_HIST, replace=False)
        versions = [region_digest(part_file, B, CAP, slot_len)]
        h_id_topic: list[tuple[str, int]] = []
        h_true_region: list[int] = []
        for i in idx:
            try:
                r = int(eng.ground_truth_region(vecs[int(i)].tolist()))
                mid = eng.put(f"h{i}", vecs[int(i)].tolist(), "{}")
            except Exception:
                continue
            eng.flush()
            h_id_topic.append((mid, int(topics[int(i)])))
            h_true_region.append(r)
            versions.append(region_digest(part_file, B, CAP, slot_len))

        n_h = len(h_id_topic)

        # Per-record diff at each granularity: the regions that changed over the g flushes around
        # the record's write. Coarser g merges more flushes, so the candidate set grows.
        def candidates_at(g: int) -> list[set[int]]:
            out = []
            for j in range(n_h):
                lo = (j // g) * g
                hi = min(lo + g, n_h)
                cand = changed(versions[lo], versions[hi])
                out.append(cand if cand else set(range(B)))
            return out

        # --- induced writes label regions with topics, exactly as E1 does.
        pool_by_topic: dict[int, list[int]] = defaultdict(list)
        post = np.setdiff1d(np.arange(len(vecs)), idx)
        for i in post:
            pool_by_topic[int(topics[int(i)])].append(int(i))
        trace = tmp / "t.jsonl"
        trace.write_text("", encoding="utf-8")
        eng.set_trace_sink(str(trace))
        offset = 0
        by_topic: dict[int, list[list[int]]] = {t: [] for t in range(n_topics)}
        for t in range(n_topics):
            pool = pool_by_topic.get(t, [])
            picks = (
                rng.choice(pool, size=min(BUDGET, len(pool)), replace=False) if len(pool) else []
            )
            for i in picks:
                with contextlib.suppress(Exception):
                    eng.put(f"p{int(i)}", vecs[int(i)].tolist(), "{}")
                eng.flush()
                fl, offset = read_flushes(trace, offset)
                by_topic[t].extend(fl)
        labels = frequency_labels(by_topic, B)

        with contextlib.suppress(Exception):
            eng.close()

        # --- oracle: the region is known exactly.
        o_hits = sum(
            1 for (_, tp), r in zip(h_id_topic, h_true_region, strict=True) if labels.get(r) == tp
        )
        o_unl = sum(1 for r in h_true_region if r not in labels)
        oracle = guess_acc(o_hits, o_unl, n_h, n_topics)

        # --- composed: the region is only known to be one of the diff candidates. The adversary
        # predicts the label of the labelled candidate the map is most confident about; if none of
        # the candidates is labelled, it abstains (and is credited the uniform guess).
        composed = {}
        for g in GRANULARITIES:
            cands = candidates_at(g)
            hits = unl = 0
            for (_, tp), cand in zip(h_id_topic, cands, strict=True):
                labelled = [labels[r] for r in cand if r in labels]
                if not labelled:
                    unl += 1
                    continue
                # majority vote among labelled candidates, ties by first
                pred = max(set(labelled), key=labelled.count)
                hits += int(pred == tp)
            composed[g] = {
                "acc_H": guess_acc(hits, unl, n_h, n_topics),
                "coverage_H": 1 - unl / max(n_h, 1),
                "mean_candidates": float(np.mean([len(c) for c in cands])),
            }
        return {"oracle_acc": oracle, "composed": composed, "n_h": n_h}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args()

    texts, topics, names, _months, source = load_stream()
    vecs = embed_cached(texts)
    n_topics = len(names)
    cents = lab.kmeans_centroids(vecs, B, seed=0)
    chance = 1.0 / n_topics
    print(f"B={B} C={CAP} w={W} topics={n_topics} chance={chance:.3f} source={source}", flush=True)

    runs = [one_trial(vecs, topics, n_topics, cents, t) for t in range(args.trials)]
    om, olo, ohi = lab.boot_ci([r["oracle_acc"] for r in runs])
    results = {
        "experiment": "E17",
        "buckets": B,
        "capacity": CAP,
        "epoch_writes": W,
        "n_topics": n_topics,
        "chance": chance,
        "budget_per_topic": BUDGET,
        "source": source,
        "oracle": {"acc_H": om, "lo": olo, "hi": ohi, "adv_H": om - chance},
        "composed": [],
    }
    print(
        f"oracle (region known): acc {om:.3f} [{olo:.3f},{ohi:.3f}] adv {om - chance:+.3f}",
        flush=True,
    )
    for g in GRANULARITIES:
        accs = [r["composed"][g]["acc_H"] for r in runs]
        m, lo, hi = lab.boot_ci(accs)
        cand = float(np.mean([r["composed"][g]["mean_candidates"] for r in runs]))
        cov = float(np.mean([r["composed"][g]["coverage_H"] for r in runs]))
        row = {
            "granularity_flushes": g,
            "acc_H": m,
            "lo": lo,
            "hi": hi,
            "adv_H": m - chance,
            "adv_H_lo": lo - chance,
            "mean_candidates": cand,
            "coverage_H": cov,
        }
        results["composed"].append(row)
        print(
            f"  composed g={g:>2}: acc {m:.3f} [{lo:.3f},{hi:.3f}] adv {m - chance:+.3f} "
            f"({cand:.1f} candidate regions, coverage {cov:.2f})",
            flush=True,
        )
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
