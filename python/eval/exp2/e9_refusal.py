"""E9: the refusal channel, and the capacity-confidentiality trade-off behind it.

The channel
=======================
Mnemos-P refuses a ``put`` whose partition is full. A refusal that writes nothing is observable:
an adversary that has labelled regions sees the region it expected to receive the write did not,
and so learns that the region's live load has reached ``C``. Because cohort P's contribution to
that load is something the adversary already watches -- it induced most of it -- the remainder is
**cohort H's load on that region**, and via the region's topic label that is the asset G5 protects.
Admissibility excludes this from Theorem 1, but the capacity arm below shows the excluded case is
the ordinary case under realistic skew, so excluding it is not a defence.

Design
======
The question is about a *region's* fullness, so the experiment targets regions directly rather
than topics: half the target regions are filled to capacity during cohort H and half are left with
room, then the adversary induces writes destined for each and watches whether the region is
rewritten. Padding is the noise floor, so ``w`` is kept small against ``B`` (4 against 64, a base
rate near 0.06) rather than letting padding swamp the signal.

``silent``  a refusal writes nothing (the earlier behaviour): the channel is measured.
``padded``  a refusal still rewrites the region the record wanted: the channel should close.

The adversary is granted each record's region, which is the strongest footing for the localisation
step and keeps the measurement about the *refusal* rather than about how hard labelling is.

    cd python && python -m eval.exp2.e9_refusal
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e9_refusal.json"

B = 64
CAP = 32
DIM = 768
BLOCK = 256
W = 4  # small against B so padding is a low noise floor
TARGET_REGIONS = 24  # half filled, half left open
INDUCED_PER_REGION = 20
# An "open" region must still have room after every induced write, or it would fill part-way
# through and contaminate its own measurement: OPEN_FILL + INDUCED_PER_REGION must stay under CAP.
OPEN_FILL = 4
TRIALS = 20
ZIPF_A = 1.3


def flush_regions(trace: Path, offset: int) -> tuple[list[list[int]], int]:
    """Regions written since ``offset``, segmented by flush. Returns the new offset too."""
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


def run_channel(mode: str, vecs, trial: int) -> dict:
    rng = np.random.default_rng(4000 + trial)
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.merge(
            lab.base_overrides(
                dim=DIM, buckets=B, capacity=CAP, block=BLOCK, epoch_writes=W, probe_spread=1
            ),
            {"v2": {"refusal_mode": mode}},
        )
        eng, _, _ = lab.create("p", tmp / "s", cents_global, over, {})

        # The region map must come from *this* store. Every `create` draws a fresh master key and
        # therefore a fresh keyed permutation, so a map derived from any other store is unrelated
        # to this one. Asking the engine also removes the quantisation mismatch that a float
        # nearest-centroid computed in numpy would introduce at partition boundaries.
        region_of_record = np.array([eng.ground_truth_region(v.tolist()) for v in vecs])

        by_region: dict[int, np.ndarray] = {}
        for r in range(B):
            idx = np.where(region_of_record == r)[0]
            if len(idx) >= CAP + INDUCED_PER_REGION + 4:
                by_region[r] = idx
        usable = sorted(by_region)
        rng.shuffle(usable)
        targets = usable[:TARGET_REGIONS]
        full_set = set(targets[: len(targets) // 2])

        # Cohort H: fill the "full" regions to capacity, leave the others with room.
        admitted = Counter()
        for r in targets:
            want = CAP if r in full_set else OPEN_FILL
            pool = by_region[r][:want]
            for i in pool:
                try:
                    eng.put(f"h{i}", vecs[int(i)].tolist(), "{}")
                    admitted[r] += 1
                except Exception:
                    pass
        lab.drain(eng)
        eng.flush()

        # Post-tau: induce writes destined for each target region, one flush each.
        trace = tmp / "t.jsonl"
        trace.write_text("", encoding="utf-8")
        eng.set_trace_sink(str(trace))
        offset = 0
        rates, labels = [], []
        for r in targets:
            pool = by_region[r][CAP : CAP + INDUCED_PER_REGION]
            hits = 0
            for i in pool:
                with contextlib.suppress(Exception):
                    eng.put(f"p{i}", vecs[int(i)].tolist(), "{}")
                eng.flush()
                fl, offset = flush_regions(trace, offset)
                if any(r in f for f in fl):
                    hits += 1
            rates.append(hits / max(len(pool), 1))
            labels.append(1 if admitted[r] >= CAP else 0)

        st = eng.stat()
        out = {
            "mode": mode,
            "trial": trial,
            "rates": rates,
            "full": labels,
            "refused": int(st["refused"]),
            "ghost_writes": int(st.get("ghost_writes", 0)),
        }
        with contextlib.suppress(Exception):
            eng.close()
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def skewed_indices(topics: np.ndarray, n_topics: int, n: int, rng) -> np.ndarray:
    ranks = np.arange(1, n_topics + 1)
    wts = 1.0 / ranks**ZIPF_A
    wts /= wts.sum()
    order = rng.permutation(n_topics)
    tw = np.zeros(n_topics)
    tw[order] = wts
    by_topic = {t: np.where(topics == t)[0] for t in range(n_topics)}
    picks = []
    for _ in range(n):
        t = int(rng.choice(n_topics, p=tw))
        if len(by_topic[t]):
            picks.append(int(rng.choice(by_topic[t])))
    return np.array(picks)


def capacity_arm(kind: str, vecs, topics, n_topics) -> dict:
    """How many real memories the design admits before it first refuses, under realistic skew."""
    rng = np.random.default_rng(11)
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.base_overrides(
            dim=DIM, buckets=B, capacity=CAP, block=BLOCK, epoch_writes=W, probe_spread=1
        )
        eng, _, _ = lab.create(kind, tmp / "s", cents_global, over, {})
        trace = skewed_indices(topics, n_topics, B * CAP * 2, rng)
        admitted = 0
        first_refusal = None
        refused_by_topic = Counter()
        for n, i in enumerate(trace):
            try:
                eng.put(f"m{n}", vecs[i].tolist(), "{}")
                admitted += 1
            except Exception:
                if first_refusal is None:
                    first_refusal = admitted
                refused_by_topic[int(topics[i])] += 1
            if n % (W * 4) == 0:
                lab.drain(eng)
        lab.drain(eng)
        total = B * CAP
        out = {
            "engine": kind,
            "slots": total,
            "admitted_total": admitted,
            "first_refusal_after": first_refusal if first_refusal is not None else admitted,
            "utilisation_at_first_refusal": (
                first_refusal if first_refusal is not None else admitted
            )
            / total,
            "refusals_by_topic": dict(refused_by_topic.most_common()),
        }
        with contextlib.suppress(Exception):
            eng.close()
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


cents_global: np.ndarray | None = None


def main() -> None:
    global cents_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args()
    t0 = time.time()

    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    names = sorted({r["topic"] for r in rows})
    idx = {t: i for i, t in enumerate(names)}
    vecs = embed_cached([r["text"] for r in rows])
    topics = np.array([idx[r["topic"]] for r in rows])
    cents_global = lab.kmeans_centroids(vecs, B, seed=0)
    n_topics = len(names)
    print(f"D2: {len(rows)} records, {n_topics} topics; B={B} C={CAP} w={W}", flush=True)

    results = {
        "experiment": "E9",
        "buckets": B,
        "capacity": CAP,
        "epoch_writes": W,
        "induced_per_region": INDUCED_PER_REGION,
        "padding_base_rate": 1 - (1 - 1 / B) ** (W + 1),
        "channel": [],
        "capacity_arm": [],
    }

    for mode in ("silent", "padded"):
        rates, labels, refusals, ghosts = [], [], 0, 0
        per_trial_auc, per_trial_gap = [], []
        for trial in range(args.trials):
            r = run_channel(mode, vecs, trial)
            rates.extend(r["rates"])
            labels.extend(r["full"])
            refusals += r["refused"]
            ghosts += r["ghost_writes"]
            # One AUC per trial, so the interval is over independent runs rather than over
            # regions within a run, which are not independent of each other.
            rt, lt = np.array(r["rates"]), np.array(r["full"])
            if len(np.unique(lt)) > 1:
                per_trial_auc.append(float(roc_auc_score(lt, -rt)))
                per_trial_gap.append(float(rt[lt == 0].mean() - rt[lt == 1].mean()))
        ra, la = np.array(rates), np.array(labels)
        pooled = float(roc_auc_score(la, -ra)) if len(np.unique(la)) > 1 else float("nan")
        auc, lo, hi = lab.boot_ci(per_trial_auc)
        gap, glo, ghi = lab.boot_ci(per_trial_gap)
        mf = float(ra[la == 1].mean()) if (la == 1).any() else float("nan")
        mo = float(ra[la == 0].mean()) if (la == 0).any() else float("nan")
        row = {
            "mode": mode,
            "auc_full_vs_open": auc,
            "lo": lo,
            "hi": hi,
            "auc_pooled": pooled,
            "advantage": abs(auc - 0.5) if auc == auc else float("nan"),
            "mean_write_rate_full_regions": mf,
            "mean_write_rate_open_regions": mo,
            "gap": gap,
            "gap_lo": glo,
            "gap_hi": ghi,
            "refusals": refusals,
            "ghost_writes": ghosts,
            "trials": len(per_trial_auc),
            "n_regions_observed": len(labels),
            "n_full": int(la.sum()),
        }
        results["channel"].append(row)
        print(
            f"[{mode:>6}] AUC(full vs open) {auc:.3f} [{lo:.3f},{hi:.3f}]  "
            f"write-rate full {mf:.3f} open {mo:.3f}  gap {gap:+.3f} [{glo:+.3f},{ghi:+.3f}]  "
            f"refusals {refusals} ghosts {ghosts}",
            flush=True,
        )

    for kind in ("p", "m"):
        r = capacity_arm(kind, vecs, topics, n_topics)
        results["capacity_arm"].append(r)
        print(
            f"[{kind}] admitted {r['admitted_total']} of {r['slots']} slots; first refusal after "
            f"{r['first_refusal_after']} ({r['utilisation_at_first_refusal']:.1%} utilisation)",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
