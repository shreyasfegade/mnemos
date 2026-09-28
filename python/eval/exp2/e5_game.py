"""E5: the dual-epoch game, redone to 

What was wrong with the previous run
====================================
The old harness built its "maximally concentrating" history by drawing records whose assigned
partition was one of a few, and wrapped every ``put`` in ``except Exception: pass``. Concentrating
64 records into 4 partitions of capacity 32 overflows, so puts were refused, and the refusals were
swallowed. The histories were therefore **inadmissible**, which is precisely the case Theorem 1
excludes, and the "no keyed permutation" row measured the refusal channel rather than the
permutation. That is why an ablation the theorem does not require came out distinguishable.

What this run does instead
==========================
1. **Admissibility is constructed, then verified.** A history only accepts a record if its
   partition still has room, and every instance asserts the engine's ``refused`` counter is zero.
2. **An inadmissible arm is run on purpose**, in both refusal modes, so the refusal channel is
   measured as its own row rather than contaminating the others .
3. **Held-out evaluation.** The classifier is fitted on a training split and scored on a disjoint
   test split, so the reported AUC is not an in-sample artifact, and an AUC below 0.5 is reported
   as the anti-learning it is rather than as a distinguisher.
4. **Power.** 1,000 instances per bit by default, and the *upper* confidence bound is reported as
   the empirical ceiling for this attacker, because a test bounds leakage from below, never above.
5. **A hand-built distinguisher per ablation**, whose statistic is named and whose advantage is
   interpretable, alongside the learned one.
6. **Mnemos-M is played too**, at the same geometry.

    cd python && python -m eval.exp2.e5_game --instances 1000
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e5_game.json"

B = 16
# Capacity carries a margin over what the concentrated history needs (2 partitions x 48 = 96 for
# 64 records). The margin matters: the harness picks partitions with a float nearest-centroid
# while the engine picks with a quantised one, and at a partition boundary the two can disagree.
# Without the margin those few disagreements overflow a partition and the "admissible" arms log
# refusals, which is exactly the contamination this experiment exists to remove.
CAP = 64
DIM = 768
BLOCK = 256
W = 8
H_LEN = 64
RHO_PUTS = 12
# Post-tau puts aimed at the partitions sigma_0 concentrates into. Under b = 0 those partitions
# may be full and the puts refused; under b = 1 they are not. This is what puts the refusal
# channel inside the game rather than leaving it to be measured only in E9.
RHO_TARGETED = 8
CONCENTRATED = 4

# Admissibility is a property of the **whole game**, not of the history alone: the definition says
# no partition exceeds capacity *at any point during the game*, and rho's puts land in the very
# partitions sigma_0 concentrates into -- the targeted ones by construction, and the uniform ones
# by chance. Bounding cohort H at C and letting rho overflow is what put refusals into an
# "admissible" arm at 1,000 instances per bit, invisible at 50.
#
# Cohort H is therefore bounded so that the whole game fits: every one of rho's puts could land in
# one region and still not reach C, with slack for the quantisation boundary. The bound must also
# leave room for H_LEN offers across CONCENTRATED partitions, which is why C is 64 rather than 48:
#   2 * 40 = 80 >= 64 offers, and worst case 40 + 8 + 12 = 60 < 64.
H_REGION_CAP = CAP - RHO_TARGETED - RHO_PUTS - 4

# Each arm names the engine, its ablation, its parameter overrides, and whether the histories it
# submits are admissible. Only the admissible arms test Theorem 1.
# ``concentrated`` is how many partitions history sigma_0 aims at. Admissibility is a property of
# the *pair* (concentrated, CAP, H_LEN): with CAP = 32 and H_LEN = 64, two partitions hold the
# history exactly and one cannot, so the inadmissible arms use one. Getting this wrong is how the
# previous run ended up with zero refusals in the arms that were supposed to have them.
ARMS: list[dict] = [
    {
        "name": "P shipped",
        "kind": "p",
        "abl": {},
        "over": {},
        "admissible": True,
        "concentrated": 2,
    },
    {
        "name": "P no S1 (growing regions)",
        "kind": "p",
        "abl": {"growing_regions": True},
        "over": {},
        "admissible": True,
        "concentrated": 2,
    },
    {
        "name": "P no S2 (unsealed slots)",
        "kind": "p",
        "abl": {"unsealed_slots": True},
        "over": {},
        "admissible": True,
        "concentrated": 2,
    },
    {
        "name": "P no S3 (per-slot writes)",
        "kind": "p",
        "abl": {"slot_writes_only": True},
        "over": {},
        "admissible": True,
        "concentrated": 2,
    },
    {
        "name": "P identity permutation",
        "kind": "p",
        "abl": {"identity_permutation": True},
        "over": {},
        "admissible": True,
        "concentrated": 2,
    },
    {
        "name": "P inadmissible, padded refusal",
        "kind": "p",
        "abl": {},
        "over": {"v2": {"refusal_mode": "padded"}},
        "admissible": False,
        "concentrated": 1,
    },
    {
        "name": "P inadmissible, silent refusal",
        "kind": "p",
        "abl": {},
        "over": {"v2": {"refusal_mode": "silent"}},
        "admissible": False,
        "concentrated": 1,
    },
    {
        "name": "M shipped",
        "kind": "m",
        "abl": {},
        "over": {},
        "admissible": True,
        "concentrated": 2,
    },
    {
        "name": "M content placement",
        "kind": "m",
        "abl": {"content_placement": True},
        "over": {},
        "admissible": True,
        "concentrated": 2,
    },
    # The 2x2 that explains the no-S3 arm. Whole-region rewrites (S3) and random intra-region
    # placement are two independent ways of keeping a write from revealing a region's occupancy:
    # the index of a per-slot write is uniform over the free slots, and under admissibility the
    # free set is a uniformly random subset whatever cohort H put there. Only when BOTH are
    # removed should the slot index become the occupancy.
    {
        "name": "P first-free slot (S3 kept)",
        "kind": "p",
        "abl": {"first_free_slot": True},
        "over": {},
        "admissible": True,
        "concentrated": 2,
    },
    {
        "name": "P no S3, first-free slot",
        "kind": "p",
        "abl": {"slot_writes_only": True, "first_free_slot": True},
        "over": {},
        "admissible": True,
        "concentrated": 2,
    },
]


def store_files(d: Path) -> dict[str, bytes]:
    out = {}
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.name != "k":
            out[str(f.relative_to(d)).replace("\\", "/")] = f.read_bytes()
    return out


def features(
    snap: dict[str, bytes], rho_regions: list[int], rho_slots: list[tuple[int, int]], n_regions: int
) -> dict[str, float]:
    """Named features, so a hand-built distinguisher can pick one out by name."""
    main = snap.get("partitions.bin") or snap.get("slab.bin") or b""
    f: dict[str, float] = {
        "main_len": float(len(main)),
        "manifest_len": float(len(snap.get("manifest.enc", b""))),
    }
    region_sizes = sorted(len(v) for k, v in snap.items() if k.startswith("regions/"))
    arr = np.array(region_sizes, dtype=float) if region_sizes else np.zeros(1)
    f["region_size_std"] = float(arr.std())
    f["region_size_max"] = float(arr.max())
    f["region_size_nonzero"] = float((arr > 0).sum())
    if main:
        blocks = [main[i : i + 512] for i in range(0, len(main), 512)]
        f["distinct_blocks"] = float(len({bytes(b) for b in blocks}))
        f["zero_blocks"] = float(sum(1 for b in blocks if not any(b)))
    else:
        f["distinct_blocks"] = 0.0
        f["zero_blocks"] = 0.0
    hist = Counter(rho_regions)
    counts = np.array([hist.get(r, 0) for r in range(n_regions)], dtype=float)
    f["rho_write_std"] = float(counts.std())
    f["rho_low_region_share"] = float(counts[:CONCENTRATED].sum() / max(counts.sum(), 1.0))
    for r in range(n_regions):
        f[f"rho_r{r}"] = float(counts[r])
    # Slot indices. Under the per-slot-write ablation a put writes only its own slot, and which
    # slot is free depends on how full the region already is, so the slot indices a post-tau put
    # lands on carry the occupancy that whole-region rewrites hide. The shipped engine reports no
    # slots at all, so these are constant for it.
    slots = [s for _, s in rho_slots]
    f["rho_distinct_slots"] = float(len(set(rho_slots)))
    f["rho_slot_mean"] = float(np.mean(slots)) if slots else 0.0
    f["rho_slot_max"] = float(np.max(slots)) if slots else 0.0
    f["rho_slot_std"] = float(np.std(slots)) if slots else 0.0
    return f


def build_history(
    kind: str, eng, vecs, assign, b: int, rng, admissible: bool, concentrated: int
) -> int:
    """Write cohort H. Returns the number of records actually written.

    Under ``admissible`` a record is only offered if its partition still has room, which is the
    condition Theorem 1 assumes and the old harness violated.
    """
    load = Counter()
    written = 0
    pool = np.where(np.isin(assign, np.arange(concentrated)))[0] if b == 0 else np.arange(len(vecs))

    # Exactly H_LEN offers, never a retry-until-success loop. An inadmissible history is *meant*
    # to overflow, and retrying until 64 records land turns "this history overflows" into tens of
    # thousands of refused calls, which is a runaway rather than a measurement.
    #
    # The load is tracked against the region **the engine would choose**, not against a float
    # nearest-centroid computed here. The two disagree for records near a partition boundary,
    # because the engine assigns over quantised vectors, and although the disagreements are rare
    # they accumulate: at 1,000 instances per bit they put refusals into the arm whose whole
    # purpose is to have none. Asking the engine costs one call per offer and makes the
    # admissibility claim exact rather than approximate.
    offers = 0
    while offers < H_LEN:
        r = int(rng.choice(pool))
        if admissible and kind == "p":
            region = int(eng.ground_truth_region(vecs[r].tolist()))
            if load[region] >= H_REGION_CAP:
                continue  # not an offer: an admissible game never drives any partition to capacity
        else:
            region = -1
        offers += 1
        try:
            eng.put(f"h{r}", vecs[r].tolist(), "{}")
        except Exception:
            continue
        load[region] += 1
        written += 1
    # Mnemos-M's flush is clocked and drains only w records, so the history must be drained
    # rather than flushed once, or cohort H would be truncated for that engine only.
    lab.drain(eng)
    eng.flush()
    return written


def play(arm: dict, vecs, cents, assign, b: int, seed: int) -> tuple[dict[str, float], int, int]:
    rng = np.random.default_rng(seed)
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.merge(
            lab.base_overrides(
                dim=DIM, buckets=B, capacity=CAP, block=BLOCK, epoch_writes=W, probe_spread=1
            ),
            arm["over"],
        )
        eng, _, _ = lab.create(arm["kind"], tmp / "s", cents, over, arm["abl"])

        build_history(
            arm["kind"], eng, vecs, assign, b, rng, arm["admissible"], arm.get("concentrated", 2)
        )
        snap = store_files(tmp / "s")  # the snapshot at tau

        trace = tmp / "t.jsonl"
        eng.set_trace_sink(str(trace))
        rho_regions: list[int] = []
        rho_slots: list[tuple[int, int]] = []
        ids = []
        # rho is common to both bits: the same draw, from the same generator seed.
        prho = np.random.default_rng(99_991)
        for _ in range(RHO_PUTS):
            r = int(prho.integers(0, len(vecs)))
            with contextlib.suppress(Exception):
                ids.append(eng.put(f"p{r}", vecs[r].tolist(), "{}"))
        # ...and then puts aimed squarely at the partitions sigma_0 concentrates into. The records
        # are the same under both bits; what differs is whether those partitions have room.
        targeted = np.where(np.isin(assign, np.arange(arm.get("concentrated", 2))))[0]
        if len(targeted):
            for _ in range(RHO_TARGETED):
                r = int(targeted[prho.integers(0, len(targeted))])
                with contextlib.suppress(Exception):
                    ids.append(eng.put(f"t{r}", vecs[r].tolist(), "{}"))
        lab.drain(eng)
        eng.flush()
        if ids:
            with contextlib.suppress(Exception):
                eng.delete(ids[0])
        eng.flush()
        refused = int(eng.stat().get("refused", 0))
        if trace.exists():
            for line in trace.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    if e["kind"] == "region_write":
                        rho_regions.append(int(e["region"]))
                        for s in e.get("slots", []) or []:
                            rho_slots.append((int(e["region"]), int(s)))
        with contextlib.suppress(Exception):
            eng.close()
        return features(snap, rho_regions, rho_slots, B), b, refused
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def heldout_accuracy(
    y_tr: np.ndarray, s_tr: np.ndarray, y_te: np.ndarray, s_te: np.ndarray, srng, n_boot: int = 4000
) -> tuple[float, float, float]:
    """Balanced accuracy on the held-out half, at a threshold fixed on the training half.

    AUC alone bounds information only loosely: the tightest general statement is that some
    threshold achieves Youden's J >= AUC - 1/2, which caps even a perfect separation at a 0.19-bit
    Fano bound. The leakage ledger needs the accuracy an attacker actually *commits to*, and the
    threshold has to be chosen without looking at the data it is scored on, or the number is
    optimistic by construction.

    ``srng`` is a generator of its own. The split and the AUC intervals share one generator across
    arms, and ``advance_split_rng`` replays that sequence exactly; drawing from it here would break
    the replay.
    """
    cand = np.unique(s_tr)
    best_t, best_ba = cand[0], -1.0
    for t in cand:
        pred = s_tr >= t
        ba = 0.5 * (pred[y_tr == 1].mean() + (~pred[y_tr == 0]).mean())
        if ba > best_ba:
            best_t, best_ba = t, ba
    pred = s_te >= best_t

    def ba_of(idx: np.ndarray) -> float:
        yy, pp = y_te[idx], pred[idx]
        if yy.min() == yy.max():
            return float("nan")
        return 0.5 * (pp[yy == 1].mean() + (~pp[yy == 0]).mean())

    n = len(y_te)
    base = ba_of(np.arange(n))
    boots = [ba_of(srng.integers(0, n, n)) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(base), float(lo), float(hi)


def auc_ci(y: np.ndarray, s: np.ndarray, rng, n_boot: int = 4000) -> tuple[float, float, float]:
    base = roc_auc_score(y, s)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(base), float(lo), float(hi)


# The statistic a human would look at for each ablation, rather than a learned combination.
HAND = {
    "P no S1 (growing regions)": ("region_size_std", "spread of per-region file sizes"),
    "P no S2 (unsealed slots)": ("zero_blocks", "count of all-zero 512-byte blocks"),
    "P no S3 (per-slot writes)": ("rho_slot_mean", "mean slot index of post-tau per-slot writes"),
    "P identity permutation": ("rho_low_region_share", "share of post-tau writes in regions 0-3"),
    "P inadmissible, silent refusal": ("rho_write_std", "spread of post-tau writes over regions"),
    "P inadmissible, padded refusal": ("rho_write_std", "spread of post-tau writes over regions"),
    "P shipped": ("rho_write_std", "spread of post-tau writes over regions"),
    "M shipped": ("rho_write_std", "spread of post-tau writes over regions"),
    "M content placement": ("rho_write_std", "spread of post-tau writes over regions"),
    "P first-free slot (S3 kept)": ("rho_slot_mean", "mean slot index of post-tau per-slot writes"),
    "P no S3, first-free slot": ("rho_slot_mean", "mean slot index of post-tau per-slot writes"),
}


def advance_split_rng(rng, n_instances: int, n_boot: int = 4000) -> None:
    """Consume exactly the draws one arm's evaluation consumes, without evaluating it.

    The held-out split and both bootstrap intervals draw from one generator shared across arms,
    so arm k's split depends on every arm before it. Running a subset of arms would therefore give
    an arm a different split from the one a full run gives it, and the same arm would report
    different numbers depending on how it was invoked. The draw count per arm is fixed (one
    shuffle of 2n indices, then two bootstraps of ``n_boot`` resamples of the n held-out
    instances), so replaying it for the skipped arms makes a subset run reproduce a full run
    exactly.
    """
    n = 2 * n_instances
    idx = np.arange(n)
    rng.shuffle(idx)
    n_te = n - n // 2
    for _ in range(2):
        for _ in range(n_boot):
            rng.integers(0, n_te, n_te)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=1000, help="game instances per bit")
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ][:3000]
    vecs = embed_cached([r["text"] for r in rows])
    cents = lab.kmeans_centroids(vecs, B, seed=0)
    assign = np.argmax(vecs @ cents.T, axis=1)

    unknown = set(args.arms or []) - {a["name"] for a in ARMS}
    if unknown:
        sys.exit(f"unknown arm(s): {sorted(unknown)}")
    rng = np.random.default_rng(20260916)
    results = {
        "experiment": "E5",
        "buckets": B,
        "capacity": CAP,
        "instances_per_bit": args.instances,
        "note": "held-out AUC; the upper bound is the empirical ceiling for this attacker",
        "rows": [],
    }
    # A subset run keeps every other arm's row rather than overwriting the file with only the
    # arms it ran: the full 1,000-per-bit game takes hours, and losing it to a two-arm rerun is the
    # failure this guards against. Rows from a different instance count are not mixed in.
    kept: dict[str, dict] = {}
    if args.arms is not None and OUT.exists():
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        if prev.get("instances_per_bit") == args.instances:
            kept = {r["arm"]: r for r in prev.get("rows", [])}
        else:
            sys.exit(
                f"{OUT.name} holds {prev.get('instances_per_bit')} instances per bit, this run "
                f"{args.instances}; refusing to merge rows measured at different power"
            )

    def emit() -> None:
        order = [a["name"] for a in ARMS]
        results["rows"] = sorted(kept.values(), key=lambda r: order.index(r["arm"]))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

    for arm in ARMS:
        if args.arms is not None and arm["name"] not in args.arms:
            advance_split_rng(rng, args.instances)
            continue
        t0 = time.time()
        X, y, refusals = [], [], 0
        for i in range(args.instances):
            for b in (0, 1):
                f, lab_b, ref = play(arm, vecs, cents, assign, b, seed=10_000 * i + b)
                X.append(f)
                y.append(lab_b)
                refusals += ref
        names = sorted(X[0])
        Xm = np.array([[x[n] for n in names] for x in X])
        y = np.array(y)

        # Held-out split, stratified by construction (bits alternate).
        idx = np.arange(len(y))
        rng.shuffle(idx)
        cut = len(idx) // 2
        tr, te = idx[:cut], idx[cut:]
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, random_state=0)
        clf.fit(Xm[tr], y[tr])
        s_te = clf.predict_proba(Xm[te])[:, 1]
        s_tr = clf.predict_proba(Xm[tr])[:, 1]
        auc, lo, hi = auc_ci(y[te], s_te, rng)
        srng = np.random.default_rng([20260927, [a["name"] for a in ARMS].index(arm["name"])])
        acc, acc_lo, acc_hi = heldout_accuracy(y[tr], s_tr, y[te], s_te, srng)

        # Hand-built: one named statistic, orientation chosen on the training split only.
        stat_name, stat_desc = HAND[arm["name"]]
        col = names.index(stat_name)
        sign = 1.0 if roc_auc_score(y[tr], Xm[tr, col]) >= 0.5 else -1.0
        h_auc, h_lo, h_hi = auc_ci(y[te], sign * Xm[te, col], rng)
        h_acc, h_acc_lo, h_acc_hi = heldout_accuracy(
            y[tr], sign * Xm[tr, col], y[te], sign * Xm[te, col], srng
        )

        row = {
            "arm": arm["name"],
            "engine": arm["kind"],
            "admissible": arm["admissible"],
            "refusals_observed": refusals,
            "learned": {
                "auc": auc,
                "lo": lo,
                "hi": hi,
                "advantage": abs(auc - 0.5),
                "accuracy": acc,
                "accuracy_lo": acc_lo,
                "accuracy_hi": acc_hi,
            },
            "n_heldout": len(te),
            "hand_built": {
                "statistic": stat_name,
                "description": stat_desc,
                "auc": h_auc,
                "lo": h_lo,
                "hi": h_hi,
                "advantage": abs(h_auc - 0.5),
                "accuracy": h_acc,
                "accuracy_lo": h_acc_lo,
                "accuracy_hi": h_acc_hi,
            },
            "distinguishable": lo > 0.5 or h_lo > 0.5,
            "ceiling": max(hi, h_hi) - 0.5,
            "seconds": time.time() - t0,
        }
        kept[arm["name"]] = row
        print(
            f"{arm['name']:>34}: learned {auc:.3f} [{lo:.3f},{hi:.3f}]  "
            f"hand({stat_name}) {h_auc:.3f} [{h_lo:.3f},{h_hi:.3f}]  "
            f"refusals={refusals}  "
            f"{'DISTINGUISHABLE' if row['distinguishable'] else 'at chance'}  "
            f"({row['seconds']:.0f}s)",
            flush=True,
        )
        emit()

    emit()
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
