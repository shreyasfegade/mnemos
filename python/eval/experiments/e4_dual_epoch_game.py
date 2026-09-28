"""E4: the dual-epoch security game, played against the live v2 engine, with ablations.

The game (paper Def. of dual-epoch security)
============================================
The adversary picks two put-only cohort-H histories sigma0 and sigma1 of equal length. The
challenger flips a bit b, writes sigma_b before tau, then runs a common post-tau sequence rho
(puts, deletes, searches) while the adversary observes the tau snapshot and every post-tau disk
delta. The adversary guesses b; its advantage is |Pr[correct] - 1/2|.

We make the two histories maximally different so that any residual dependence shows up: sigma0
concentrates all records into a handful of partitions, sigma1 spreads them uniformly. Under S1-S4
the tau snapshot is byte-independent of the contents and the post-tau trace depends only on rho,
so no distinguisher does better than chance. Each ablation removes one condition and should let a
distinguisher win.

Distinguisher
=============
A gradient-boosted classifier trained across many game instances on features of the tau snapshot
and the post-tau trace: per-region file sizes (for the growing-regions ablation), total store
size, count of distinct 512-byte blocks (entropy proxy for the unsealed ablation), the multiset
of regions written during rho, and the manifest size. Advantage is reported as AUC - 0.5 with a
95% bootstrap interval; AUC uses cross-validated out-of-fold predictions so the classifier cannot
memorise instances.

    cd python && python -m eval.experiments.e4_dual_epoch_game
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from eval.lab import corpus as C
from eval.lab.embcache import embed_cached
from eval.lab.stores import create_v2

OUT = C.repo_root() / "python" / "eval" / "results" / "e4_dual_epoch_game.json"
B = 32
Cap = 32
DIM = 768
H_LEN = 64
RHO_PUTS = 16
INSTANCES = 60
CONCENTRATED = 4  # sigma0 uses this many partitions

ABLATIONS = {
    "shipped": {},
    "growing_regions": {"growing_regions": True},
    "unsealed_slots": {"unsealed_slots": True},
    "slot_writes_only": {"slot_writes_only": True},
    "no_padding": {"no_padding": True},
    "identity_permutation": {"identity_permutation": True},
}


def store_files(d: Path) -> dict[str, bytes]:
    out = {}
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.name != "k":
            out[str(f.relative_to(d)).replace("\\", "/")] = f.read_bytes()
    return out


def features(snap: dict[str, bytes], rho_regions: list[int], geo_regions: int) -> list[float]:
    part = snap.get("partitions.bin", b"")
    feats: list[float] = [float(len(part)), float(len(snap.get("manifest.enc", b"")))]
    # Growing-regions ablation exposes per-region files; summarise their size distribution.
    region_sizes = sorted(len(v) for k, v in snap.items() if k.startswith("regions/"))
    if region_sizes:
        arr = np.array(region_sizes, dtype=float)
        feats += [arr.sum(), arr.std(), arr.max(), float((arr > 0).sum())]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]
    # Distinct 512-byte blocks in the partition file: low for zero-padded (unsealed) free slots.
    if part:
        blocks = {part[i : i + 512] for i in range(0, len(part), 512)}
        zero = sum(
            1
            for i in range(0, len(part), 512)
            if part[i : i + 512] == bytes(min(512, len(part) - i))
        )
        feats += [float(len(blocks)), float(zero)]
    else:
        feats += [0.0, 0.0]
    # rho region histogram: which physical regions the post-tau puts touched.
    hist = Counter(rho_regions)
    feats += [float(hist.get(r, 0)) for r in range(geo_regions)]
    return feats


def play(ablation: dict, vecs: np.ndarray, cents, b: int, seed: int) -> tuple[list[float], int]:
    rng = np.random.default_rng(seed)
    tmp = Path(tempfile.mkdtemp())
    over = {
        "embedding": {"dim": DIM},
        "index": {
            "buckets": B,
            "bucket_capacity": Cap,
            "top_k": 10,
            "probe_width": 4,
            "probe_spread": 1,
        },
        "crypto": {"block_size_bytes": 1024, "max_entry_bytes": 1024},
        "epoch": {"epoch_writes": 16, "epoch_ms": 3_600_000},
        "memory": {"arena_bytes": 1 << 20},
        "v2": {"durable": False, "flush_policy": "on_demand"},
    }
    eng, _, _ = create_v2(tmp / "s", cents, over, ablation)

    # cohort H, written before tau. b picks the history.
    if b == 0:
        # concentrated: choose records whose assigned partition is one of a few.
        assign = np.argmax(vecs @ cents.T, axis=1)
        pool = np.where(np.isin(assign, np.arange(CONCENTRATED)))[0]
    else:
        pool = np.arange(len(vecs))
    for _ in range(H_LEN):
        r = int(rng.choice(pool))
        with contextlib.suppress(Exception):
            eng.put(f"h{r}", vecs[r].tolist(), "{}")
    eng.flush()

    snap = store_files(tmp / "s")  # the tau snapshot

    # cohort P (common rho): puts drawn uniformly, plus a search and a delete.
    trace = tmp / "t.jsonl"
    eng.set_trace_sink(str(trace))
    rho_regions: list[int] = []
    rids = []
    for _ in range(RHO_PUTS):
        r = int(rng.integers(0, len(vecs)))
        try:
            mid = eng.put(f"p{r}", vecs[r].tolist(), "{}")
            rids.append(mid)
        except Exception:
            pass
    eng.flush()
    if rids:
        eng.delete(rids[0])
    eng.flush()
    if trace.exists():
        for line in trace.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                if e["kind"] == "region_write":
                    rho_regions.append(int(e["region"]))
    eng.close()
    return features(snap, rho_regions, B), b


def auc_ci(y: np.ndarray, s: np.ndarray, rng, n_boot: int = 5000) -> tuple[float, float, float]:
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


def main() -> None:
    rows = [
        json.loads(ln)
        for ln in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if ln.strip()
    ][:3000]
    vecs = embed_cached([r["text"] for r in rows])
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=B, random_state=0, n_init=2).fit(vecs)
    cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)

    rng = np.random.default_rng(20260915)
    results = {"experiment": "E4", "buckets": B, "instances_per_bit": INSTANCES, "rows": []}
    for name, abl in ABLATIONS.items():
        X, y = [], []
        for i in range(INSTANCES):
            for b in (0, 1):
                f, lab = play(abl, vecs, cents, b, seed=10_000 * i + b + hash(name) % 997)
                X.append(f)
                y.append(lab)
        X = np.array(X)
        y = np.array(y)
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, random_state=0)
        oof = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
        auc, lo, hi = auc_ci(y, oof, rng)
        adv = abs(auc - 0.5)
        row = {
            "ablation": name,
            "auc": auc,
            "auc_lo": lo,
            "auc_hi": hi,
            "advantage": adv,
            "distinguishable": lo > 0.5 or hi < 0.5,
        }
        results["rows"].append(row)
        print(
            f"{name:>22}: AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  adv {adv:.3f}  "
            f"{'DISTINGUISHABLE' if row['distinguishable'] else 'at chance'}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
