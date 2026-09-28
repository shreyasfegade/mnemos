"""E6 driver: prepare embeddings on the GPU, run the deployed-store attack in WSL, collect.

The attack itself runs inside WSL, because that is where a FUSE mount and the four store
libraries live (``python/eval/e8/run_e6.py``). Embeddings are computed here, where the CUDA build
of torch is, and handed over as ``.npy``.

    cd python && python -m eval.exp2.e6_stores            # prepare, run, collect
    cd python && python -m eval.exp2.e6_stores --collect  # collect an already-finished run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np

from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e6_stores.json"
E8 = C.repo_root() / "python" / "eval" / "e8"
WSL_SCRIPT = (
    "/mnt/c/Users/Admin/Desktop/Projects/Code/03_Apps-Products/cybsersec/"
    "python/eval/e8/run_e6_wsl.sh"
)


def prepare(n_max: int) -> tuple[int, int]:
    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    names = sorted({r["topic"] for r in rows})
    idx = {t: i for i, t in enumerate(names)}
    rng = np.random.default_rng(0)
    order = rng.permutation(len(rows))[:n_max]
    rows = [rows[int(i)] for i in order]
    vecs = embed_cached([r["text"] for r in rows]).astype(np.float32)
    labels = np.array([idx[r["topic"]] for r in rows], dtype=np.int64)
    (E8 / "data").mkdir(parents=True, exist_ok=True)
    np.save(E8 / "data" / "emb.npy", vecs)
    np.save(E8 / "data" / "labels.npy", labels)
    print(f"prepared {len(rows)} records, {len(names)} topics -> {E8 / 'data'}", flush=True)
    return len(rows), len(names)


def collect() -> dict:
    """Aggregate the per-seed runs into one row per store, with intervals."""
    from collections import defaultdict

    from eval.exp2 import lab

    runs: dict[str, list[dict]] = defaultdict(list)
    for f in sorted((E8 / "results6").glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            runs[r["store"]].append(r)
        except Exception as e:
            print(f"  unreadable {f.name}: {e}")

    results = {"experiment": "E6", "stores": [], "raw": {k: v for k, v in runs.items()}}
    for store, rs in sorted(runs.items()):
        loc_m, loc_lo, loc_hi = lab.boot_ci([r["localisation_heldout"] for r in rs])
        # Taken from the raw fields rather than the stored ``adv_hist``, so runs written before the
        # guessing adversary was adopted are rescored identically: the guess credits each
        # unlabelled cohort-H record with 1/K (see run_e6.py). Nothing is re-measured.
        adv = [
            r["acc_hist_all"] + (1.0 - r["coverage_hist"]) * r["chance"] - r["chance"] for r in rs
        ]
        adv_m, adv_lo, adv_hi = lab.boot_ci(adv)
        abst_m, _, _ = lab.boot_ci([r["acc_hist_all"] - r["chance"] for r in rs])
        results["stores"].append(
            {
                "store": store,
                "seeds": len(rs),
                "placement": rs[0]["placement"],
                "schema": rs[0]["schema"],
                "time_placed": rs[0]["placement"].startswith("TIME"),
                "chance": rs[0]["chance"],
                "n_hist": rs[0]["n_hist"],
                "budget_per_topic": rs[0]["budget_per_topic"],
                "mean_blocks_per_write": float(np.mean([r["mean_blocks_per_write"] for r in rs])),
                "localisation_heldout": loc_m,
                "localisation_lo": loc_lo,
                "localisation_hi": loc_hi,
                "adv_hist": adv_m,
                "adv_hist_lo": adv_lo,
                "adv_hist_hi": adv_hi,
                "adv_hist_abstaining": abst_m,
                "coverage_hist": float(np.mean([r["coverage_hist"] for r in rs])),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(
        f"\n{'store':>16} {'placed by':>9} {'blk/wr':>7} {'loc (held-out)':>22} {'adv_hist':>22} "
        f"{'cov':>5}"
    )
    for s in results["stores"]:
        kind = "TIME" if s["time_placed"] else "content"
        print(
            f"{s['store']:>16} {kind:>9} {s['mean_blocks_per_write']:7.1f} "
            f"{s['localisation_heldout']:6.3f} "
            f"[{s['localisation_lo']:.3f},{s['localisation_hi']:.3f}] "
            f"{s['adv_hist']:+6.3f} [{s['adv_hist_lo']:+.3f},{s['adv_hist_hi']:+.3f}] "
            f"{s['coverage_hist']:5.2f}"
        )
    print(f"\nwrote {OUT.relative_to(C.repo_root())}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--stores", default=None, help="space-separated subset")
    ap.add_argument("--hist", type=int, default=300)
    ap.add_argument("--budget", type=int, default=30)
    args = ap.parse_args()

    if args.collect:
        collect()
        return

    t0 = time.time()
    prepare(args.n)
    env = []
    if args.stores:
        env.append(f"STORES='{args.stores}'")
    env.append(f"HIST={args.hist}")
    env.append(f"BUDGET={args.budget}")
    cmd = ["wsl", "-d", "Ubuntu", "--", "bash", "-c", f"{' '.join(env)} bash {WSL_SCRIPT}"]
    print("running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True, check=False)
    print(p.stdout[-8000:])
    if p.stderr.strip():
        print("stderr:", p.stderr[-3000:])
    collect()
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
