"""E25: what Mnemos-M/disk costs, beside Mnemos-M and Mnemos-P.

The paper positions Mnemos-M/disk -- headers and payloads on disk, only the position map and logical
partition ids resident, writes by the clocked sweep -- as the design above the memory limit. E10
measured the other designs and not this one. This runs E10's benchmark (same geometry rule, same
corpus tiling, same timing of search, sparse puts and unlock) for the shipped sweep-flush Mnemos-M
with its in-memory index, for Mnemos-M/disk, and for Mnemos-P, at 10^4 and 10^5 records.

Beyond 10^5 the paper gives an analytical extrapolation and says where measurement stops.

    cd python && python -m eval.exp2.e25_scale
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from eval.exp2 import e10_performance as E10
from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e25_scale.json"
SIZES = (10_000, 100_000)

SWEEP = {"flush_mode": "sweep", "sweep_slots": 8, "journal_slots": 4}
E10.VARIANTS.update(
    {
        "m-sweep-ivf": ("m", {"m": {"search": "ivf", "probe_width": 8, **SWEEP}}),
        "m-disk": (
            "m",
            {
                "m": {
                    "search": "ivf",
                    "probe_width": 8,
                    "header_residency": "disk",
                    "payload_residency": "disk",
                    **SWEEP,
                }
            },
        ),
    }
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--engines", nargs="+", default=["m-sweep-ivf", "m-disk", "p"])
    args = ap.parse_args()
    t0 = time.time()

    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ][:4000]
    base = embed_cached([r["text"] for r in rows])
    need = max(args.sizes)
    vecs = np.tile(base, (int(np.ceil(need / len(base))), 1))[:need].astype(np.float32)

    out: dict = {"experiment": "E25", "dim": E10.DIM, "block": E10.BLOCK, "rows": []}
    rng = np.random.default_rng(25)
    for n in args.sizes:
        cents = lab.kmeans_centroids(base, E10.geometry(n)[0], seed=0)
        for kind in args.engines:
            r = E10.bench_engine(kind, n, vecs, cents, rng)
            r["resident_bytes_per_record"] = (
                r["resident_bytes"] / max(r.get("added", 0), 1) if "error" not in r else None
            )
            out["rows"].append(r)
            if "error" in r:
                print(f"n={n:>7} {kind:>12}: FAILED {r['error']}", flush=True)
            else:
                print(
                    f"n={n:>7} {kind:>12}: search p50 {r['search_ms']['p50']:8.3f} ms "
                    f"p95 {r['search_ms']['p95']:8.3f} | put(sparse) p50 "
                    f"{r['put_sparse_ms']['p50']:7.3f} ms | RAM {r['resident_bytes'] / 1e6:8.2f} MB "
                    f"| disk {r['disk_bytes'] / 1e6:8.1f} MB | unlock {r['unlock_s']:6.2f} s",
                    flush=True,
                )
            OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    out["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())} in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
