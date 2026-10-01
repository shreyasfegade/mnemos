"""Aggregate E16 per-seed runs into results/e16_real_layouts.json (adv_hist with a CI across seeds).

    cd python && python -m eval.e8.aggregate_e16
"""

from __future__ import annotations

import json
from collections import defaultdict

from eval.exp2 import lab
from eval.lab import corpus as C

SRC = C.repo_root() / "python" / "eval" / "e8" / "results16"
OUT = C.repo_root() / "python" / "eval" / "results" / "e16_real_layouts.json"


def main() -> None:
    by_store: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(SRC.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        by_store[d["store"]].append(d)

    stores = []
    for store, runs in sorted(by_store.items()):
        advs = [r["adv_hist"] for r in runs]
        m, lo, hi = lab.boot_ci(advs)
        r0 = runs[0]
        stores.append(
            {
                "store": store,
                "seeds": len(runs),
                "placement": r0.get("placement"),
                "schema": r0.get("schema"),
                "chance": r0.get("chance"),
                "n_hist": r0.get("n_hist"),
                "budget_per_topic": r0.get("budget_per_topic"),
                "block_size": r0.get("block_size"),
                "adv_hist": m,
                "adv_hist_lo": lo,
                "adv_hist_hi": hi,
                "coverage_hist": sum(r.get("coverage_hist", 0.0) for r in runs) / len(runs),
                "mean_blocks_per_write": sum(r.get("mean_blocks_per_write", 0.0) for r in runs)
                / len(runs),
            }
        )
        print(f"{store}: adv_hist {m:+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(runs)} seeds")

    OUT.write_text(json.dumps({"experiment": "E16", "stores": stores}, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
