"""Aggregate E32, pgvector under gocryptfs at several capture granularities.

E16 snapshots the ciphertext after every insert, the finest capture an adversary could hope for.
A synchronization client coalesces changes, so the adversary's versions each carry several writes.
E32 repeats E16 with versions of ``g`` induced writes (``run_e6.py --checkpoint-every g``) and
reports ``adv_hist`` with a bootstrap interval across seeds at each ``g``.

    cd python && python -m eval.e8.aggregate_e32
"""

from __future__ import annotations

import json
from collections import defaultdict

from eval.exp2 import lab
from eval.lab import corpus as C

SRC = C.repo_root() / "python" / "eval" / "e8" / "results32"
OUT = C.repo_root() / "python" / "eval" / "results" / "e32_pg_granularity.json"


def main() -> None:
    by_g: dict[int, list[dict]] = defaultdict(list)
    for f in sorted(SRC.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        by_g[int(d.get("checkpoint_every", 1))].append(d)

    rows = []
    for g, runs in sorted(by_g.items()):
        m, lo, hi = lab.boot_ci([r["adv_hist"] for r in runs])
        loc_m, loc_lo, loc_hi = lab.boot_ci([r["localisation_heldout"] for r in runs])
        rows.append(
            {
                "checkpoint_every": g,
                "seeds": len(runs),
                "chance": runs[0]["chance"],
                "n_hist": runs[0]["n_hist"],
                "budget_per_topic": runs[0]["budget_per_topic"],
                "induced_versions": float(sum(r["induced_versions"] for r in runs) / len(runs)),
                "adv_hist": m,
                "adv_hist_lo": lo,
                "adv_hist_hi": hi,
                "acc_hist": m + runs[0]["chance"],
                "localisation_heldout": loc_m,
                "localisation_lo": loc_lo,
                "localisation_hi": loc_hi,
                "coverage_hist": sum(r["coverage_hist"] for r in runs) / len(runs),
                "seconds_per_run": sum(r["seconds"] for r in runs) / len(runs),
            }
        )
        print(f"g={g:>3}: adv_hist {m:+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(runs)} seeds")
    OUT.write_text(
        json.dumps({"experiment": "E32", "store": "pgvector", "rows": rows}, indent=1),
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
