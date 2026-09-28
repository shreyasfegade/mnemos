"""Regenerate every figure the paper includes, from the results currently on disk.

One entry point, so `eval.repro` has one thing to call and a reviewer has one command to run. It
is deliberately dumb: each figure module already knows where its inputs are and where its output
goes, and duplicating that knowledge here would give the two copies somewhere to disagree.

**A figure module that fails is a failure of the run**, not a warning. A paper with four of its
five figures regenerated and one silently stale is worse than a paper with none regenerated,
because the stale one still looks current.

Usage::

    cd python && python -m eval.figures.emit_all
"""

from __future__ import annotations

import importlib
import sys
import traceback

#: In the order they appear in the paper.
FIGURES = [
    ("fig1_transfer", "structure transfer: advantage vs budget, D2 and D3"),
    ("fig2_threat_model", "the tau-arrival adversary and the two cohorts"),
    ("fig3_lattice", "the dual-epoch security lattice"),
    ("fig4_scaling", "search latency vs corpus size, log-log"),
    ("fig5_mitigation", "query-equality recovery and cost vs probe spread"),
]


def main() -> None:
    failures: list[str] = []
    for name, description in FIGURES:
        print(f"  {name:<18} {description} ...", end="", flush=True)
        try:
            mod = importlib.import_module(f"eval.figures.{name}")
            mod.main()
            print(" ok")
        except Exception:
            print(" FAILED")
            traceback.print_exc()
            failures.append(name)

    if failures:
        print(f"\n{len(failures)} figure(s) failed: {', '.join(failures)}")
        print("The remaining figures are current; these are not. Do not build the paper until")
        print("this is resolved -- a stale figure is indistinguishable from a fresh one.")
        sys.exit(1)
    print(f"\n{len(FIGURES)} figures regenerated.")


if __name__ == "__main__":
    main()
