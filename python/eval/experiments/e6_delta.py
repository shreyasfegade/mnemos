"""E6 -- the asymmetry between cohorts, with confidence intervals.

    Delta   = Adv_post - Adv_hist
    ratio   = Adv_hist / Adv_post

The empirical form of Theorem 1: does the historical cohort leak strictly less than the
post-compromise one, and by how much?

Where the numbers come from, and why this file no longer measures anything
--------------------------------------------------------------------------
This was specified as a standalone experiment with its own sweep over ``epoch_writes`` and
``epoch_ms``. It never became one. E4 already runs both cohorts through the same attack over an
injection-budget sweep, and already bootstraps ``Delta`` and ``ratio`` **inside** each resample --
which is the part that is easy to get wrong, and the reason the two must not be computed by
separate runs. Adding a second measurement of the same quantity would give the paper two numbers
for one claim and no principled way to choose between them when they disagreed.

So this module *derives* E6's table from E4's results rather than re-measuring. It is a view, and
it says so. If a future version needs the epoch-schedule sweep the docstring below once promised,
that is a new experiment and it should say so in its own name.

The three outcomes, all publishable -- written down before any data existed, and kept
-------------------------------------------------------------------------------------
**Delta large and schedule-dependent.** Theorem 1 confirmed with a strong empirical story.

**Delta small but non-zero.** Theorem 1 confirmed, modestly.

**Delta ~ 0.** A null result, *reported as a null result*: the epoch schedule fully closes the gap
between cohorts, which is an actionable engineering guideline ("batch your writes and the
distinction disappears") and a legitimate finding.

What actually happened is the middle one, and it is corpus-dependent: on the public corpus the
interval for ``ratio`` excludes 1.0 (historicity buys 5.7%); on the synthetic corpus it contains
1.0 and no benefit is detectable at all. We report the former.

Usage::

    cd python && python -m eval.experiments.e6_delta
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from eval.lab import corpus as C

#: (results file, label). D2 is primary; D3 is reported alongside because the two disagree about
#: whether the asymmetry is resolvable at all, and hiding that would be the interesting half.
SOURCES = [
    ("e4_transfer_bge_d2-beir.json", "D2 (public, BEIR)"),
    ("e4_transfer_bge.json", "D3 (synthetic)"),
]


@dataclass(frozen=True)
class DeltaRow:
    """One injection budget, one corpus."""

    corpus: str
    budget: int
    adv_hist: float
    adv_post: float
    delta: float
    delta_lo: float
    delta_hi: float
    ratio: float | None
    ratio_lo: float | None
    ratio_hi: float | None
    defined: bool
    """Whether the ratio means anything at this budget.

    Below the budget at which the attack resolves at all, both cohorts sit at chance and the
    ratio is ``0/0``. E4 records ``None`` for its interval there, correctly. Rendering that as
    ``0.000`` would put a number in the table saying "the historical cohort is infinitely
    safer", when what happened is that the attack did not work on either cohort."""
    resolvable: bool
    """Whether the ratio interval excludes 1.0 -- i.e. whether historicity buys anything we can
    measure at all. This is the actual finding, not `ratio` itself."""


def rows_for(path: Path, label: str) -> list[DeltaRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    out: list[DeltaRow] = []
    for r in rows:
        # Unpadded only: the padded arm is Mnemos, where both cohorts sit at chance and the
        # ratio is a quotient of two zeroes.
        if r.get("padded") or r.get("axis") != "budget":
            continue
        lo, hi = r.get("ratio_lo"), r.get("ratio_hi")
        defined = lo is not None and hi is not None and r["adv_post"] > 0.0
        out.append(
            DeltaRow(
                corpus=label,
                budget=int(r["x"]),
                adv_hist=r["adv_hist"],
                adv_post=r["adv_post"],
                delta=r["delta"],
                delta_lo=r["delta_lo"],
                delta_hi=r["delta_hi"],
                ratio=r["ratio"] if defined else None,
                ratio_lo=lo if defined else None,
                ratio_hi=hi if defined else None,
                defined=defined,
                resolvable=bool(defined and not (lo <= 1.0 <= hi)),
            )
        )
    return sorted(out, key=lambda x: x.budget)


def main(out: Path | None = None) -> None:
    root = C.repo_root()
    results = root / "python" / "eval" / "results"

    all_rows: list[DeltaRow] = []
    for filename, label in SOURCES:
        path = results / filename
        if not path.is_file():
            print(f"  skipping {label}: {filename} not found (run E4 first)")
            continue
        all_rows.extend(rows_for(path, label))

    if not all_rows:
        raise SystemExit(
            "no E4 results found. E6 is a view over E4, not an independent measurement:\n"
            "  python -m eval.experiments.e4_transfer --backend bge --corpus "
            "data/d2-beir/notes.jsonl"
        )

    print("\n=== E6: cohort asymmetry (derived from E4) ===\n")
    print(
        f"  {'corpus':<20} {'budget':>7}  {'Adv_hist':>9} {'Adv_post':>9}  {'ratio [95% CI]':>26}"
    )
    for r in all_rows:
        mark = "*" if r.resolvable else " "
        if r.defined:
            interval = f"{r.ratio:.3f} [{r.ratio_lo:.3f}, {r.ratio_hi:.3f}]"
        else:
            # Not "0.000". The attack did not resolve on either cohort, so there is no ratio.
            interval = "undefined (0/0)"
        print(
            f"  {r.corpus:<20} {r.budget:>7}  {r.adv_hist:>9.3f} {r.adv_post:>9.3f}"
            f"  {interval:>26}{mark}"
        )
    print("\n  * interval excludes 1.0 -- historicity buys something measurable")
    print("  undefined: neither cohort resolved above chance, so the ratio is 0/0. Reporting")
    print("             it as 0.000 would read as 'the old data is infinitely safer' when")
    print("             what happened is that the attack failed on both.")

    dest = out or results / "e6_delta.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "experiment": "E6",
                "description": "cohort asymmetry with CIs; a view over E4, not a separate run",
                "derived_from": [s for s, _ in SOURCES],
                "rows": [asdict(r) for r in all_rows],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {dest.relative_to(root)}")


if __name__ == "__main__":
    main()
