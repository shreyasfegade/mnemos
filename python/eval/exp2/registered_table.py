"""The supplement's table of every registered prediction and its outcome.

Each row is a check from ``verify_r4``: the claim it binds, the range registered before the run
as read from the registration commits in git (so a failed prediction shows what was predicted, not
the widened range the verifier now holds), the measured
value, and whether it held. Generated, so the table and the verifier cannot disagree.

    cd python && python -m eval.exp2.registered_table
"""

from __future__ import annotations

import subprocess
import types

from eval.exp2 import verify_r4
from eval.lab import corpus as C

OUT = C.repo_root() / "paper" / "tables" / "supp_registered.tex"
#: The commits that registered the predictions, before their runs: the original ranges are read
#: from them, so a failed prediction shows what was predicted, not the widened range.
REGISTRATIONS = ("c5a2300", "c302948")


def registered_ranges() -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for commit in REGISTRATIONS:
        src = subprocess.run(
            ["git", "show", f"{commit}:python/eval/exp2/verify_r4.py"],
            cwd=C.repo_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        mod = types.ModuleType(f"verify_r4_{commit}")
        exec(compile(src, f"verify_r4@{commit}", "exec"), mod.__dict__)
        for check, _src in mod.CHECKS:
            out.setdefault(check.name, (check.lo, check.hi))
    return out


def tex(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def main() -> None:
    measured: dict[str, float] = {}
    for fn in verify_r4.EXTRACTORS.values():
        measured.update(fn())
    original = registered_ranges()
    rows = []
    for check, _src in verify_r4.CHECKS:
        v = measured.get(check.name)
        failed = "PREDICTION FAILED" in check.why
        if v is None:
            outcome = "not run"
        elif failed:
            outcome = r"\textbf{failed}"
        else:
            outcome = "held"
        lo, hi = original.get(check.name, (check.lo, check.hi))
        reg = f"[{lo:g}, {hi:g}]"
        rows.append(
            f"{tex(check.name)} & {tex(reg)} & {'--' if v is None else f'{v:.3f}'} & {outcome} \\\\"
        )
    body = "\n".join(rows)
    OUT.write_text(
        r"""\begin{table*}[t]
\centering
\caption{Every prediction registered for the later experiments, committed before its run,
with the measured value. A failed prediction is kept; the verifier's range for it is widened only
to the claim that survives, and its note records the original prediction and the numbers.}
\label{tab:supp-registered}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{p{0.52\textwidth}p{0.22\textwidth}rl}
\toprule
Check & Registered & Measured & Outcome \\
\midrule
"""
        + body
        + r"""
\bottomrule
\end{tabular}
\end{table*}
""",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(C.repo_root())} with {len(rows)} rows")


if __name__ == "__main__":
    main()
