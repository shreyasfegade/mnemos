"""E3 -- attack A3 on the forward-private baselines vs Mnemos. **Figure 1.**

The experiment the whole project turns on, and the one that gates it twice:

    GATE G2 (2026-09-09)  A3 must reach Adv_hist >= 0.50 on Sophos AND Diana at 25% aux.
                          If not, the finding is weaker than claimed -> the specification s10.2.

    GATE G4 (2026-10-14)  A3 must stay at Adv_hist <= 0.05 against Mnemos at 50% aux.
                          If not, history independence is not doing its job.

G2 comes first on purpose. It runs against **baselines only**, before a single line of Mnemos
exists. If forward-private snapshots turn out not to leak the way ``THREAT-MODEL.md`` s3.2
claims, that is a five-week saving rather than a November catastrophe.

Grid
----
    schemes  x  aux fractions      x  N            x  trials
    ---------------------------------------------------------
    sophos      0.10, 0.25, 0.50      1k, 10k         30
    diana       0.10, 0.25, 0.50      1k, 10k         30
    mnemos      0.10, 0.25, 0.50      1k, 10k         30

Reporting
---------
Report **all three schemes at all three aux fractions**, including any configuration where
Mnemos does worse than hoped. Figure 1's persuasive power comes from the gap between two
honestly-drawn curves; a curve with points removed persuades nobody who checks, and the people
who check are the reviewers.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from eval.attacks import a3_snapshot

SCHEMES = ("sophos", "diana", "mnemos")
AUX_FRACTIONS = (0.10, 0.25, 0.50)
CORPUS_SIZES = (1_000, 10_000)
TRIALS = 30

#: Gate thresholds, from params.toml [targets.security].
GATE_G2_BASELINE_MIN = 0.50
GATE_G4_MNEMOS_MAX = 0.05


def check_gates(results: list[a3_snapshot.AttackResult]) -> dict[str, bool]:
    """Evaluate G2 and G4 against the measured results.

    Returns a verdict per gate rather than raising: a failed gate is a decision point in
    ``the specification`` s10, not a crash. The correct response to G2 failing is to take fallback
    s10.2 with five weeks in hand -- which only works if the failure is reported clearly
    instead of aborting the run.
    """

    def by(scheme, aux):
        return [r.advantage for r in results if r.scheme == scheme and r.aux_fraction == aux]

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    g2 = (
        mean(by("sophos", 0.25)) >= GATE_G2_BASELINE_MIN
        and mean(by("diana", 0.25)) >= GATE_G2_BASELINE_MIN
    )
    g4 = mean(by("mnemos", 0.50)) <= GATE_G4_MNEMOS_MAX

    return {"G2_baselines_leak": g2, "G4_mnemos_resists": g4}


def main(out: Path) -> None:
    results: list[a3_snapshot.AttackResult] = []
    for n in CORPUS_SIZES:
        store_paths = {s: Path(f"data/stores/{s}-{n}") for s in SCHEMES}
        results.extend(a3_snapshot.sweep(store_paths, trials=TRIALS))

    verdict = check_gates(results)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"results": [asdict(r) for r in results], "gates": verdict},
            indent=2,
        ),
        encoding="utf-8",
    )

    for gate, passed in verdict.items():
        print(f"{gate}: {'PASS' if passed else 'FAIL'}")
    if not verdict["G2_baselines_leak"]:
        print("\nG2 FAILED — the snapshot finding is weaker than claimed.")
        print("Take the fallback in the specification s10.2 now, while there is time to.")


if __name__ == "__main__":
    main(Path("python/eval/results/e3_snapshot.json"))
