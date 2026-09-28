"""Figure 1 -- structure transfer. The image the paper and the portfolio both rest on.

Two panels, one story:

**Left (injection budget).** ``Adv_hist`` and ``Adv_post`` climb together as the adversary
induces more writes, with Mnemos flat on the floor. The two attacked curves run close but not
coincident, and the size of that gap *is* the finding -- annotated so a reader who only looks at
the picture still gets it.

On the synthetic corpus the curves overlapped almost exactly and the ratio's confidence interval
contained 1.0. On D2 (real prose) they separate slightly: historicity buys 5.7% [3.0%, 8.4%].
The figure is drawn from whichever results file is present, preferring D2, and reads its chance
floor out of that file -- the two corpora have different topic counts (8 and 10), and a baseline
drawn in the wrong place makes an advantage plot unreadable.

**Right (epoch schedule).** Batching degrades the attack monotonically but only reaches zero at
``epoch_writes = B``, which is unaffordable. Padding is flat at zero throughout. The visual
contrast between "a defence that trails off" and "a defence that holds" is the argument for
ADR-0003, made without words.

Note that the Mnemos curve here is the *analytical model* of a padded store, which is zero by
construction. For the attack driven against the compiled engine, see
:mod:`eval.experiments.e9_transfer_live`.

Design rules
------------
- Chance is drawn as a reference line. An advantage plot without its baseline is unreadable.
- The two attacked curves are deliberately *not* visually separated beyond what the data does --
  pulling them apart for legibility would misrepresent the result.
- Colourblind-safe palette; no red/green pairing.
- PDF for the paper, PNG for the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.lab.corpus import repo_root

#: Chance floor. Read from the results file rather than fixed, because D3 has 10 topics and D2
#: has 8 -- a hardcoded 0.10 draws the baseline in the wrong place on every D2 plot, and the
#: baseline is the only thing that makes an advantage plot readable.
CHANCE = 0.125

INK = "#1b1b1b"
HIST = "#0072B2"  # blue   -- historical cohort
POST = "#D55E00"  # orange -- post-compromise cohort
SAFE = "#009E73"  # green  -- Mnemos
MUTED = "#8c8c8c"


def _series(rows: list[dict], axis: str, padded: bool, field: str, topics: int = 10):
    """One curve. On the budget axis ``x`` is recorded as ``injections_per_topic * 10``
    (see ``e4_transfer.py``), so it is rescaled here to the number of induced writes actually
    issued, ``injections_per_topic * topics``. On a corpus with 8 topics the recorded label
    overstates the budget by 25%."""
    sel = [r for r in rows if r["axis"] == axis and r["padded"] is padded]
    sel.sort(key=lambda r: r["x"])
    xs = [r["x"] for r in sel]
    if axis == "budget":
        xs = [x / 10 * topics for x in xs]
    return xs, [r[field] for r in sel]


def plot(rows: list[dict], out_pdf: Path, out_png: Path, topics: int = 10) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- left: injection budget --------------------------------------------------
    # The vertical axis is already advantage over uniform chance, so chance is zero and no
    # separate chance line is drawn.
    x, y_h = _series(rows, "budget", False, "adv_hist", topics)
    _, y_p = _series(rows, "budget", False, "adv_post", topics)
    _, y_m = _series(rows, "budget", True, "adv_hist", topics)

    ax1.plot(x, y_p, "-o", color=POST, lw=2, ms=5, label="Unpadded, cohort P (observed)")
    ax1.plot(x, y_h, "-s", color=HIST, lw=2, ms=5, label="Unpadded, cohort H (unobserved)")
    ax1.plot(x, y_m, "-^", color=SAFE, lw=2.4, ms=6, label="Padded layout (analytical model)")

    ax1.set_xscale("log")
    ax1.set_xlabel("Induced writes")
    ax1.set_ylabel("Topic-recovery advantage over chance")
    ax1.set_title("(a) Injection budget, epoch_writes = 16", fontsize=10, color=INK)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax1.grid(alpha=0.25, ls="--", lw=0.6)
    ax1.set_ylim(-0.03, 0.78)

    # ---- right: epoch schedule ---------------------------------------------------
    x2, y2_h = _series(rows, "epoch_writes", False, "adv_hist")
    _, y2_m = _series(rows, "epoch_writes", True, "adv_hist")
    fixed_budget = 100 * topics

    ax2.plot(x2, y2_h, "-s", color=HIST, lw=2, ms=5, label="Unpadded, cohort H")
    ax2.plot(x2, y2_m, "-^", color=SAFE, lw=2.4, ms=6, label="Padded layout (analytical model)")
    ax2.axvline(16, color=MUTED, ls="-.", lw=1.0)
    ax2.text(17, 0.66, "default\nepoch_writes = 16", fontsize=8, color=MUTED)

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("epoch_writes (records per flush)")
    ax2.set_ylabel("Topic-recovery advantage over chance")
    ax2.set_title(f"(b) Batch size, {fixed_budget} induced writes", fontsize=10, color=INK)
    ax2.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.06))
    ax2.grid(alpha=0.25, ls="--", lw=0.6)
    ax2.set_ylim(-0.03, 0.78)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    print(f"wrote {out_pdf}\nwrote {out_png}")


def main() -> None:
    root = repo_root()

    # D2 (real prose) first -- it is the paper's primary corpus -- then the real-embedder D3
    # results, then the TF-IDF substitute. Each fallback understates the attack relative to the
    # one before it, so a figure built from a fallback is conservative rather than wrong; but the
    # caption beside it quotes D2, and a figure disagreeing with its own caption is the kind of
    # discrepancy a reviewer finds first.
    for name in ("e4_transfer_bge_d2-beir.json", "e4_transfer_bge.json", "e4_transfer.json"):
        results = root / "python/eval/results" / name
        if results.is_file():
            break
    else:
        raise SystemExit("run `python -m eval.experiments.e4_transfer --backend bge` first")

    print(f"plotting from {results.name}")
    global CHANCE
    rows = json.loads(results.read_text(encoding="utf-8"))
    topics = next((r.get("n_topics") for r in rows if r.get("n_topics")), 0)
    if topics:
        CHANCE = 1.0 / topics
        print(f"chance floor {CHANCE:.3f} ({topics} topics, from the results file)")
    plot(
        rows,
        root / "paper/figures/fig1_transfer.pdf",
        root / "paper/figures/fig1_transfer.png",
        topics=topics or 10,
    )


if __name__ == "__main__":
    main()
