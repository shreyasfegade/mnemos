"""Figure 5 -- the query-equality mitigation as a design space, not a promise.

A limitations section that says "this could be mitigated by randomising the probe set" costs
nothing to write and tells a reader nothing. The interesting content is the exchange rate: how
much read volume, and how much recall, buys how much reduction in recovery.

Two panels because there are two costs and they are paid by different knobs
--------------------------------------------------------------------------
**Left --- spread (`s`), which costs read volume.** Widening the probe set with fresh randomness
each call breaks the identity between two searches for the same query, so pairwise recovery
collapses. It does *not* break intersection: the real probes are in every observation, so an
adversary who can group repetitions still recovers them. Both curves are drawn, and the gap
between them is the point.

**Right --- retention (`p`), which costs recall.** Dropping each genuine probe with probability
`1-p` makes a real probe survive `r` intersected repetitions with probability only `p^r`, so
the intersection curve finally falls too --- and Recall@10 falls with it. Plotting recovery and
recall on the same axes is the only honest way to present a defence that trades one for the
other.

Design rules
------------
- Recovery and recall share a 0-1 axis, so no twin-axis sleight of hand is needed.
- Pairwise recovery is plotted as **Youden's J** (true-positive minus false-positive rate), not
  as raw recall. Raw recall is gameable: an adversary who answers "same query" to every pair
  scores 1.000 while learning nothing, which is exactly what a probe set covering the whole
  index produces. J scores that adversary at 0, which is the truth.
- The `s=1, p=1.0` point (Mnemos as shipped, as measured by A5) is marked on both panels, so
  the reader can see where the system currently sits before any mitigation.
- Recall is drawn dashed to separate "cost" from "leakage" at a glance.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.lab.corpus import repo_root

INK = "#1b1b1b"
PAIRWISE = "#D55E00"
INTERSECT = "#0072B2"
RECALL = "#009E73"
MUTED = "#8c8c8c"


def load() -> list[dict]:
    path = repo_root() / "python" / "eval" / "results" / "e10_probe_spread.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m eval.experiments.e10_probe_spread` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def build(rows: list[dict]) -> plt.Figure:
    fig, (ax_s, ax_p) = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)

    # ---- left: spread sweep at p = 1.0 --------------------------------------------------
    spread_rows = sorted(
        (r for r in rows if abs(r["retention"] - 1.0) < 1e-9), key=lambda r: r["spread"]
    )
    xs = [r["spread"] for r in spread_rows]
    ax_s.plot(
        xs,
        [r["advantage"] for r in spread_rows],
        marker="o",
        ms=5.5,
        lw=1.7,
        color=PAIRWISE,
        label="query equality (pairwise, TPR $-$ FPR)",
    )
    ax_s.plot(
        xs,
        [r["intersection_recovery"] for r in spread_rows],
        marker="s",
        ms=5.5,
        lw=1.7,
        color=INTERSECT,
        label="recovery under intersection",
    )
    ax_s.plot(
        xs,
        [r["recall_at_10"] for r in spread_rows],
        marker="D",
        ms=5,
        lw=1.5,
        ls="--",
        color=RECALL,
        label="Recall@10",
    )

    ax_s.set_xlabel("probe spread $s$  (read volume multiple)", fontsize=9)
    ax_s.set_ylabel("recovery  /  recall", fontsize=9)
    ax_s.set_xticks(xs)
    ax_s.set_title("Spread costs read volume", fontsize=9.5, color=INK)

    # ---- right: retention sweep at s = 4 ------------------------------------------------
    ret_rows = sorted((r for r in rows if r["spread"] == 4), key=lambda r: -r["retention"])
    xp = [r["retention"] for r in ret_rows]
    ax_p.plot(xp, [r["advantage"] for r in ret_rows], marker="o", ms=5.5, lw=1.7, color=PAIRWISE)
    ax_p.plot(
        xp,
        [r["intersection_recovery"] for r in ret_rows],
        marker="s",
        ms=5.5,
        lw=1.7,
        color=INTERSECT,
    )
    ax_p.plot(
        xp, [r["recall_at_10"] for r in ret_rows], marker="D", ms=5, lw=1.5, ls="--", color=RECALL
    )

    ax_p.set_xlabel("probe retention $p$  (at $s=4$)", fontsize=9)
    ax_p.set_xticks(xp)
    ax_p.invert_xaxis()  # stronger mitigation to the right, matching the left panel
    ax_p.set_title("Retention costs recall", fontsize=9.5, color=INK)

    # Mark where the shipped system sits.
    shipped = next((r for r in rows if r["spread"] == 1 and abs(r["retention"] - 1.0) < 1e-9), None)
    if shipped:
        ax_s.annotate(
            "Mnemos as shipped\n(A5: recovery 1.000)",
            (1, shipped["advantage"]),
            textcoords="offset points",
            xytext=(16, -6),
            fontsize=7.5,
            color=MUTED,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        )

    for ax in (ax_s, ax_p):
        ax.set_ylim(-0.05, 1.08)
        ax.grid(alpha=0.2, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = ax_s.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=8,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout()
    return fig


def main() -> None:
    rows = load()
    out = repo_root() / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig = build(rows)
    for ext in ("pdf", "png"):
        path = out / f"fig5_mitigation.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"wrote {path.relative_to(repo_root())}")
    plt.close(fig)


if __name__ == "__main__":
    main()
