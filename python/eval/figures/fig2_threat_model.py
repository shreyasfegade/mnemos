"""Figure 2 -- the tau-arrival adversary, drawn.

The threat model is the paper's actual contribution and it is currently carried entirely by
prose. A reader has to hold four things in their head simultaneously -- two cohorts, two
observation regimes, the direction structure transfer travels, and the fact that partitions are
shared between cohorts -- and prose delivers them one at a time.

What the figure has to make unmissable
--------------------------------------
1. **The adversary arrives partway through.** Everything left of tau was written unobserved;
   everything right of it is fully observed and partly induced.
2. **Partitions are semantic, not temporal.** A partition holding post-tau SQL memories holds
   pre-tau SQL memories too. This is the whole mechanism and it is a *spatial* fact, which is
   why a picture beats a paragraph.
3. **The map is learned rightwards and applied leftwards.** One arrow, pointing back across
   tau, is the finding.

Design rules
------------
- Time runs left to right, because every reader already expects that of a timeline.
- The two cohorts are the same colours as in Figure 1 (blue historical, orange post), so a
  reader who has seen one figure can read the other without re-learning the palette.
- The partition band sits *below* the timeline and spans both cohorts, because the point is
  that it does not respect tau.
- No red/green pairing anywhere.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from eval.lab.corpus import repo_root

INK = "#1b1b1b"
HIST = "#0072B2"
POST = "#D55E00"
SAFE = "#009E73"
MUTED = "#8c8c8c"
FAINT = "#e8e8e8"


def build() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.1, 5.9)
    ax.axis("off")

    tau = 5.2

    # ---- cohort bands ------------------------------------------------------------------
    ax.add_patch(Rectangle((0.4, 3.5), tau - 0.4, 0.9, facecolor=HIST, alpha=0.18, ec=HIST))
    ax.add_patch(Rectangle((tau, 3.5), 9.6 - tau, 0.9, facecolor=POST, alpha=0.18, ec=POST))

    ax.text(
        (0.4 + tau) / 2,
        4.60,
        "cohort H  ·  written before",
        ha="center",
        va="bottom",
        fontsize=9,
        color=HIST,
        weight="bold",
    )
    ax.text(
        (tau + 9.6) / 2,
        4.60,
        "cohort P  ·  written after",
        ha="center",
        va="bottom",
        fontsize=9,
        color=POST,
        weight="bold",
    )

    ax.text(
        (0.4 + tau) / 2,
        3.95,
        "only the on-disk state at compromise\n(no operations, no timing)",
        ha="center",
        va="center",
        fontsize=7.5,
        color=INK,
    )
    ax.text(
        (tau + 9.6) / 2,
        3.95,
        "all storage-level activity observed;\nwrites can be induced",
        ha="center",
        va="center",
        fontsize=7.5,
        color=INK,
    )

    # ---- the timeline itself -----------------------------------------------------------
    ax.annotate(
        "",
        xy=(9.7, 3.1),
        xytext=(0.4, 3.1),
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.2},
    )
    ax.text(9.7, 2.82, "time", ha="right", va="top", fontsize=8, color=MUTED)

    ax.plot([tau, tau], [1.90, 5.28], color=INK, lw=1.6, ls="--")
    ax.text(
        tau,
        5.34,
        r"$\tau$  compromise",
        ha="center",
        va="bottom",
        fontsize=10,
        color=INK,
        weight="bold",
    )

    # ---- the partition band ------------------------------------------------------------
    # Deliberately spans tau: this is the mechanism, stated geometrically.
    band_y = 1.05
    ax.add_patch(Rectangle((0.4, band_y), 9.2, 0.85, facecolor=FAINT, ec=MUTED, lw=0.8))
    ax.text(
        0.4,
        band_y + 0.95,
        "index partitions (by content, not time)",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )

    # A few partitions, each drawn with entries from BOTH sides of tau.
    for i, cx in enumerate([1.6, 3.4, 6.2, 8.2]):
        ax.add_patch(
            Rectangle((cx - 0.62, band_y + 0.1), 1.24, 0.65, facecolor="white", ec=INK, lw=0.9)
        )
        ax.text(
            cx, band_y + 0.62, f"$B_{{{i}}}$", ha="center", va="center", fontsize=7.5, color=MUTED
        )
        # entries: blue = historical, orange = post. Every partition holds both.
        for j, colour in enumerate([HIST, HIST, POST, HIST, POST]):
            ax.plot(
                cx - 0.45 + j * 0.225, band_y + 0.3, marker="s", ms=4.2, color=colour, mec="none"
            )

    # ---- the two arrows that carry the argument ----------------------------------------
    # 1. induced writes go into partitions, and the adversary watches which.
    ax.add_patch(
        FancyArrowPatch(
            (7.2, 3.42),
            (7.2, 1.98),
            arrowstyle="-|>",
            mutation_scale=11,
            color=POST,
            lw=1.4,
        )
    )
    ax.text(
        7.35,
        2.72,
        "induced writes\nlabel partitions",
        ha="left",
        va="center",
        fontsize=7.5,
        color=POST,
    )

    # 2. the learned map applies BACKWARDS. This arrow is the paper.
    ax.add_patch(
        FancyArrowPatch(
            # Routed BELOW the partition band. Threading it between the boxes made the
            # arrow read as pointing at one partition rather than at the whole cohort.
            (7.4, band_y - 0.16),
            (1.5, band_y - 0.16),
            arrowstyle="-|>",
            mutation_scale=13,
            color=INK,
            lw=1.8,
            connectionstyle="arc3,rad=-0.10",
        )
    )
    ax.text(
        5.0,
        0.22,
        "partition labels learned after $\\tau$ apply to entries written before $\\tau$",
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK,
        weight="bold",
    )

    return fig


def main() -> None:
    out = repo_root() / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig = build()
    for ext in ("pdf", "png"):
        path = out / f"fig2_threat_model.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"wrote {path.relative_to(repo_root())}")
    plt.close(fig)


if __name__ == "__main__":
    main()
