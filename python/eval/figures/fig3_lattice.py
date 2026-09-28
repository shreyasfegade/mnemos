"""Figure 3 -- the leakage-ordering lattice, and where each scheme sits in it.

Theorem 1 orders two leakage profiles. Corollary 1 says which structural property is
*necessary* for that ordering to be strict. Both are currently prose plus a table, and the
relationship between them is exactly the kind of thing a lattice diagram carries better than
either.

The claim the picture has to make
---------------------------------
``L_pre`` is below ``L_post`` in every scheme --- the historical cohort never leaks *more*.
That is the easy half and it is not the finding. The finding is that **the gap can be zero**,
and whether it is zero depends on one structural property: whether the on-disk layout is
history-independent. A scheme with an append-structured layout sits at the degenerate point
where the two profiles coincide, no matter how good its forward privacy is.

So the figure is two columns, not one axis: forward privacy runs one way, history independence
runs the other, and they are orthogonal. Sophos and Diana are strong on the first and absent on
the second, which is why they land in the same cell as a scheme with no forward privacy at all
*for this particular question*.

Design rules
------------
- The vertical axis is leakage, increasing upward, so "lower is safer" matches intuition.
- The measured `Adv_hist` for each scheme is printed on its marker. A lattice with no numbers
  is a diagram of an opinion.
- Mnemos appears twice: as measured against the *model* and as measured against the *live
  engine* (E9). Showing only the favourable one would repeat the mistake that finding F5 was.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from eval.lab.corpus import repo_root

INK = "#1b1b1b"
HIST = "#0072B2"
POST = "#D55E00"
SAFE = "#009E73"
MUTED = "#8c8c8c"


def _live_adv() -> float | None:
    """Best measured `Adv_hist` against the live engine, if E9 has been run."""
    path = repo_root() / "python" / "eval" / "results" / "e9_transfer_live.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not rows:
        return None
    return max(r["hist_correct"] / max(r["hist_total"], 1) for r in rows)


def build() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    live = _live_adv()

    # (label, x, adv_hist, colour, note)
    points = [
        ("Naive EtS", 0.0, 0.88, POST, "no forward privacy,\nappend layout"),
        ("Sophos-FP", 1.0, 0.876, POST, "forward-private,\nappend layout"),
        ("Diana-FP", 1.6, 0.874, POST, None),
        ("Mnemos\n(model)", 2.9, 0.000, SAFE, "padded + history-independent"),
    ]
    if live is not None:
        points.append(("Mnemos\n(live, E9)", 3.6, live, SAFE, None))

    for label, x, adv, colour, note in points:
        ax.plot([x], [adv], marker="o", ms=11, color=colour, mec="white", mew=1.4, zorder=3)
        ax.annotate(
            f"{adv:.3f}",
            (x, adv),
            textcoords="offset points",
            xytext=(0, 13),
            ha="center",
            fontsize=8.5,
            color=INK,
            weight="bold",
        )
        ax.annotate(
            label,
            (x, adv),
            textcoords="offset points",
            xytext=(0, -26),
            ha="center",
            fontsize=8.5,
            color=INK,
        )
        if note:
            ax.annotate(
                note,
                (x, adv),
                textcoords="offset points",
                xytext=(0, -48),
                ha="center",
                fontsize=7,
                color=MUTED,
            )

    ax.axhline(0.10, color=MUTED, ls=":", lw=1.1)
    ax.text(3.95, 0.115, "chance", ha="right", va="bottom", fontsize=8, color=MUTED)

    # The structural boundary: everything left of it has an append-structured layout.
    ax.axvline(2.35, color=INK, ls="--", lw=1.0, alpha=0.55)
    ax.text(
        2.28,
        0.62,
        "append-structured layout\n$\\Rightarrow$ gap collapses",
        ha="right",
        va="center",
        fontsize=8,
        color=INK,
    )
    ax.text(
        2.44,
        0.62,
        "history-independent\nlayout",
        ha="left",
        va="center",
        fontsize=8,
        color=SAFE,
    )

    # The one comparison the figure exists to make.
    ax.add_patch(
        FancyArrowPatch(
            (1.6, 0.80),
            (2.9, 0.09),
            arrowstyle="-|>",
            mutation_scale=12,
            color=INK,
            lw=1.3,
            connectionstyle="arc3,rad=-0.25",
            alpha=0.75,
        )
    )
    ax.text(
        2.25,
        0.42,
        "Corollary 1",
        ha="center",
        va="center",
        fontsize=8,
        color=INK,
        style="italic",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )

    ax.set_ylim(-0.16, 1.02)
    ax.set_xlim(-0.55, 4.1)
    ax.set_ylabel(
        "$\\mathrm{Adv}_{\\mathrm{hist}}$  (topic recovery on the unobserved cohort)", fontsize=9
    )
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.18, lw=0.7)
    ax.set_axisbelow(True)

    return fig


def main() -> None:
    out = repo_root() / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig = build()
    for ext in ("pdf", "png"):
        path = out / f"fig3_lattice.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"wrote {path.relative_to(repo_root())}")
    plt.close(fig)


if __name__ == "__main__":
    main()
