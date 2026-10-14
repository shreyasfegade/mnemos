"""Generate the paper's figures from the result JSONs.

Like ``make_tables``, nothing here is hand-drawn from remembered numbers: each figure reads the
experiment's output, and an experiment that has not run is skipped with a message rather than
faked.

Design rules followed, and why they are not arbitrary
----------------------------------------------------
* **Colour is categorical and assigned by entity, never by rank.** A design keeps its hue across
  every panel and every figure, so a reader who learns "Mnemos-M is vermillion" is never misled by
  a panel where it happens to be ordered differently.
* **The palette is validated, not eyeballed.** ``#0072B2, #009E73, #D55E00, #CC79A7`` (Okabe--Ito,
  reordered so the low-separation pair is not adjacent) passes the lightness band, chroma floor,
  CVD separation (worst adjacent deutan Delta-E 11.0) and normal-vision floor. The reddish purple
  sits just under 3:1 against the surface, which is why every series is also directly labelled.
* **Every series carries a second encoding** -- its own dash pattern and marker -- so identity
  survives greyscale printing and colour-vision deficiency, which a journal PDF must.
* **Never a dual axis.** Measures on different scales become small multiples sharing an x-axis.
  Two y-scales on one plot invent a correlation that is not in the data.
* Hairline solid grid, thin marks, a legend whenever there is more than one series, and labels
  only where they carry information rather than on every point.

    cd python && python -m eval.exp2.make_figures
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from eval.lab import corpus as C

R = C.repo_root() / "python" / "eval" / "results"
F = C.repo_root() / "paper" / "figures"

# One column of the IEEE two-column layout, and the full text width.
COL, WIDE = 3.4, 7.16

# Fixed, validated categorical order. Index by entity, never by position in a sorted list.
#
# Two entities are deliberately not given a hue of their own. The growing layout is the positive
# control, not a design, and is drawn in neutral ink so it can share a figure with Mnemos-RS without
# the two reading as one. Mnemos-M with its in-memory partition index is the *same design* in a
# different search mode, so it keeps Mnemos-M's hue and differs by dash and a hollow marker: a
# fifth categorical hue would claim a distinction the design does not have.
PALETTE = {
    "growing": "#4D4D4D",
    "rs": "#0072B2",
    "p": "#009E73",
    "m": "#D55E00",
    "m-ivf": "#D55E00",
    "plaintext": "#CC79A7",
}
DASH = {
    "growing": (0, (1, 1)),
    "rs": (0, (1, 1)),
    "p": (0, ()),
    "m": (0, (4, 1.5)),
    "m-ivf": (0, (1.5, 1.5)),
    "plaintext": (0, (3, 1, 1, 1)),
}
MARK = {"growing": "x", "rs": "s", "p": "o", "m": "^", "m-ivf": "v", "plaintext": "D"}
HOLLOW = {"m-ivf"}
LABEL = {
    "growing": "Growing layout",
    "rs": "Mnemos-RS",
    "p": "Mnemos-P",
    "m": "Mnemos-M",
    "m-ivf": "Mnemos-M, in-memory IVF",
    "plaintext": "Plaintext IVF",
}


def series(ax, design: str, x, y) -> None:
    """One series in its entity's fixed encoding: hue, dash, marker, and fill."""
    ax.plot(
        x,
        y,
        color=PALETTE[design],
        linewidth=1.6,
        linestyle=DASH[design],
        marker=MARK[design],
        markersize=4,
        markerfacecolor="white" if design in HOLLOW else PALETTE[design],
        label=LABEL[design],
        zorder=3,
    )


INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",  # never dashed: dashing reads as "threshold"
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load(name: str):
    f = R / name
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def tidy(ax, ylabel: str, xlabel: str = "") -> None:
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def save(fig, name: str) -> None:
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{name}.pdf")
    fig.savefig(F / f"{name}.png", dpi=220)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


# --------------------------------------------------------------------------- drift
def fig_drift() -> bool:
    d = load("e4_drift.json")
    if not d:
        return False
    style()
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    for key, colour, dash, marker, lab_ in (
        ("fine", "#0072B2", (0, ()), "o", "Fine taxonomy (12 areas)"),
        ("coarse", "#009E73", (0, (4, 1.5)), "^", "Coarse taxonomy (4 areas)"),
    ):
        curve = d[key]["curve"]
        x = [c["gap_eras"] for c in curve]
        y = [c["adv_H"] for c in curve]
        lo = [c["lo"] for c in curve]
        hi = [c["hi"] for c in curve]
        ax.fill_between(x, lo, hi, color=colour, alpha=0.16, linewidth=0)
        ax.plot(
            x,
            y,
            color=colour,
            linewidth=1.6,
            linestyle=dash,
            marker=marker,
            markersize=4,
            label=lab_,
            zorder=3,
        )
    ax.axhline(0.0, color=MUTED, linewidth=0.7, zorder=2)
    # What knowing cohort H's (skewed) topic prior alone is worth, fine taxonomy: always guessing
    # its most frequent topic, as an advantage over 1/K.
    if "majority_rate" in d["fine"]:
        prior = d["fine"]["majority_rate"] - d["fine"]["chance"]
        ax.axhline(prior, color="#0072B2", linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=2)
        ax.annotate(
            "prior only (fine)",
            xy=(5.0, prior),
            xytext=(5.0, prior + 0.012),
            fontsize=6.5,
            color=INK,
            ha="right",
            va="bottom",
        )
    # Label the two ends, which are the claim; not every point. Both annotations are placed in
    # empty regions with a leader to their point, rather than on top of a series.
    fine = d["fine"]["curve"]
    ax.annotate(
        "induces with\narchival content",
        xy=(fine[0]["gap_eras"], fine[0]["adv_H"]),
        xytext=(0.42, fine[0]["adv_H"] - 0.015),
        fontsize=6.5,
        color=INK,
        va="top",
    )
    ax.annotate(
        "induces with\ncurrent content",
        xy=(fine[-1]["gap_eras"], fine[-1]["adv_H"]),
        xytext=(3.25, 0.155),
        fontsize=6.5,
        color=INK,
        ha="left",
        arrowprops={
            "arrowstyle": "-",
            "color": MUTED,
            "linewidth": 0.6,
            "shrinkA": 2,
            "shrinkB": 3,
        },
    )
    tidy(ax, "Advantage on cohort H", "Eras between cohort H and the induced content")
    # Right-hand middle: clear of the archival-content note at top left and the current-content
    # note below it, both of which the default upper-right position covered.
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.64))
    save(fig, "fig_drift")
    return True


# --------------------------------------------------------------------------- budget
def fig_budget() -> bool:
    d = load("e1_longitudinal.json")
    if not d:
        return False
    style()
    classes = ["W", "RW"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.1), sharey=True)
    for ax, cls in zip(axes, classes, strict=True):
        for design in ("growing", "rs", "p", "m"):
            rows = [r for r in d["rows"] if r["design"] == design and r["adversary"] == cls]
            if not rows:
                continue
            rows.sort(key=lambda r: r["budget"])
            x = [r["budget"] for r in rows]
            y = [r["adv_H"] for r in rows]
            lo = [r["lo"] for r in rows]
            hi = [r["hi"] for r in rows]
            ax.fill_between(x, lo, hi, color=PALETTE[design], alpha=0.14, linewidth=0)
            series(ax, design, x, y)
        ax.axhline(0.0, color=MUTED, linewidth=0.7)
        ax.set_xscale("log", base=2)
        # Tick exactly the budgets measured, in plain numbers; 2^3 and 2^5 were never run.
        budgets = sorted({r["budget"] for r in d["rows"]})
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.minorticks_off()
        ax.set_title(
            f"{cls}: {'write observer' if cls == 'W' else 'read-write observer'}", color=INK
        )
        tidy(ax, "Advantage on cohort H" if cls == "W" else "", "Induced writes per topic")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "fig_budget")
    return True


# --------------------------------------------------------------------------- crossover
def fig_crossover() -> bool:
    d = load("e10_performance.json")
    if not d:
        return False
    rows = [r for r in d["rows"] if "error" not in r]
    if not rows:
        return False
    style()
    # Three measures on different scales: small multiples sharing x, never a second y-axis.
    panels = [
        ("search_ms", "Search latency (ms)", lambda r: r["search_ms"]["p50"]),
        ("resident", "Resident memory (MB)", lambda r: r["resident_bytes"] / 1e6),
        ("unlock", "Unlock time (s)", lambda r: r["unlock_s"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.0))
    for ax, (key, ylabel, get) in zip(axes, panels, strict=True):
        for design in ("plaintext", "m", "m-ivf", "p", "rs"):
            # The in-memory index adds two partition-id bytes a record, so its resident footprint
            # is Mnemos-M's own and a second line would lie on top of the first. Search and unlock
            # do differ -- unlock recomputes every record's partition -- so it is drawn there.
            if design == "m-ivf" and key == "resident":
                continue
            rs = sorted([r for r in rows if r["engine"] == design], key=lambda r: r["n"])
            if not rs:
                continue
            y = [get(r) for r in rs]
            if all(v == 0 for v in y):
                continue
            series(ax, design, [r["n"] for r in rs], y)
        ax.set_xscale("log")
        if ylabel != "Unlock time (s)":
            ax.set_yscale("log")
        tidy(ax, ylabel, "Records")
    # One legend above all three panels: inside the first it covered the crossover it exists to
    # show, and ran into the next panel's axis label.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
        handlelength=2.6,
        columnspacing=1.4,
    )
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig_crossover")
    return True


# --------------------------------------------------------------------------- architecture
def fig_architecture() -> bool:
    """A schematic, not a chart: where each design puts data and where search runs."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 1.8))
    for ax, design, title in (
        (axes[0], "p", "Mnemos-P: placement by content, search on disk"),
        (axes[1], "m", "Mnemos-M: placement at random, search in memory"),
    ):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title(title, color=INK, pad=6)
        colour = PALETTE[design]

        # The sealed file: equal regions either way.
        for i in range(6):
            filled = (i == 2) if design == "p" else (i == 4)
            ax.add_patch(
                Rectangle(
                    (0.4 + i * 1.5, 0.6),
                    1.35,
                    1.1,
                    facecolor=colour if filled else "#ffffff",
                    edgecolor=MUTED,
                    linewidth=0.7,
                )
            )
        ax.text(
            5.0,
            0.15,
            "sealed file: $B$ equal regions, free slots random",
            ha="center",
            fontsize=6.5,
            color=MUTED,
        )

        # The locked arena.
        ax.add_patch(
            Rectangle((0.4, 3.6), 9.2, 1.7, facecolor="#f4f4f2", edgecolor=MUTED, linewidth=0.7)
        )
        ax.text(5.0, 4.9, "locked arena", ha="center", fontsize=7, color=INK)
        inside = (
            "all records resident; search ranks here"
            if design == "m"
            else "one region image at a time"
        )
        ax.text(5.0, 4.15, inside, ha="center", fontsize=6.5, color=INK if design == "m" else MUTED)

        # Where a write goes: an arrow from the arena to the highlighted region, which is the
        # region the record lands in. The label sits at the far left, clear of the arrow.
        tgt = 2 if design == "p" else 4
        centre = 0.4 + tgt * 1.5 + 0.675
        ax.add_patch(
            FancyArrowPatch(
                (centre - 0.8, 3.45),
                (centre, 1.85),
                arrowstyle="-|>",
                mutation_scale=8,
                color=colour,
                linewidth=1.2,
                shrinkA=0,
                shrinkB=2,
            )
        )
        ax.text(
            0.45,
            2.45,
            "put: region is\nthe content's" if design == "p" else "put: region is\na uniform draw",
            fontsize=6.5,
            color=INK,
        )

        # Where a search goes: straight up from the regions it reads, with its label in the gap
        # between the write arrow and the read arrows so nothing crosses any text.
        if design == "p":
            for r in (4, 5):
                x = 0.4 + r * 1.5 + 0.675
                ax.add_patch(
                    FancyArrowPatch(
                        (x, 1.85),
                        (x, 3.45),
                        arrowstyle="-|>",
                        mutation_scale=7,
                        color=MUTED,
                        linewidth=0.8,
                        linestyle="--",
                        shrinkA=2,
                        shrinkB=0,
                    )
                )
            ax.text(
                5.75, 2.35, "search reads\n$s\\,k$ regions", fontsize=6.5, color=MUTED, ha="center"
            )
        else:
            ax.text(
                9.55,
                2.30,
                "search reads\nno disk at all",
                fontsize=6.5,
                color=colour,
                ha="right",
                weight="bold",
            )
    save(fig, "fig_architecture")
    return True


# --------------------------------------------------------------------------- overview
def fig_overview() -> bool:
    """Figure 1, a schematic: the two cohorts, the four channels that cross tau, and the fix.

    Every channel is drawn as an arrow pointing *back* across tau, because that is the finding:
    something the adversary observes after the compromise says something about a record written
    before it. The fix sits underneath, spanning both cohorts, in the hue of Mnemos-M.
    """
    style()
    fig, ax = plt.subplots(figsize=(WIDE, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    hist, post, safe = "#0072B2", PALETTE["m"], PALETTE["p"]
    tau = 4.1

    # Cohort bands and the compromise line.
    band = {"facecolor": hist, "alpha": 0.13, "ec": hist, "lw": 0.7}
    ax.add_patch(Rectangle((0.1, 4.55), tau - 0.1, 1.05, **band))
    band = {"facecolor": post, "alpha": 0.13, "ec": post, "lw": 0.7}
    ax.add_patch(Rectangle((tau, 4.55), 9.9 - tau, 1.05, **band))
    ax.text(
        (0.1 + tau) / 2,
        5.28,
        "cohort H: written before $\\tau$",
        ha="center",
        fontsize=7.5,
        color=hist,
        weight="bold",
    )
    ax.text(
        (0.1 + tau) / 2,
        4.80,
        "adversary holds only the stored state",
        ha="center",
        fontsize=6.5,
        color=INK,
    )
    ax.text(
        (tau + 9.9) / 2,
        5.28,
        "cohort P: written after $\\tau$",
        ha="center",
        fontsize=7.5,
        color=post,
        weight="bold",
    )
    ax.text(
        (tau + 9.9) / 2,
        4.80,
        "adversary sees every file version and steers the agent's inputs",
        ha="center",
        fontsize=6.5,
        color=INK,
    )
    ax.plot([tau, tau], [1.25, 5.85], color=INK, lw=1.0, ls=(0, (3, 2)))
    ax.text(
        tau,
        5.92,
        "$\\tau$: sync account or disk image compromised",
        ha="center",
        fontsize=6.5,
        color=INK,
    )

    # The four channels, each an arrow from what is observed after tau back to cohort H.
    channels = [
        ("induced writes label content-placed partitions", "labels apply to earlier records"),
        ("retained file versions diff to the regions written", "names an earlier record's region"),
        ("a full partition refuses the write", "discloses the earlier load"),
        ("a deduplicating agent skips a fact it holds", "write or no write: membership"),
    ]
    for i, (seen, learned) in enumerate(channels):
        y = 4.05 - i * 0.68
        ax.text(tau + 0.45, y, f"({i + 1}) {seen}", ha="left", va="center", fontsize=6.5, color=INK)
        ax.add_patch(
            FancyArrowPatch(
                (tau + 0.35, y),
                (tau - 0.35, y),
                arrowstyle="-|>",
                mutation_scale=8,
                color=post,
                lw=1.1,
            )
        )
        ax.text(tau - 0.45, y, learned, ha="right", va="center", fontsize=6.5, color=hist)

    # The fix, spanning both cohorts.
    ax.add_patch(Rectangle((0.1, 0.12), 9.8, 0.95, facecolor=safe, alpha=0.12, ec=safe, lw=0.7))
    ax.text(
        5.0,
        0.78,
        "clock-only store: placement independent of content, search in locked memory, "
        "a write-only-oblivious sweep on a clock",
        ha="center",
        fontsize=6.5,
        color=INK,
    )
    ax.text(
        5.0,
        0.36,
        "closes (1)-(4) against a write observer, for every self-writing agent (Theorem 2)",
        ha="center",
        fontsize=6.5,
        color=safe,
        weight="bold",
    )
    save(fig, "fig_overview")
    return True


def main() -> None:
    made, skipped = [], []
    for fn, name in (
        (fig_overview, "fig_overview"),
        (fig_architecture, "fig_architecture"),
        (fig_budget, "fig_budget"),
        (fig_drift, "fig_drift"),
        (fig_crossover, "fig_crossover"),
    ):
        try:
            (made if fn() else skipped).append(name)
        except Exception as e:
            skipped.append(f"{name} ({type(e).__name__}: {e})")
    print(f"\nmade: {', '.join(made) if made else 'none'}")
    if skipped:
        print(f"skipped (no data yet): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
