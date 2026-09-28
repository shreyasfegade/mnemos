"""Figure 4 -- search latency against corpus size, on log-log axes.

The performance claim is a *shape* claim: Mnemos's probe width grows as `sqrt(n)` because the
partition count does, while a scheme that scans a posting list grows linearly. A table of two
columns cannot show a shape, and the paper currently has only a table.

Why this figure had to wait
---------------------------
An earlier draft reported the overhead *falling* with corpus size and Mnemos overtaking
Sophos. Both were artifacts of a fixed `buckets = 64, capacity = 128` geometry: at `n = 10^4`
the index held 8,192 slots for 10,000 entries, so 18% of the corpus was silently never indexed
and the "flat" curve was measuring a store that did not contain the data. The geometry is now
sized per corpus (`B ~ sqrt(n)`, `C` scaled to fit), and this figure plots what that actually
costs --- which is more, not less.

Reading the figure
------------------
Slope on log-log axes *is* the exponent. A line of slope 1 is linear; slope 0.5 is `sqrt(n)`.
Both reference slopes are drawn faintly so a reader can measure the curves against them by eye
rather than taking our word for the exponent.

Data source
-----------
Criterion's own `estimates.json`, read directly from `target/criterion/`, so the plotted number
is the same median the benchmark harness reported and no figure-specific transcription step
exists to disagree with the table.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.lab.corpus import repo_root

INK = "#1b1b1b"
MUTED = "#8c8c8c"

#: (criterion group, label, colour, marker). Order sets legend order.
SERIES = [
    ("plaintext_ivf", "Plaintext IVF", "#8c8c8c", "o"),
    ("sophos_fp", "Sophos-FP", "#0072B2", "s"),
    ("diana_fp", "Diana-FP", "#56B4E9", "^"),
    ("mnemos_d0", "Mnemos ($d{=}0$)", "#009E73", "D"),
    ("mnemos_d4", "Mnemos ($d{=}4$)", "#D55E00", "v"),
    ("naive_ets", "Naive EtS", "#CC79A7", "x"),
]


def read_estimates() -> dict[str, dict[int, float]]:
    """Median search latency in microseconds, per scheme, per corpus size."""
    root = repo_root() / "target" / "criterion" / "search"
    out: dict[str, dict[int, float]] = {}
    if not root.is_dir():
        return out
    for group, *_ in SERIES:
        gdir = root / group
        if not gdir.is_dir():
            continue
        points: dict[int, float] = {}
        for size_dir in gdir.iterdir():
            if not size_dir.is_dir() or not size_dir.name.isdigit():
                continue
            # Named baselines, not `new/`. `new/` is whatever ran last, so a figure reading it
            # silently mixes runs: after measuring the mitigation at s = 2, the Mnemos curves
            # came from that run and every baseline curve from the previous one. Mnemos is
            # plotted in its **shipped** configuration (s = 2) and the baselines from the same
            # frozen run as the table, so the figure and `tab:perf` cannot disagree.
            preferred = ("frozen-s2",) if group.startswith("mnemos") else ()
            est = None
            for candidate in (*preferred, "frozen-s1", "new", "base"):
                path = size_dir / candidate / "estimates.json"
                if path.is_file():
                    est = path
                    break
            if est is None:
                continue
            data = json.loads(est.read_text(encoding="utf-8"))
            # Criterion reports nanoseconds.
            median = data.get("median", {}).get("point_estimate")
            if median is None:
                continue
            points[int(size_dir.name)] = float(median) / 1000.0
        if points:
            out[group] = points
    return out


def build(data: dict[str, dict[int, float]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    for group, label, colour, marker in SERIES:
        pts = data.get(group)
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[x] for x in xs]
        ax.plot(xs, ys, marker=marker, ms=5.5, lw=1.6, color=colour, label=label)

    # Reference slopes, anchored at the smallest measured plaintext point so they sit in range.
    # Anchored on the Mnemos curve, because the sqrt(n) claim is about Mnemos and a
    # reference slope drawn somewhere else makes the reader do the translation.
    anchor = data.get("mnemos_d0") or next(iter(data.values()), None)
    if anchor:
        x0 = min(anchor)
        y0 = anchor[x0]
        xs = np.array([x0, max(max(p) for p in data.values())], dtype=float)
        ax.plot(xs, y0 * (xs / x0) ** 1.0, ls="--", lw=0.9, color=MUTED, alpha=0.7)
        ax.text(
            xs[-1],
            y0 * (xs[-1] / x0) ** 1.0,
            "  slope 1 (linear)",
            fontsize=7.5,
            color=MUTED,
            va="center",
        )
        ax.plot(xs, y0 * (xs / x0) ** 0.5, ls=":", lw=0.9, color=MUTED, alpha=0.7)
        ax.text(
            xs[-1],
            y0 * (xs[-1] / x0) ** 0.5,
            "  slope 0.5 ($\\sqrt{n}$)",
            fontsize=7.5,
            color=MUTED,
            va="center",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size $n$ (entries)", fontsize=9)
    ax.set_ylabel("median search latency ($\\mu s$)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2, which="both", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def main() -> None:
    data = read_estimates()
    if not data:
        raise SystemExit(
            "no criterion estimates found under target/criterion/search. Run:\n"
            "  MNEMOS_BENCH_DIM=768 cargo bench -p mnemos-bench --bench search"
        )
    sizes = sorted({n for pts in data.values() for n in pts})
    print(f"schemes: {len(data)}  sizes: {sizes}")
    for group, pts in data.items():
        print(f"  {group:16s} " + "  ".join(f"n={n}: {pts[n]:.1f}us" for n in sorted(pts)))

    out = repo_root() / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig = build(data)
    for ext in ("pdf", "png"):
        path = out / f"fig4_scaling.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"wrote {path.relative_to(repo_root())}")
    plt.close(fig)


if __name__ == "__main__":
    main()
