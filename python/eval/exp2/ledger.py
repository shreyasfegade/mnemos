"""A leakage ledger: every measured channel, in one unit.

Each experiment reports its channel in the unit natural to it -- an AUC, an advantage over a
chance rate, a precision against a 1/B null, a Youden J. That makes the design space impossible to
read on one axis: is a permutation recovered at 0.719 worse or better than a refusal channel at
AUC 1.000? This module converts every one of them into the same quantity, a **lower bound on the
mutual information, in bits, between the secret and what the adversary observes**, per observation
of the unit the channel is about.

The conversion is Fano's inequality. An adversary that guesses a uniformly distributed K-ary
secret S from its view V with accuracy a has

    I(S; V) >= log2 K - h(1 - a) - (1 - a) log2 (K - 1)            (a > 1/K)

and 0 otherwise, where h is the binary entropy. Two properties make this the right tool here.

* It is a *lower* bound, and a leakage measurement can only ever be one: an experiment exhibits an
  attacker, and a better attacker can only extract more. A ledger that reported point estimates of
  "the leakage" would be claiming what no experiment can establish.
* It needs only an accuracy, which every experiment either reports or implies, so every channel is
  converted by the same rule rather than by a per-experiment argument.

The ``conservative`` column applies Fano to the lower end of the 95% interval on the accuracy
rather than to the point estimate, so it is a bound that holds with the experiment's confidence
rather than one that holds on average.

**For the channels that measured at chance, a lower bound of zero is not the interesting number.**
The ``mde`` column is the minimum detectable effect: the smallest effect, in the experiment's own
unit, that the experiment had 80% power to detect at the 5% level, computed from the width of its
own interval. It turns "at chance" from an absence of evidence into a bounded statement -- this
attacker, at this sample size, would have seen anything at least this large -- and it is the only
honest way to report a null.

Two conversions need care, and are documented where they happen: AUC-only results (converted via
Youden's J, which is loose), and results with no interval at all (marked, never silently promoted).

Usage::

    cd python && python -m eval.exp2.ledger
"""

from __future__ import annotations

import json
import math

import numpy as np

from eval.lab import corpus as C

R = C.repo_root() / "python" / "eval" / "results"
OUT = R / "ledger.json"

Z_ALPHA = 1.959964  # two-sided 5%
Z_POWER = 0.841621  # 80% power


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def fano_bits(acc: float, k: int, h_x: float | None = None) -> float:
    """Fano lower bound on ``I(X;V)`` for a K-ary secret guessed with accuracy ``acc``.

    Fano's inequality gives ``H(X | V) <= h(Pe) + Pe*log2(K-1)``, hence
    ``I(X;V) = H(X) - H(X|V) >= H(X) - h(Pe) - Pe*log2(K-1)``. The submission used
    ``H(X) = log2 K``, which holds only for a **uniform** prior; under a skewed
    topic prior ``H(X) < log2 K``, so the ``log2 K`` form can exceed the true mutual information and
    is not a valid lower bound. We pass the **empirical** entropy ``h_x`` of the secret, measured
    from the label distribution, and default to ``log2 K`` only where the prior is uniform by
    construction --- the binary history games, whose challenge bit is a fair coin. The accuracy is
    compared against the prior's best constant guess (passed by the caller as the chance rate);
    below it the bound is zero.
    """
    if acc is None or not math.isfinite(acc):
        return 0.0
    hx = math.log2(k) if h_x is None else h_x
    pe = 1.0 - acc
    cond = h2(pe) + (pe * math.log2(k - 1) if k > 1 else 0.0)
    return max(0.0, hx - cond)


def entropy_of(counts) -> float:
    """Empirical Shannon entropy (bits) of a label distribution, for the Fano prior term."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def mde(lo: float, hi: float, n: int | None = None, metric: str = "") -> tuple[float | None, str]:
    """Minimum detectable effect at 80% power and 5% two-sided.

    From the interval's width where it has one. A **zero-width** interval is not infinite
    precision: it means the statistic took the same value in every resample -- the padded refusal
    rewrites the region either way, a clocked flush happens on every probe -- and the width then
    says nothing about power. For an AUC with a known sample size the detectable effect is taken
    from the Hanley-McNeil standard error at AUC = 0.5 instead; otherwise it is reported as
    undetermined, never as zero.
    """
    if hi - lo > 1e-12:
        return (Z_ALPHA + Z_POWER) * (hi - lo) / (2 * Z_ALPHA), "from the interval width"
    if metric == "AUC" and n:
        n0 = n1 = n / 2
        se = math.sqrt((n0 + n1 + 1) / (12 * n0 * n1))
        return (Z_ALPHA + Z_POWER) * se, f"Hanley-McNeil at n={n}; the interval is a point"
    return None, "undetermined: the statistic did not vary across resamples"


def load(name: str) -> dict | None:
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def row(
    channel: str,
    design: str,
    adversary: str,
    experiment: str,
    unit: str,
    k: int,
    acc: float,
    acc_lo: float | None,
    metric: str,
    value: float,
    lo: float | None,
    hi: float | None,
    null: float,
    how: str,
    n: int | None = None,
    h_x: float | None = None,
    prior: str = "uniform (assumed)",
) -> dict:
    # ``h_x`` is the empirical entropy of the secret's prior; when None the secret is uniform by
    # construction (a fair challenge bit, or a keyed permutation) and log2 K is exact. Passing the
    # empirical entropy is what keeps the Fano value a valid *lower* bound under a skewed prior.
    bits = None if acc is None else fano_bits(acc, k, h_x)
    cons = fano_bits(acc_lo, k, h_x) if acc_lo is not None else None
    has_ci = lo is not None and hi is not None
    # "leaks" means the whole interval lies above the null. An interval entirely *below or spanning*
    # it is "not detected": the attacker oriented its statistic on the training half, so a held-out
    # value at or under the null is an absence of evidence, reported with the minimum detectable
    # effect rather than as a closure. A result with no interval is neither, and says so.
    if not has_ci:
        status = "no interval"
    elif lo > null:
        status = "leaks"
    else:
        status = "not detected"
    return {
        "channel": channel,
        "design": design,
        "adversary": adversary,
        "experiment": experiment,
        "unit": unit,
        "k": k,
        "metric": metric,
        "value": value,
        "lo": lo,
        "hi": hi,
        "null": null,
        "accuracy": acc,
        "accuracy_lo": acc_lo,
        "prior": prior,
        "entropy_bits": math.log2(k) if h_x is None else h_x,
        "bits": bits,
        "bits_conservative": cons,
        "bits_max": math.log2(k) if h_x is None else h_x,
        "status": status,
        **dict(
            zip(
                ("mde", "mde_basis"),
                mde(lo, hi, n, metric) if has_ci and status == "not detected" else (None, ""),
                strict=True,
            )
        ),
        "n": n,
        "conversion": how,
    }


def from_e5(out: list[dict]) -> None:
    d = load("e5_game.json")
    if not d:
        return
    names = {
        "P shipped": ("snapshot + post-tau writes (the game)", "Mnemos-P"),
        "P no S1 (growing regions)": ("game, S1 removed (growing regions)", "Mnemos-P ablation"),
        "P no S2 (unsealed slots)": ("game, S2 removed (unsealed slots)", "Mnemos-P ablation"),
        "P no S3 (per-slot writes)": ("game, per-slot writes, random slot", "Mnemos-P ablation"),
        "P first-free slot (S3 kept)": (
            "game, first-free slot, whole-region writes",
            "Mnemos-P ablation",
        ),
        "P no S3, first-free slot": ("game, per-slot writes, first-free slot", "Mnemos-P ablation"),
        "P identity permutation": ("game, no keyed permutation", "Mnemos-P ablation"),
        "P inadmissible, silent refusal": ("refusal, silent", "Mnemos-P ablation"),
        "P inadmissible, padded refusal": ("refusal, padded", "Mnemos-P"),
        "M shipped": ("snapshot + post-tau writes (the game)", "Mnemos-M"),
        "M content placement": ("game, content placement", "Mnemos-M ablation"),
    }
    for r in d["rows"]:
        if r["arm"] not in names:
            continue
        channel, design = names[r["arm"]]
        # The stronger of the two attackers, judged on the held-out half. The Fano value is computed
        # from the held-out balanced *accuracy* the attacker commits to, never the AUC: AUC is a
        # ranking measure, not a decoding accuracy, so there is no valid AUC->Fano conversion. A row
        # without a recorded accuracy carries its AUC but no bit.
        best = max((r["learned"], r["hand_built"]), key=lambda a: abs(a["auc"] - 0.5))
        if "accuracy" in best:
            acc, acc_lo = best["accuracy"], best["accuracy_lo"]
            how = "Fano (K=2) on held-out balanced accuracy, threshold fixed on the training half"
        else:
            acc = acc_lo = None
            how = "AUC only; no accuracy recorded, so no MI bound (AUC is not a decoding accuracy)"
        out.append(
            row(
                channel,
                design,
                "W",
                "E5",
                "one game instance (cohort H history bit)",
                2,
                acc,
                acc_lo,
                "AUC",
                best["auc"],
                best["lo"],
                best["hi"],
                0.5,
                how,
                n=r.get("n_heldout"),
                prior="uniform (fair challenge bit)",
            )
        )


def from_e4(out: list[dict]) -> None:
    d = load("e4_drift.json")
    if not d:
        return
    k = d["fine"]["n_topics"]
    ch = d["fine"]["chance"]
    hx = d["fine"].get("label_entropy_bits")
    prior = (
        "empirical topic entropy"
        if hx is not None
        else "uniform (assumed; corpus skew not recorded)"
    )
    # Cohort H's topic prior on this corpus is skewed, so beating 1/K in accuracy can be the prior
    # alone. A leak is judged on balanced accuracy (mean per-topic recall), which the prior cannot
    # raise; failing that, against an adversary that always guesses the most frequent topic.
    shift = ch - d["fine"].get("majority_rate", ch)
    for c in d["fine"]["curve"]:
        if c["gap_eras"] not in (0, max(x["gap_eras"] for x in d["fine"]["curve"])):
            continue
        if "adv_H_bal" in c:
            metric, v, lo, hi = "balanced advantage", c["adv_H_bal"], c["lo_bal"], c["hi_bal"]
        elif shift:
            metric, v, lo, hi = (
                "accuracy over majority",
                *(c[x] + shift for x in ("adv_H", "lo", "hi")),
            )
        else:
            metric, v, lo, hi = "advantage", c["adv_H"], c["lo"], c["hi"]
        out.append(
            row(
                f"structure transfer, {c['adversary']} adversary",
                "growing layout",
                "W",
                "E4",
                "one cohort-H record's topic",
                k,
                ch + c["adv_H"],
                ch + c["lo"],
                metric,
                v,
                lo,
                hi,
                0.0,
                f"Fano (K={k}) on topic accuracy = chance + advantage",
                h_x=hx,
                prior=prior,
            )
        )


def entropy_bits(labels) -> float:
    counts = np.bincount(np.asarray(labels))
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def from_e16(out: list[dict]) -> None:
    """pgvector under gocryptfs: cohort H is a random draw from a corpus whose topics are balanced
    (D2, eight BEIR domains), so the advantage over 1/K is not a prior effect."""
    d = load("e16_real_layouts.json")
    labels_file = R.parent / "e8" / "data" / "labels.npy"
    if not d or not labels_file.exists():
        return
    hx = entropy_bits(np.load(labels_file))
    for s in d["stores"]:
        k = round(1 / s["chance"])
        out.append(
            row(
                f"structure transfer, deployed store: {s['store']}",
                s["store"],
                "W",
                "E16",
                "one cohort-H record's topic",
                k,
                s["chance"] + s["adv_hist"],
                s["chance"] + s["adv_hist_lo"],
                "advantage",
                s["adv_hist"],
                s["adv_hist_lo"],
                s.get("adv_hist_hi"),
                0.0,
                f"Fano (K={k}) on topic accuracy = chance + advantage",
                h_x=hx,
                prior="empirical topic entropy (balanced corpus)",
            )
        )


def from_agent_streams(out: list[dict]) -> None:
    """The agent-written streams on the growing layout, at the largest budget measured. Their
    topic priors are skewed, so a leak is judged on balanced accuracy (mean per-topic recall),
    which the prior alone cannot raise; the Fano bound still uses raw accuracy and the empirical
    entropy, so it stays a valid lower bound."""
    from eval.exp2 import e1_longitudinal as e1m
    from eval.exp2 import e2_agent_attack as e2m

    for name, exp, key, extra in (
        ("e1_longitudinal.json", "E1", "design", {"adversary": "W"}),
        ("e2_agent_attack.json", "E2", "layout", {}),
    ):
        d = load(name)
        if not d:
            continue
        rows = [
            r
            for r in d["rows"]
            if r.get(key) == "growing"
            and "adv_H_bal" in r
            and all(r.get(a) == b for a, b in extra.items())
        ]
        if not rows:
            continue
        budget = "budget" if exp == "E1" else "budget_turns_per_topic"
        top = max(rows, key=lambda r: (r[budget], r.get("w_real", 1)))
        if exp == "E1":
            _t, topics, _n, month_idx, _s = e1m.load_stream()
            n_months = int(month_idx.max()) + 1
            hx = entropy_bits(topics[month_idx < round(n_months * 0.5)])
        else:
            hx = entropy_bits(e2m.load_memories(None)[1])
        k = d["n_topics"]
        ch = d["chance"]
        burst = f", bursts of {top['w_real']}" if "w_real" in top else ""
        out.append(
            row(
                f"structure transfer, agent-written memories ({exp}{burst})",
                "growing layout",
                "W",
                exp,
                "one cohort-H record's topic",
                k,
                ch + top["adv_H"],
                ch + top["lo"],
                "balanced advantage",
                top["adv_H_bal"],
                top["lo_bal"],
                top["hi_bal"],
                0.0,
                f"Fano (K={k}) on topic accuracy; leak judged on balanced accuracy",
                h_x=hx,
                prior="empirical topic entropy (skewed)",
            )
        )


def from_e6(out: list[dict]) -> None:
    d = load("e6_stores.json")
    if not d:
        return
    hx = d.get("label_entropy_bits")
    prior = (
        "empirical topic entropy"
        if hx is not None
        else "uniform (assumed; corpus skew not recorded)"
    )
    for s in d["stores"]:
        k = round(1 / s["chance"])
        out.append(
            row(
                f"structure transfer, deployed store: {s['store']}",
                s["store"],
                "W",
                "E6",
                "one cohort-H record's topic",
                k,
                s["chance"] + s["adv_hist"],
                s["chance"] + s["adv_hist_lo"],
                "advantage",
                s["adv_hist"],
                s["adv_hist_lo"],
                s.get("adv_hist_hi"),
                0.0,
                f"Fano (K={k}) on topic accuracy = chance + advantage",
                h_x=hx,
                prior=prior,
            )
        )


def from_e7(out: list[dict]) -> None:
    d = load("e7_permutation.json")
    if not d:
        return
    b = d["buckets"]
    for m in d["modes"]:
        out.append(
            row(
                f"permutation recovery ({m['mode']})",
                "Mnemos-P",
                "RW",
                "E7",
                "one region's logical partition",
                b,
                m["exact_mean"],
                m["exact_lo"],
                "accuracy",
                m["exact_mean"],
                m["exact_lo"],
                m["exact_hi"],
                m["null"],
                f"Fano (K={b}) on the fraction of regions whose partition is recovered exactly",
                prior="uniform (keyed permutation over B regions)",
            )
        )


def from_e8(out: list[dict]) -> None:
    d = load("e8_multisnapshot.json")
    if not d:
        return
    b = d["buckets"]
    for g in d["granularity_sweep"]:
        out.append(
            row(
                f"multi-version diff, g={g['granularity_flushes']}",
                "Mnemos-P",
                "W",
                "E8",
                "one pre-tau record's region",
                b,
                g["precision"],
                g["lo"],
                "precision",
                g["precision"],
                g["lo"],
                g["hi"],
                g["null_precision"],
                f"Fano (K={b}) on the probability of naming the region",
                h_x=d.get("region_entropy_bits"),
                prior="empirical region entropy"
                if d.get("region_entropy_bits") is not None
                else "uniform (assumed; region skew not recorded)",
            )
        )
    for c in d["cross_session"]:
        tag = "on" if c["rewrite_at_lock"] else "off"
        out.append(
            row(
                f"multi-version diff, across a session, rewrite_at_lock {tag}",
                "Mnemos-P",
                "W",
                "E8",
                "one pre-tau record's region",
                b,
                c["precision"],
                c["lo"],
                "precision",
                c["precision"],
                c["lo"],
                c["hi"],
                c["null_precision"],
                f"Fano (K={b}) on the probability of naming the region",
                h_x=d.get("region_entropy_bits"),
                prior="empirical region entropy"
                if d.get("region_entropy_bits") is not None
                else "uniform (assumed; region skew not recorded)",
            )
        )


def from_e9(out: list[dict]) -> None:
    d = load("e9_refusal.json")
    if not d:
        return
    for c in d["channel"]:
        # The gap in rewrite rate between full and open regions is TPR - FPR of the test "was the
        # region rewritten", i.e. Youden's J for one induced write, so balanced accuracy is
        # (1 + J)/2 exactly rather than via the AUC bound.
        acc = 0.5 * (1 + c["gap"])
        acc_lo = 0.5 * (1 + c["gap_lo"])
        out.append(
            row(
                f"refusal, {c['mode']}",
                "Mnemos-P",
                "W",
                "E9",
                "one region's fullness, per induced write",
                2,
                acc,
                acc_lo,
                "AUC",
                c["auc_full_vs_open"],
                c["lo"],
                c["hi"],
                0.5,
                "Fano (K=2) on (1 + J)/2, J = rewrite-rate gap between full and open regions",
                prior="uniform (full vs open region, balanced)",
            )
        )


def from_query_equality(out: list[dict]) -> None:
    d = load("e15_query_equality.json")
    if d:
        for m in d["modes"]:
            # Youden's J on held-out pairs is TPR - FPR of the adversary's fixed test, so balanced
            # accuracy on one pair is (1 + J)/2 exactly, as for the refusal channel.
            j = m["threshold_j"]
            out.append(
                row(
                    f"query equality ({m['mode']}), overlap threshold",
                    "Mnemos-P",
                    "RW",
                    "E15",
                    "one pair of searches",
                    2,
                    0.5 * (1 + j["mean"]),
                    0.5 * (1 + j["lo"]),
                    "Youden J",
                    j["mean"],
                    j["lo"],
                    j["hi"],
                    0.0,
                    "Fano (K=2) on (1 + J)/2; tau fixed on held-out queries",
                    prior="uniform (equal vs distinct query pair, balanced)",
                )
            )
        return
    d = load("e10_query_equality.json")
    if not d:
        return
    for m in d["modes"]:
        j = m["overlap_threshold_best"]["youden_j"]
        # No interval exists for this experiment (12 queries x 20 repeats, earlier round). The row
        # is kept so the channel is not silently missing, and its status says why it is not
        # classified.
        out.append(
            row(
                f"query equality ({m['mode']}), overlap threshold",
                "Mnemos-P",
                "RW",
                "query equality (earlier harness)",
                "one pair of searches",
                2,
                0.5 * (1 + j),
                None,
                "Youden J",
                j,
                None,
                None,
                0.0,
                "Fano (K=2) on (1 + J)/2; NO INTERVAL",
            )
        )


def from_e3(out: list[dict]) -> None:
    d = load("e3_dedup.json")
    if not d:
        return
    for r in d.get("rows", []):
        lo, hi = r.get("lo"), r.get("hi")
        design = {"p": "Mnemos-P", "m": "Mnemos-M"}.get(r.get("engine"), r.get("engine"))
        # No AUC->accuracy conversion: use the recorded held-out accuracy, or none (AUC-only row).
        acc = r.get("accuracy")
        acc_lo = r.get("accuracy_lo")
        out.append(
            row(
                f"dedup membership ({r.get('dedup', '?')})",
                design,
                "W",
                "E3",
                "one fact's membership in cohort H",
                2,
                acc,
                acc_lo,
                "AUC",
                r["auc"],
                lo,
                hi,
                0.5,
                "Fano (K=2) on held-out accuracy; AUC-only rows carry no MI bound",
                prior="uniform (member vs non-member, balanced)",
            )
        )


def main() -> None:
    out: list[dict] = []
    for f in (
        from_e4,
        from_agent_streams,
        from_e5,
        from_e6,
        from_e16,
        from_e7,
        from_e8,
        from_e9,
        from_query_equality,
        from_e3,
    ):
        f(out)
    OUT.write_text(json.dumps({"rows": out}, indent=1), encoding="utf-8")

    print(
        f"{'channel':<58} {'design':<18} {'cls':<3} {'status':<11} "
        f"{'bits':>6} {'cons':>6} {'max':>5}  mde"
    )
    for r in out:
        cons = "--" if r["bits_conservative"] is None else f"{r['bits_conservative']:.3f}"
        if r["mde"] is None:
            m = "undetermined" if r["status"] == "not detected" else ""
        else:
            m = f"{r['mde']:.3f} ({r['metric']})"
        bits = "  --  " if r["bits"] is None else f"{r['bits']:>6.3f}"
        print(
            f"{r['channel'][:58]:<58} {r['design'][:18]:<18} {r['adversary']:<3} {r['status']:<12} "
            f"{bits} {cons:>6} {r['bits_max']:>5.2f}  {m}"
        )
    print(f"\nwrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
