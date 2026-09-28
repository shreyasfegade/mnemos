"""E14: gather the review-round results, and check that every reported number carries an interval.

The review's last experiment item is not an experiment: it asks that every table carry confidence
intervals, and that a corpus listed in the setup either be used or removed. This collects the
results of the 2026-09-16 round into one file and audits them for exactly that.

An entry is flagged when it reports a point estimate with no interval beside it. The audit is
deliberately crude -- it looks for the interval keys the harness emits -- because its job is to
catch a table that was written without one, not to judge the statistics.

    cd python && python -m eval.exp2.collect
"""

from __future__ import annotations

import json

from eval.lab import corpus as C

R = C.repo_root() / "python" / "eval" / "results"
OUT = R / "ALL2.json"

# The review's numbering for this round, and the file each experiment writes.
EXPECTED = {
    "E1": "e1_longitudinal.json",
    "E2": "e2_agent_attack.json",
    "E3": "e3_dedup.json",
    "E4": "e4_drift.json",
    "E5": "e5_game.json",
    "E6": "e6_stores.json",
    "E7": "e7_permutation.json",
    "E8": "e8_multisnapshot.json",
    "E9": "e9_refusal.json",
    "E10": "e10_performance.json",
    "E11": "e11_retrieval.json",
    "E12": "e12_oblivious.json",
    "E13": "e13_artifacts.json",
    # Not in the review's numbering: the query-equality re-measurement it asked for (the numbers
    # the paper quoted came from an earlier harness with no interval).
    "E15": "e15_query_equality.json",
}

INTERVAL_KEYS = {
    "lo",
    "hi",
    "auc_lo",
    "auc_hi",
    "ndcg_lo",
    "ndcg_hi",
    "recall_lo",
    "recall_hi",
    "exact_lo",
    "exact_hi",
    "neighbourhood_lo",
    "neighbourhood_hi",
    "p95",
    "p99",
    "localisation_lo",
    "localisation_hi",
    "adv_hist_lo",
    "adv_hist_hi",
    "gap_lo",
    "gap_hi",
    "exact_recovery_lo",
    "exact_recovery_hi",
}

# Subtrees that are raw per-run records rather than reported estimates. A single run has no
# interval by construction; flagging it would train the reader to ignore this audit.
RAW_SUBTREES = {"raw", "repeats", "per_seed"}
POINT_KEYS = {
    "adv_H",
    "adv_hist",
    "auc",
    "advantage",
    "ndcg",
    "recall",
    "precision",
    "exact_recovery",
    "recovery_mean",
    "exposure",
}


def walk(node, path="") -> list[tuple[str, set, set]]:
    """Every dict that reports a point estimate, with the interval keys it carries."""
    out = []
    if isinstance(node, dict):
        points = POINT_KEYS & set(node)
        if points:
            out.append((path, points, INTERVAL_KEYS & set(node)))
        for k, v in node.items():
            if k in RAW_SUBTREES:
                continue
            out.extend(walk(v, f"{path}/{k}"))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(walk(v, f"{path}[{i}]"))
    return out


def main() -> None:
    summary: dict = {"round": "2026-09-16", "experiments": {}, "missing": [], "audit": []}
    for eid, fname in EXPECTED.items():
        f = R / fname
        if not f.exists():
            summary["missing"].append({"experiment": eid, "file": fname})
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            summary["experiments"][eid] = {"error": str(e)}
            continue
        summary["experiments"][eid] = data
        bare = [p for p, pts, ivs in walk(data, eid) if pts and not ivs]
        if bare:
            summary["audit"].append(
                {
                    "experiment": eid,
                    "point_estimates_without_intervals": bare[:12],
                    "count": len(bare),
                }
            )

    # Enron: the setup listed it and no experiment ever used it. Record which corpora are in play.
    e3 = C.repo_root() / "data" / "e3"
    corpora = sorted(p.stem for p in e3.glob("*.jsonl"))
    summary["corpora_present"] = corpora
    summary["enron_used"] = any("enron" in str(v) for v in summary["experiments"].values())

    OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    have = sorted(summary["experiments"])
    print(f"collected {len(have)} of {len(EXPECTED)}: {', '.join(have)}")
    if summary["missing"]:
        print("missing:", ", ".join(m["experiment"] for m in summary["missing"]))
    if summary["audit"]:
        print("\npoint estimates with no interval beside them:")
        for a in summary["audit"]:
            eg = a["point_estimates_without_intervals"][:3]
            print(f"  {a['experiment']}: {a['count']} sites, e.g. {eg}")
    else:
        print("\nevery point estimate carries an interval")
    print(f"\ncorpora present: {', '.join(corpora)}")
    print(f"enron referenced by any result: {summary['enron_used']}")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
