"""E28: what the induction attack costs the adversary.

A descriptive analysis of results already on disk, so it registers no prediction: it reads E2's
agent run (wall time per induced turn with each local model, and how often a turn writes) and E2's
attack sweep (advantage per budget), and reports, per layout and burst size, the smallest budget
whose 95% interval on the cohort-H advantage excludes zero, expressed as adversary messages (one
induced turn is one supplied document, ticket or email) and as agent time.

    cd python && python -m eval.exp2.e28_attack_cost
"""

from __future__ import annotations

import json

from eval.lab import corpus as C

RES = C.repo_root() / "python" / "eval" / "results"
OUT = RES / "e28_attack_cost.json"


def main() -> None:
    agent = json.loads((RES / "e2_agent.json").read_text(encoding="utf-8"))
    attack = json.loads((RES / "e2_agent_attack.json").read_text(encoding="utf-8"))
    n_topics = attack["n_topics"]

    models = [
        {
            "model": r["model"],
            "seconds_per_turn": r["seconds"] / r["turns"],
            "write_probability": r["write_probability"],
            "writes_per_turn": r["mean_writes_per_turn"],
        }
        for r in agent["induction"]
    ]
    slowest = max(m["seconds_per_turn"] for m in models)

    cells: dict[tuple[str, int], list[dict]] = {}
    for r in attack["rows"]:
        cells.setdefault((r["layout"], r["w_real"]), []).append(r)
    minimum = []
    for (layout, w_real), rows in sorted(cells.items()):
        rows.sort(key=lambda r: r["budget_turns_per_topic"])
        hit = next((r for r in rows if r["lo"] > 0), None)
        b = hit["budget_turns_per_topic"] if hit else None
        minimum.append(
            {
                "layout": layout,
                "w_real": w_real,
                "min_budget_turns_per_topic": b,
                "budgets_tested": [r["budget_turns_per_topic"] for r in rows],
                "adversary_messages": b * n_topics if b else None,
                "agent_minutes_slowest_model": b * n_topics * slowest / 60 if b else None,
                "adv_H_at_min": hit["adv_H"] if hit else None,
                "lo_at_min": hit["lo"] if hit else None,
            }
        )
        msg = f"{b} turns/topic = {b * n_topics} messages" if b else "never clears zero"
        print(f"{layout:>8} w_real={w_real}: {msg}")

    out = {
        "experiment": "E28",
        "kind": "descriptive (computed from E2 results; no registered prediction)",
        "n_topics": n_topics,
        "models": models,
        "minimum_detectable_budget": minimum,
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for m in models:
        s, p = m["seconds_per_turn"], m["write_probability"]
        print(f"{m['model']}: {s:.2f} s/turn, p(write) {p}")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
