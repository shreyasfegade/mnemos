"""Gather every experiment result JSON into one summary the paper and tracker reference.

cd python && python -m eval.collect_results
"""

from __future__ import annotations

import json

from eval.lab import corpus as C

R = C.repo_root() / "python" / "eval" / "results"


def main() -> None:
    summary = {}
    for f in sorted(R.glob("*.json")):
        try:
            summary[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            summary[f.stem] = {"error": str(e)}
    out = R / "ALL.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("collected:", ", ".join(sorted(summary)))
    print(f"wrote {out.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
