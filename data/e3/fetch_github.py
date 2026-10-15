"""Fetch the GitHub-issues corpus for E3: kubernetes/kubernetes issues labelled with one sig/*.

The ``sig/*`` labels are a topical taxonomy maintained by the project (node, network, storage,
api-machinery, ...), and every issue carries a creation timestamp, so the corpus supports both a
chronological cohort split and a label-derived topic ground truth.

The REST issues listing refuses pagination past page 100, so this uses the search API in yearly
windows per label (each window returns at most 1,000 results). Pull requests are excluded, and
an issue is kept only if it carries exactly one ``sig/*`` label. Output:
``data/e3/github/issues.jsonl`` with ``{id, created_at, topic, title, body}``.

    python data/e3/fetch_github.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = "kubernetes/kubernetes"
OUT = Path(__file__).resolve().parent / "github"
SIGS = [
    "node", "network", "storage", "api-machinery", "scheduling", "auth",
    "apps", "cli", "cloud-provider", "instrumentation", "testing", "windows",
]
YEARS = range(2014, 2027)
PER_SIG_CAP = 1500
PAUSE_S = 2.2  # search API: 30 requests per minute when authenticated


def search(q: str, page: int) -> dict:
    cmd = ["gh", "api", "-X", "GET", "search/issues", "-f", f"q={q}", "-f", "per_page=100",
           "-f", f"page={page}", "-f", "sort=created", "-f", "order=asc"]
    for attempt in range(5):
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            return json.loads(r.stdout)
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"search failed: {q} page {page}: {r.stderr.decode(errors='replace')}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: dict[int, dict] = {}
    for sig in SIGS:
        kept_sig = 0
        for year in YEARS:
            if kept_sig >= PER_SIG_CAP:
                break
            q = f"repo:{REPO} is:issue label:sig/{sig} created:{year}-01-01..{year}-12-31"
            for page in range(1, 11):
                data = search(q, page)
                time.sleep(PAUSE_S)
                items = data.get("items", [])
                for it in items:
                    sigs = [lab["name"] for lab in it.get("labels", [])
                            if lab["name"].startswith("sig/")]
                    if len(sigs) != 1 or it["number"] in rows:
                        continue
                    rows[it["number"]] = {
                        "id": it["number"],
                        "created_at": it["created_at"],
                        "topic": sigs[0].removeprefix("sig/"),
                        "title": it["title"],
                        "body": (it.get("body") or "").strip()[:6000],
                    }
                    kept_sig += 1
                if len(items) < 100 or kept_sig >= PER_SIG_CAP:
                    break
            print(f"sig/{sig} {year}: kept {kept_sig}", flush=True)
    ordered = sorted(rows.values(), key=lambda r: r["created_at"])
    out = OUT / "issues.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in ordered:
            fh.write(json.dumps(r) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps({
        "repo": REPO, "sigs": SIGS, "years": [YEARS.start, YEARS.stop - 1],
        "per_sig_cap": PER_SIG_CAP, "kept": len(ordered), "sha256": digest,
        "filter": "issues only (search is:issue), exactly one sig/* label",
        "fetched": time.strftime("%Y-%m-%d"),
    }, indent=2), encoding="utf-8")
    print("done", len(ordered), digest)


if __name__ == "__main__":
    main()
