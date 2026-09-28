"""Assemble the LLM-authored part files into the final Mnemos datasets.

The prose in data/_parts/ was authored by hand (LLM-written). This script only adds
structural fields -- ids, global sequence numbers, synthetic monotonic timestamps,
the ``meta`` JSON string, ``words`` counts -- and writes a manifest. It generates
no content.

Usage:  python data/assemble.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARTS = ROOT / "_parts"

TAXONOMY = [
    "sql", "networking", "filesystem", "concurrency", "crypto",
    "containers", "web", "debugging", "algorithms", "cloud",
]

CATEGORIES = {"bugfix", "feature", "refactor", "performance", "hardening"}
DIFFICULTIES = {"easy", "medium", "hard"}

NOTE_BASE_MS = 1_700_000_000_000
NOTE_STEP_MS = 30_000


def _read_lines(path: Path) -> list[dict]:
    recs: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=True, separators=(",", ":")) + "\n")


def build_notes() -> list[dict]:
    out: list[dict] = []
    seq = 0
    for topic in TAXONOMY:
        recs = _read_lines(PARTS / "notes" / f"{topic}.jsonl")
        assert len(recs) == 200, f"notes/{topic}: expected 200, got {len(recs)}"
        for i, r in enumerate(recs):
            meta = {k: r[k] for k in ("category", "difficulty", "language", "component", "tags")}
            out.append({
                "note_id": f"d3-{topic}-{i:05d}",
                "topic": topic,
                "category": r["category"],
                "difficulty": r["difficulty"],
                "language": r["language"],
                "component": r["component"],
                "source_task": f"task-{topic}-{seq % 60:03d}",
                "tags": r["tags"],
                "written_at_ms": NOTE_BASE_MS + seq * NOTE_STEP_MS,
                "seq": seq,
                "text": r["text"],
                "meta": json.dumps(meta, separators=(",", ":")),
            })
            seq += 1
    return out


def build_tasks() -> list[dict]:
    out: list[dict] = []
    for topic in TAXONOMY:
        recs = _read_lines(PARTS / "tasks" / f"{topic}.jsonl")
        assert len(recs) == 60, f"tasks/{topic}: expected 60, got {len(recs)}"
        for i, r in enumerate(recs):
            out.append({
                "task_id": f"task-{topic}-{i:03d}",
                "topic": topic,
                "difficulty": r["difficulty"],
                "tags": r["tags"],
                "task": r["task"],
            })
    return out


def build_clusters() -> list[dict]:
    out: list[dict] = []
    for cluster in TAXONOMY:
        recs = _read_lines(PARTS / "clusters" / f"{cluster}.jsonl")
        assert len(recs) == 60, f"clusters/{cluster}: expected 60, got {len(recs)}"
        for i, r in enumerate(recs):
            out.append({
                "doc_id": f"clu-{cluster}-{i:03d}",
                "cluster": cluster,
                "title": r["title"],
                "text": r["text"],
                "words": len(r["text"].split()),
            })
    return out


def params_hash() -> str:
    return hashlib.sha256((ROOT.parent / "params.toml").read_bytes()).hexdigest()


def validate(notes: list[dict], tasks: list[dict], clusters: list[dict]) -> None:
    assert len(notes) == 2000, len(notes)
    assert len(tasks) == 600, len(tasks)
    assert len(clusters) == 600, len(clusters)

    assert len({n["text"] for n in notes}) == 2000, "duplicate note text"
    assert len({t["task"] for t in tasks}) == 600, "duplicate task text"
    assert len({c["text"] for c in clusters}) == 600, "duplicate cluster text"

    for n in notes:
        assert n["topic"] in TAXONOMY
        assert n["category"] in CATEGORIES
        assert n["difficulty"] in DIFFICULTIES
        assert n["text"].strip()
        meta = json.loads(n["meta"])
        assert meta["category"] == n["category"]
    for t in tasks:
        assert t["topic"] in TAXONOMY
        assert t["difficulty"] in DIFFICULTIES
    for c in clusters:
        assert c["cluster"] in TAXONOMY
        assert c["words"] == len(c["text"].split())

    times = [n["written_at_ms"] for n in notes]
    assert times == sorted(times) and len(set(times)) == len(times), "timestamps not monotonic"
    print("  ok uniqueness, enums, meta consistency, monotonic timestamps")


def write_manifest(notes, tasks, clusters) -> None:
    manifest = {
        "generator": "LLM-authored content in data/_parts/ + data/assemble.py (structure only)",
        "version": "2.0.0",
        "notes": "deterministic; no RNG used",
        "params_hash": params_hash(),
        "params_file": "params.toml",
        "files": [
            {"path": "d3-swebench/notes.jsonl", "count": len(notes),
             "schema": ["note_id", "topic", "category", "difficulty", "language",
                        "component", "source_task", "tags", "written_at_ms", "seq",
                        "text", "meta"]},
            {"path": "d3-swebench/tasks.jsonl", "count": len(tasks),
             "schema": ["task_id", "topic", "difficulty", "tags", "task"]},
            {"path": "retrieval-clusters.jsonl", "count": len(clusters),
             "schema": ["doc_id", "cluster", "title", "text", "words"]},
        ],
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    notes = build_notes()
    tasks = build_tasks()
    clusters = build_clusters()
    validate(notes, tasks, clusters)
    _write_jsonl(ROOT / "d3-swebench" / "notes.jsonl", notes)
    _write_jsonl(ROOT / "d3-swebench" / "tasks.jsonl", tasks)
    _write_jsonl(ROOT / "retrieval-clusters.jsonl", clusters)
    write_manifest(notes, tasks, clusters)
    print(f"  ok notes={len(notes)} tasks={len(tasks)} clusters={len(clusters)}")


if __name__ == "__main__":
    main()
