"""Assemble the E3 realistic, timestamped corpora.

Four corpora, each a list of ``{id, ts, topic, text}`` written to ``data/e3/<name>.jsonl`` in
timestamp order, so an experiment can split at tau chronologically (old memories vs new) as well
as by a shuffle. Topic labels come from the source, never from the text.

``enron``      Enron email by sender folder-topic and send date. Inbox-triage agent memory.
``github``     kubernetes issues by their single sig/* label and creation date. Ticket triage.
``locomo``     Turns from the LoCoMo long-conversation benchmark, session-dated, topic = a coarse
               cluster over the turn text. Conversational memory. (LLM-written memories are
               produced separately by the agent loop; this file keeps the raw turns.)
``newsgroups`` 20 Newsgroups, the fine overlapping-topic taxonomy the review asks for. No real
               timestamps, so used only for the shuffled split and the open-world taxonomy.

    python data/e3/build_corpora.py
"""

from __future__ import annotations

import email
import hashlib
import json
import re
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "_raw"

# Enron: map top-level Maildir folder names to a coarse role/topic. Folders vary per user, so a
# small robust set of buckets keyed on folder-name substrings.
ENRON_TOPICS = [
    ("legal", ("legal", "lawsuit", "contracts")),
    ("hr", ("hr", "personal", "resume")),
    ("logistics", ("logistics", "schedul", "meetings", "calendar")),
    ("trading", ("trading", "trade", "desk", "positions")),
    ("regulatory", ("ferc", "regulatory", "california", "puc")),
    ("deals", ("deal", "transaction", "project")),
]


def _enron_topic(folder: str) -> str | None:
    f = folder.lower()
    for name, keys in ENRON_TOPICS:
        if any(k in f for k in keys):
            return name
    return None


def build_enron(limit_per_topic: int = 1200) -> None:
    tgz = RAW / "enron_mail_20150507.tar.gz"
    if not tgz.exists():
        print("enron: raw tarball missing, skipping")
        return
    rows: list[dict] = []
    counts: dict[str, int] = {}
    with tarfile.open(tgz, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            # maildir/<user>/<folder>/<n>
            if len(parts) < 4:
                continue
            topic = _enron_topic(parts[2])
            if topic is None or counts.get(topic, 0) >= limit_per_topic:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            try:
                msg = email.message_from_bytes(f.read())
            except Exception:  # noqa: BLE001
                continue
            date = msg.get("Date", "")
            body = msg.get_payload()
            if not isinstance(body, str):
                continue
            body = re.sub(r"\s+", " ", body).strip()
            if len(body) < 120:
                continue
            ts = _parse_date(date)
            if ts is None:
                continue
            subj = msg.get("Subject", "").strip()
            rows.append({
                "id": hashlib.sha1(member.name.encode()).hexdigest()[:16],
                "ts": ts, "topic": topic,
                "text": (f"Subject: {subj}\n{body}")[:4000],
            })
            counts[topic] = counts.get(topic, 0) + 1
    rows.sort(key=lambda r: r["ts"])
    _write("enron", rows)


def _parse_date(s: str) -> str | None:
    try:
        dt = email.utils.parsedate_to_datetime(s)
        return dt.astimezone().isoformat()
    except Exception:  # noqa: BLE001
        return None


def build_github() -> None:
    src = HERE / "github" / "issues.jsonl"
    if not src.exists():
        print("github: fetch not present, skipping")
        return
    rows = []
    for line in src.open(encoding="utf-8"):
        r = json.loads(line)
        text = (r["title"] + "\n" + r.get("body", "")).strip()
        if len(text) < 80:
            continue
        rows.append({"id": f"gh-{r['id']}", "ts": r["created_at"], "topic": r["topic"], "text": text[:4000]})
    rows.sort(key=lambda r: r["ts"])
    _write("github", rows)


def build_locomo() -> None:
    src = RAW / "locomo10.json"
    if not src.exists():
        print("locomo: raw json missing, skipping")
        return
    convos = json.loads(src.read_text(encoding="utf-8"))
    rows = []
    for ci, conv in enumerate(convos):
        c = conv["conversation"]
        for key in sorted(k for k in c if re.fullmatch(r"session_\d+", k)):
            dt = c.get(f"{key}_date_time", "")
            for turn in c[key]:
                text = turn.get("text", "").strip()
                if len(text) < 40:
                    continue
                rows.append({
                    "id": f"lc-{ci}-{turn.get('dia_id', len(rows))}",
                    "ts_raw": dt, "session": key, "conv": ci,
                    "speaker": turn.get("speaker", ""), "text": text[:2000],
                })
    # LoCoMo timestamps are natural-language ("1:56 pm on 8 May, 2023"); keep raw plus a monotone
    # ordinal so the chronological split is well-defined even where parsing is loose.
    for i, r in enumerate(rows):
        r["ts"] = i
    _write("locomo_turns", rows, topic_field=None)


def build_newsgroups() -> None:
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError:
        print("newsgroups: sklearn missing, skipping")
        return
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    rows = []
    for i, (text, label) in enumerate(zip(data.data, data.target, strict=True)):
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 120:
            continue
        rows.append({"id": f"ng-{i}", "ts": i, "topic": data.target_names[label], "text": text[:4000]})
    _write("newsgroups", rows)


def _write(name: str, rows: list[dict], topic_field: str | None = "topic") -> None:
    out = HERE / f"{name}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    topics = {}
    if topic_field:
        for r in rows:
            topics[r[topic_field]] = topics.get(r[topic_field], 0) + 1
    (HERE / f"{name}.manifest.json").write_text(json.dumps({
        "corpus": name, "count": len(rows), "sha256": digest,
        "topics": dict(sorted(topics.items(), key=lambda kv: -kv[1])),
    }, indent=2), encoding="utf-8")
    print(f"{name}: {len(rows)} rows, {len(topics)} topics -> {out.name}")


if __name__ == "__main__":
    build_github()
    build_locomo()
    build_newsgroups()
    build_enron()
