"""Build **D2**, the public real-prose corpus, from BEIR.

Why this exists
---------------
Every security number in the paper was, until now, measured on D3 -- which
``eval/lab/corpus.py`` itself flags as template-assembled and therefore optimistic in
embedding space. A structure-transfer attack is a claim about how *semantics* land in
partitions, so measuring it only on generated text measures the generator.

Why BEIR and not MS MARCO
-------------------------
The evaluation section previously promised "MS MARCO / NQ". Both are the wrong instrument
here, for a reason worth stating rather than quietly working around:

    A4 recovers a **topic label** for each entry. MS MARCO passages carry no topic labels.
    Producing them would mean clustering the passage embeddings and then scoring the attack's
    recovered partition-to-topic map against those same clusters -- measuring the embedding
    against itself. Any number obtained that way is uninterpretable, and flattering.

BEIR is a *collection* of independently built retrieval corpora. Using one sub-corpus per
topic gives a label that is **external to the embedding**: a document is in ``trec-covid``
because a COVID IR benchmark was built around it, not because a clustering algorithm said so.
That is exactly the property the metric needs, and it is real prose written by real people.

The eight domains are deliberately spread across register as well as subject -- scientific
claims, clinical nutrition, debate argumentation, CS abstracts, retail-investor finance,
biomedical COVID literature, community questions, and controversial-topic argument mining --
so that "topic" is not a proxy for "vocabulary size" or "sentence length".

Reproducibility
---------------
Each archive is fetched from its pinned public URL and its SHA-256 recorded in
``data/d2-beir/manifest.json`` alongside the sampling seed. Re-running reproduces the corpus
byte-for-byte, or fails loudly if upstream changed.

Usage::

    python data/fetch_d2.py
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import requests

BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

#: One BEIR sub-corpus per topic. The label is the sub-corpus name, which is external to the
#: embedding -- that is the whole point of the choice. Sizes are the download, not the sample.
DOMAINS: tuple[str, ...] = (
    "scifact",  # scientific claim verification
    "nfcorpus",  # nutrition / clinical
    "arguana",  # argumentation
    "scidocs",  # computer-science abstracts
    "fiqa",  # retail-investor finance
    "trec-covid",  # biomedical COVID literature
    "quora",  # community questions
    "webis-touche2020",  # controversial-topic argument mining
)

#: Entries sampled per domain. 1,000 x 8 = 8,000, comfortably above D3's 5,000 and enough that
#: the 256-partition index is not saturated at the sizes A4 sweeps.
PER_DOMAIN = 1_000

#: Global keyword vocabulary size. D3 uses 503 heavy-tailed terms; matching the order of
#: magnitude keeps the DSSE keyword model comparable across the two corpora rather than
#: confounding "real prose" with "different vocabulary scale".
VOCAB_SIZE = 512

#: Keywords per entry, matching D3's three tags per note.
TAGS_PER_ENTRY = 3

#: Shortest usable passage. Below this the text is a fragment and the embedding is noise.
MIN_CHARS = 120

#: Longest kept passage. bge-base truncates at 512 tokens anyway; keeping the tail would only
#: inflate the corpus on disk.
MAX_CHARS = 1_400

SEED = 20_260_827

OUT_DIR = Path(__file__).resolve().parent / "d2-beir"
CACHE = OUT_DIR / "_zips"

#: Function words and IR-boilerplate that would otherwise dominate any frequency-based keyword
#: extraction. Deliberately short: an aggressive list would start removing domain signal, and
#: the tf-idf weighting already suppresses genuinely ubiquitous terms.
STOP = frozenset(
    """
    about above after again against all also among and any are because been before being
    below between both but can cannot could did does doing down during each few for from
    further had has have having here how however into itself more most must not now off once
    only other ought our ours out over own same should such than that the their theirs them
    themselves then there these they this those through too under until very was were what
    when where which while who whom why will with would you your yours abstract background
    conclusion conclusions introduction method methods objective purpose result results study
    studies paper article using used use based show shown showed found data analysis
    """.split()
)

TOKEN = re.compile(r"[a-z][a-z\-]{3,}")


def _tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and not t.endswith("-")]


def fetch(name: str) -> Path:
    """Download one BEIR archive, cached. Returns the local path."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{name}.zip"
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  {name:20s} cached ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    url = BASE.format(name)
    print(f"  {name:20s} downloading ...", flush=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    print(f"  {name:20s} {dest.stat().st_size / 1e6:.1f} MB", flush=True)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_corpus(zip_path: Path, name: str, limit: int, rng) -> list[dict]:
    """Pull ``limit`` usable passages out of a BEIR archive's ``corpus.jsonl``.

    Reservoir-sampled while streaming, so a 640 MB archive never lands in memory and the
    sample is uniform over *usable* passages rather than over the first N lines (which in
    several BEIR corpora are ordered by document id and therefore by source publication).
    """
    inner = f"{name}/corpus.jsonl"
    kept: list[dict] = []
    seen = 0
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [n for n in zf.namelist() if n.endswith("corpus.jsonl")]
        if not candidates:
            raise FileNotFoundError(f"{zip_path}: no corpus.jsonl")
        inner = candidates[0]
        with zf.open(inner) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = (row.get("title") or "").strip()
                body = (row.get("text") or "").strip()
                text = f"{title}. {body}" if title else body
                text = " ".join(text.split())
                if len(text) < MIN_CHARS:
                    continue
                text = text[:MAX_CHARS]
                seen += 1
                item = {"source_doc_id": str(row.get("_id", seen)), "text": text}
                if len(kept) < limit:
                    kept.append(item)
                else:
                    j = int(rng.integers(0, seen))
                    if j < limit:
                        kept[j] = item
    print(f"  {name:20s} sampled {len(kept)} of {seen} usable")
    return kept


def build_vocabulary(docs: list[dict]) -> tuple[str, ...]:
    """Global keyword vocabulary, by tf-idf salience.

    Terms are ranked by ``df * log(N/df)`` -- frequent enough to be a real index term, not so
    frequent as to be a stopword. Ranking by raw frequency would fill the vocabulary with
    connective tissue; ranking by pure idf would fill it with hapax typos.
    """
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(_tokens(d["text"])))
    n = len(docs)
    scored = [
        (t, c * math.log(n / c))
        for t, c in df.items()
        if 3 <= c <= n * 0.30  # drop hapax noise and near-universal terms
    ]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return tuple(sorted(t for t, _ in scored[:VOCAB_SIZE]))


def assign_tags(docs: list[dict], vocab: tuple[str, ...]) -> None:
    """Give each document its ``TAGS_PER_ENTRY`` most salient in-vocabulary terms.

    In-place. Documents with too few in-vocabulary terms are padded from their own most
    frequent tokens' nearest vocabulary entries -- never dropped, because dropping them would
    correlate corpus membership with vocabulary richness and therefore with domain.
    """
    idx = {t: i for i, t in enumerate(vocab)}
    df: Counter[str] = Counter()
    for d in docs:
        df.update({t for t in set(_tokens(d["text"])) if t in idx})
    n = len(docs)
    for d in docs:
        tf = Counter(t for t in _tokens(d["text"]) if t in idx)
        ranked = sorted(
            tf.items(),
            key=lambda kv: (-(kv[1] * math.log(n / max(df[kv[0]], 1))), kv[0]),
        )
        tags = [t for t, _ in ranked[:TAGS_PER_ENTRY]]
        # Deterministic fallback keyed on the document's own text, so padding never becomes a
        # signal that identifies the padded documents as a group. The salt is a *strictly
        # increasing* counter rather than len(tags): keying it on len(tags) means that when the
        # candidate is already present, nothing about the input changes and the loop recomputes
        # the same hash forever. That is an infinite loop, and it burned twelve minutes of CPU
        # looking exactly like slow progress.
        salt = 0
        while len(tags) < TAGS_PER_ENTRY and salt < 4 * len(vocab):
            h = int(hashlib.sha256(f"{d['text']}|{salt}".encode()).hexdigest()[:8], 16)
            cand = vocab[h % len(vocab)]
            if cand not in tags:
                tags.append(cand)
            salt += 1
        # Guaranteed termination: if the salted walk somehow cannot find a distinct term, take
        # them in order. A document with fewer than TAGS_PER_ENTRY tags would otherwise be
        # silently different from every other document.
        for term in vocab:
            if len(tags) >= TAGS_PER_ENTRY:
                break
            if term not in tags:
                tags.append(term)
        d["tags"] = tags


def main() -> None:
    import numpy as np

    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"D2: {len(DOMAINS)} BEIR domains x {PER_DOMAIN} entries")
    archives: dict[str, Path] = {}
    for name in DOMAINS:
        archives[name] = fetch(name)

    docs: list[dict] = []
    for name in DOMAINS:
        for row in read_corpus(archives[name], name, PER_DOMAIN, rng):
            row["topic"] = name
            docs.append(row)

    print(f"  pooled {len(docs)} entries; building vocabulary ...")
    vocab = build_vocabulary(docs)
    assign_tags(docs, vocab)

    # Interleave here as well as in the loader: the pooled list is domain-blocked, and a
    # reader who loads this file with interleave_seed=None should not silently get cohorts
    # that differ by topic. See corpus.load()'s docstring for why that would be fatal.
    order = rng.permutation(len(docs))
    docs = [docs[int(i)] for i in order]

    out = OUT_DIR / "notes.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for seq, d in enumerate(docs):
            fh.write(
                json.dumps(
                    {
                        "note_id": f"d2-{seq:06d}",
                        "seq": seq,
                        "written_at_ms": 1_760_000_000_000 + seq * 1_000,
                        "topic": d["topic"],
                        "tags": d["tags"],
                        "text": d["text"],
                        "source": {"beir": d["topic"], "doc_id": d["source_doc_id"]},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "corpus": "D2",
        "description": "BEIR multi-domain public retrieval corpus, one topic per sub-corpus",
        "builder": "data/fetch_d2.py",
        "seed": SEED,
        "per_domain": PER_DOMAIN,
        "n_entries": len(docs),
        "n_topics": len(DOMAINS),
        "chance_floor": round(1.0 / len(DOMAINS), 6),
        "vocabulary_size": len(vocab),
        "tags_per_entry": TAGS_PER_ENTRY,
        "domains": [
            {"name": n, "url": BASE.format(n), "sha256": sha256(archives[n])} for n in DOMAINS
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    counts = Counter(d["topic"] for d in docs)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(docs)} entries, {len(vocab)} keywords, chance floor {1 / len(DOMAINS):.3f}")
    for t, c in sorted(counts.items()):
        print(f"    {t:20s} {c}")


if __name__ == "__main__":
    main()
