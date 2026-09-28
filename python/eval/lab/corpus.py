"""Corpus loading and the tau-arrival cohort split.

The D3 corpus (``data/d3-swebench/notes.jsonl``) carries three fields that make it usable as a
DSSE testbed rather than just text:

``tags``
    Three per note, drawn from a 503-term heavy-tailed vocabulary. This is the **keyword ->
    document mapping** that searchable encryption actually operates on. Without it we would be
    approximating the classic model; with it we can instantiate it directly.

``seq`` / ``written_at_ms``
    Authoring order. This is what lets us place tau and split the corpus into the historical
    cohort ``H`` (written before compromise, adversary sees only final state) and the
    post-compromise cohort ``P`` (written after, adversary sees everything).

Caveat worth carrying into the paper
------------------------------------
D3 is *synthetic and template-assembled*. Its tag distribution is realistic (heavy-tailed,
many ties) and that is all attack A3 consumes -- counts and order, never semantics. But recall
and nDCG numbers measured on D3 would be optimistic, because template text separates more
cleanly in embedding space than real prose. **Utility numbers come from D2 (MS MARCO / NQ),
not from here.** See ``the evaluation plan`` s5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def repo_root(start: Path | None = None) -> Path:
    """Walk up to the directory holding ``params.toml``.

    Same anchor :mod:`mnemos.params` uses, so every path in the project resolves identically
    regardless of which directory a script was launched from.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "params.toml").is_file():
            return candidate
    raise FileNotFoundError("params.toml not found — cannot locate the repository root")


DEFAULT_CORPUS = repo_root() / "data" / "d3-swebench" / "notes.jsonl"


@dataclass(frozen=True)
class Entry:
    """One memory the agent authored."""

    doc_id: int
    """Dense index, assigned in authoring order."""
    seq: int
    """Authoring sequence number from the corpus."""
    keywords: tuple[str, ...]
    """The note's tags -- its keyword set in the DSSE sense."""
    topic: str
    """Coarse semantic label, used for the entry-level re-identification metric."""
    text: str


@dataclass(frozen=True)
class Corpus:
    """A corpus split at compromise time tau."""

    historical: list[Entry]
    """Cohort H: authored before tau. The adversary never saw these being written."""
    post: list[Entry]
    """Cohort P: authored at or after tau. Fully observed, partly induced."""
    vocabulary: tuple[str, ...]
    """Every keyword appearing anywhere, in a fixed canonical order."""

    @property
    def all_entries(self) -> list[Entry]:
        return [*self.historical, *self.post]

    def keyword_counts(self, entries: list[Entry]) -> dict[str, int]:
        """``|DB(w)|`` for each keyword over the given entries.

        This is the quantity a forward-private index's persisted counter state exposes, and
        the quantity attack A3 matches against.
        """
        counts: dict[str, int] = {w: 0 for w in self.vocabulary}
        for e in entries:
            for w in e.keywords:
                counts[w] += 1
        return {w: c for w, c in counts.items() if c > 0}

    def postings(self, entries: list[Entry]) -> dict[str, list[int]]:
        """``DB(w)`` -- the document ids under each keyword, in authoring order."""
        out: dict[str, list[int]] = {}
        for e in entries:
            for w in e.keywords:
                out.setdefault(w, []).append(e.doc_id)
        return out


def load(
    path: Path = DEFAULT_CORPUS,
    n_entries: int | None = None,
    tau_fraction: float = 0.5,
    interleave_seed: int | None = 20260812,
) -> Corpus:
    """Load the corpus and split it at tau.

    Parameters
    ----------
    n_entries
        Truncate to ``n_entries`` after interleaving. Used for the N sweep.
    tau_fraction
        Where compromise happens, as a fraction of the corpus. **0.5 by default so the two
        cohorts are the same size** -- an unequal split would confound the H-vs-P comparison
        with a sample-size effect, which is precisely the confound a reviewer looks for first.
    interleave_seed
        Deterministic reshuffle of authoring order. **Required for a valid experiment**, see
        below. Pass ``None`` to keep the corpus's own ``seq`` order.

    Why authoring order is reshuffled
    ---------------------------------
    The D3 generator emits notes **topic-blocked**: entries 0-499 are all ``sql``, 500-999 all
    ``networking``, and so on. Its ``written_at_ms`` is monotone in ``seq``, so the corpus
    claims the agent spent its first thousand tasks exclusively on SQL and then never touched
    it again.

    Splitting that at tau would make cohorts H and P differ by **topic** rather than by
    **time** -- H would be SQL and networking, P would be everything else. Every measured
    difference between the cohorts would then be a topic effect wearing a compromise-time
    costume, and Delta would be meaningless.

    Real agents interleave: they work on whatever the environment hands them. A seeded
    permutation of authoring order restores that, keeps the run reproducible, and leaves the
    keyword-frequency distribution -- the only thing attack A3 consumes -- untouched.
    """
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    rows.sort(key=lambda r: r["seq"])

    if interleave_seed is not None:
        order = np.random.default_rng(interleave_seed).permutation(len(rows))
        rows = [rows[int(i)] for i in order]

    if n_entries is not None:
        rows = rows[:n_entries]

    entries = [
        Entry(
            doc_id=i,
            seq=r["seq"],
            keywords=tuple(r["tags"]),
            topic=r["topic"],
            text=r["text"],
        )
        for i, r in enumerate(rows)
    ]

    vocab = tuple(sorted({w for e in entries for w in e.keywords}))
    split = int(len(entries) * tau_fraction)

    return Corpus(historical=entries[:split], post=entries[split:], vocabulary=vocab)


def sample_auxiliary(
    entries: list[Entry],
    fraction: float,
    rng: np.random.Generator,
) -> list[Entry]:
    """The attacker's known-plaintext subset.

    Standard in the leakage-abuse literature (IKK, Cash et al.): the adversary knows some
    fraction of the corpus in the clear -- from a prior breach, a public subset, or simply
    knowing what kind of work the agent does -- and uses it to build the *expected* statistical
    profile it matches observations against.

    Sampled uniformly at random from the cohort under attack. Sampling by recency or by topic
    would model a different and more favourable adversary; uniform is the conservative choice.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"aux fraction must be in (0, 1], got {fraction}")
    k = max(1, int(len(entries) * fraction))
    idx = rng.choice(len(entries), size=k, replace=False)
    return [entries[i] for i in sorted(idx.tolist())]
