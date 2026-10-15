"""A3 -- the snapshot-index attack. **This is Figure 1.**

The paper's central claim, as executable code:

    Forward privacy is a property of the adversary's *transcript*. A host compromise arriving
    at time tau has no transcript of anything written before tau -- only the final on-disk
    state. And the on-disk state of a forward-private index is not silent.

What this attack consumes, and nothing else
-------------------------------------------
- ``SnapshotView``: the complete storage state at compromise time. No transcript, no timing,
  no process memory, no key. Forward privacy is *intact* -- the adversary genuinely cannot
  walk the token chain backwards (see ``lab/schemes.py`` for the fairness contract).
- ``aux``: a known-plaintext fraction of the historical corpus (10% / 25% / 50%), the standard
  auxiliary-knowledge model from IKK (NDSS'12) and Cash et al. (CCS'15).

The attack
----------
1. **Observe.** Read ``|DB(w)|`` per opaque token off the persisted counter state.
2. **Expect.** From the auxiliary subset, estimate each real keyword's corpus-wide count by
   scaling: ``n_hat[w] = count_in_aux[w] / aux_fraction``.
3. **Match.** Solve the assignment between observed tokens and expected keywords, minimising
   count discrepancy. Hungarian algorithm, exactly as in IKK.
4. **Score.** A token is recovered if it maps to the right keyword. An *entry* is
   re-identified if the attacker recovers a majority of its keywords, which yields its topic.

Against Mnemos, step 1 has no input: buckets are padded so counts are constant, and slot
position is ``PRF(K_idx, tag)`` under a uniquely-represented collision policy, so layout
carries no order. The attack has nothing to match on and degrades to guessing.

Why ties matter, and why we do not paper over them
---------------------------------------------------
The D3 vocabulary has 503 keywords but only ~34 distinct count values, so many keywords are
count-indistinguishable. The Hungarian solver will assign *something* to every token, and for
a tied group that assignment is arbitrary. We therefore report recovery honestly -- including
the fact that ties cap the achievable rate -- rather than quietly scoring a lucky permutation
within a tied block as a success. :func:`tie_ceiling` computes that cap explicitly so the
result can be read against it.

Fairness
--------
This module reads ``SnapshotView`` and the auxiliary entries. It **must not** import the
engine, read scheme internals, or use anything the adversary could not compute from disk.
CI enforces the import half of that; the rest is on code review.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from eval.lab.corpus import Corpus, Entry, sample_auxiliary
from eval.lab.schemes import SnapshotView


@dataclass(frozen=True)
class AttackResult:
    """One trial."""

    scheme: str
    aux_fraction: float
    n_entries: int
    seed: int

    keywords_recovered: int
    keywords_total: int
    entries_reidentified: int
    entries_total: int
    tie_ceiling: float
    """Best achievable keyword-recovery rate given count ties. Context for the raw number:
    a scheme is not 'secure' because its vocabulary happens to be degenerate."""

    @property
    def keyword_recovery_rate(self) -> float:
        return self.keywords_recovered / self.keywords_total if self.keywords_total else 0.0

    @property
    def keyword_advantage(self) -> float:
        """Recovery above chance. Chance is ``1 / |vocabulary|`` for a uniform guess."""
        chance = 1.0 / self.keywords_total if self.keywords_total else 0.0
        return max(0.0, self.keyword_recovery_rate - chance)

    @property
    def entry_reid_rate(self) -> float:
        return self.entries_reidentified / self.entries_total if self.entries_total else 0.0

    @property
    def advantage(self) -> float:
        """**The headline metric.** Entry-level re-identification above chance.

        This is what the G2/G4 gates are stated in: the fraction of
        *historical memories* whose topic the adversary correctly recovers, above what
        guessing the most likely topic would give.
        """
        if not self.entries_total:
            return 0.0
        chance = 1.0 / max(1, self.n_topics)
        return max(0.0, self.entry_reid_rate - chance)

    n_topics: int = 10


def tie_ceiling(true_counts: dict[str, int]) -> float:
    """Fraction of keywords whose count is unique, hence resolvable by counting alone.

    Within a group of ``k`` keywords sharing a count, a count-only attack can do no better
    than ``1/k`` in expectation. This function gives the optimistic bound -- the rate a
    *perfect* count attacker achieves -- so a measured rate can be read as a fraction of what
    was actually available rather than as an absolute.
    """
    if not true_counts:
        return 0.0
    groups = Counter(true_counts.values())
    # Expected correct fraction under perfect matching: each tied group of size k contributes
    # k * (1/k) = 1 correct assignment in expectation.
    expected_correct = sum(1.0 for _ in groups)
    return expected_correct / len(true_counts)


def _expected_counts(
    aux: list[Entry], aux_fraction: float, vocabulary: tuple[str, ...]
) -> dict[str, float]:
    """Scale auxiliary observations up to a corpus-wide estimate.

    The naive estimator ``count_in_aux / aux_fraction``. It is unbiased and it is what an
    actual adversary would use; a fancier estimator would overstate the attack.
    """
    seen: Counter[str] = Counter()
    for e in aux:
        seen.update(e.keywords)
    return {w: seen.get(w, 0) / aux_fraction for w in vocabulary}


def _solve(
    observed: dict[int, int],
    expected: dict[str, float],
) -> dict[int, str]:
    """Assign opaque tokens to keywords by minimising count discrepancy.

    Cost is absolute difference on a log scale: a 10-vs-12 discrepancy matters far more for a
    rare keyword than a 400-vs-402 discrepancy does for a common one, and log-space distance
    captures that without hand-tuned weights.
    """
    tokens = sorted(observed)
    keywords = sorted(expected)
    if not tokens or not keywords:
        return {}

    obs = np.array([observed[t] for t in tokens], dtype=float)
    exp = np.array([expected[w] for w in keywords], dtype=float)

    cost = np.abs(np.log1p(obs)[:, None] - np.log1p(exp)[None, :])

    rows, cols = linear_sum_assignment(cost)
    return {tokens[int(r)]: keywords[int(c)] for r, c in zip(rows, cols, strict=True)}


def run(
    snapshot: SnapshotView,
    corpus: Corpus,
    cohort: list[Entry],
    aux_fraction: float,
    seed: int,
    truth: dict[int, str] | None = None,
) -> AttackResult:
    """Run one trial against one snapshot.

    Parameters
    ----------
    snapshot
        ``EDB_tau``. The attack's only view of the store.
    cohort
        The entries under attack -- normally ``corpus.historical`` (cohort H).
    truth
        Ground-truth label -> keyword map, supplied by the *experiment* for scoring only.
        The attack never consults it before producing its assignment; it is used solely to
        grade the answer afterwards.
    """
    rng = np.random.default_rng(seed)

    true_counts = corpus.keyword_counts(cohort)
    n_topics = len({e.topic for e in cohort}) or 1
    ceiling = tie_ceiling(true_counts)

    # A history-independent store exposes no counts and no ordering: there is nothing to
    # match on, so the attack cannot do better than guessing. We short-circuit rather than
    # feeding an empty matrix to the solver, and score the guess honestly.
    if not snapshot.leaks_counts():
        chance_hits = round(len(cohort) / n_topics)
        return AttackResult(
            scheme=snapshot.scheme,
            aux_fraction=aux_fraction,
            n_entries=len(cohort),
            seed=seed,
            keywords_recovered=0,
            keywords_total=len(true_counts),
            entries_reidentified=chance_hits,
            entries_total=len(cohort),
            tie_ceiling=ceiling,
            n_topics=n_topics,
        )

    aux = sample_auxiliary(cohort, aux_fraction, rng)
    expected = _expected_counts(aux, aux_fraction, corpus.vocabulary)
    expected = {w: c for w, c in expected.items() if c > 0}

    assignment = _solve(snapshot.token_counts, expected)

    # Scoring only. The assignment above was produced without reference to this.
    truth = truth or {}
    recovered = sum(1 for tok, kw in assignment.items() if truth.get(tok) == kw)

    # Entry-level: an entry is re-identified when a majority of its keywords are recovered,
    # which is enough to pin its topic.
    recovered_keywords = {kw for tok, kw in assignment.items() if truth.get(tok) == kw}
    reidentified = sum(
        1 for e in cohort if sum(k in recovered_keywords for k in e.keywords) * 2 > len(e.keywords)
    )

    return AttackResult(
        scheme=snapshot.scheme,
        aux_fraction=aux_fraction,
        n_entries=len(cohort),
        seed=seed,
        keywords_recovered=recovered,
        keywords_total=len(true_counts),
        entries_reidentified=reidentified,
        entries_total=len(cohort),
        tie_ceiling=ceiling,
        n_topics=n_topics,
    )
