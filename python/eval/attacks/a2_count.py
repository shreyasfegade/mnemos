"""A2 -- the count attack.

Cash, Grubbs, Perry, Ristenpart, *Leakage-Abuse Attacks Against Searchable Encryption*, CCS
2015, §4: if the adversary knows the *distribution* of keyword frequencies in a corpus like the
target's, and the scheme reveals how many entries each keyword indexes, then keywords with
distinctive counts can be identified outright and the rest narrowed.

This attack consumes exactly one thing: :attr:`SnapshotView.token_counts`. It never touches a
transcript, a key, or a timing signal, which is what makes it the *snapshot* attack rather than
an access-pattern attack.

Relationship to A3
------------------
A3 is a more elaborate assignment-solver over the same channel; A2 is the direct, classical
version. Carrying both is deliberate: if the sophisticated one were the only implementation and
it scored badly, the natural objection would be that the solver was weak rather than the channel
narrow. A2 is simple enough that its score is a property of the leakage, not of the algorithm.

The ceiling, and why it is arithmetic rather than cryptographic
--------------------------------------------------------------
An adversary can only distinguish two keywords by count if their counts *differ*. In any real
corpus most keywords are rare, rare keywords have small integer counts, and small integers
collide en masse. :func:`tie_ceiling` computes the resulting upper bound directly from the count
multiset: it is the fraction of keywords whose count is unique, which no solver can exceed.

Measuring the ceiling alongside the attack is what turns "our attack got 0.12" into "0.12 is all
there is", and the second statement is the one worth publishing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from eval.lab.schemes import SnapshotView


@dataclass(frozen=True)
class CountResult:
    """One A2 trial."""

    scheme: str
    aux_fraction: float
    recovered: int
    total_tokens: int
    seed: int
    ceiling: float
    """Fraction of tokens with a *unique* count -- the hard upper bound on any count-only
    attack against this snapshot."""

    @property
    def advantage(self) -> float:
        chance = 1.0 / self.total_tokens if self.total_tokens else 0.0
        return max(0.0, self.recovered / self.total_tokens - chance)

    def as_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "aux_fraction": self.aux_fraction,
            "recovered": self.recovered,
            "total_tokens": self.total_tokens,
            "accuracy": self.recovered / self.total_tokens if self.total_tokens else 0.0,
            "advantage": self.advantage,
            "ceiling": self.ceiling,
            "seed": self.seed,
        }


def response_volumes(view: SnapshotView) -> dict[int, int]:
    """The channel, isolated: opaque token identifier -> number of entries indexed under it.

    Takes a :class:`SnapshotView` rather than a trace path because that object *is* the
    adversary's permitted view, enforced by its type. A function that reached past it would be
    measuring an adversary the threat model does not grant.
    """
    return dict(view.token_counts)


def tie_ceiling(counts: dict[int, int]) -> float:
    """Upper bound on count-only recovery: the fraction of tokens with a unique count.

    A token sharing its count with `k-1` others cannot be identified by count at all; the best
    any solver can do is guess among the `k`, and over many such groups that contributes
    nothing beyond chance. So the ceiling is the share of singleton counts.
    """
    if not counts:
        return 0.0
    freq = Counter(counts.values())
    unique = sum(1 for c in counts.values() if freq[c] == 1)
    return unique / len(counts)


def run(
    view: SnapshotView,
    truth: dict[int, str],
    aux_counts: dict[str, int],
    aux_fraction: float,
    seed: int,
) -> CountResult:
    """Match opaque tokens to keywords by count.

    Parameters
    ----------
    view
        The adversary's whole input.
    truth
        Token -> keyword. **Scoring only.** Never read before the prediction is formed.
    aux_counts
        The adversary's auxiliary knowledge: keyword -> expected count, from a corpus it
        believes resembles the target. Noised by ``1 - aux_fraction`` to model imperfect
        knowledge, because an adversary with the exact target counts is not a realistic one and
        would make the attack look far stronger than it is.
    """
    rng = np.random.default_rng(seed)
    counts = response_volumes(view)
    if not counts:
        # A history-independent store persists no counter map at all. Nothing to attack, and
        # that is the result rather than an error.
        return CountResult(
            scheme=view.scheme,
            aux_fraction=aux_fraction,
            recovered=0,
            total_tokens=max(1, len(truth)),
            seed=seed,
            ceiling=0.0,
        )

    # Degrade the auxiliary counts to model partial knowledge.
    noisy_aux: dict[str, float] = {}
    for kw, c in aux_counts.items():
        noise = rng.normal(0.0, (1.0 - aux_fraction) * max(1.0, c))
        noisy_aux[kw] = max(0.0, c + noise)

    # Greedy nearest-count assignment, each keyword used once. Simple by design -- see the
    # module docstring on why the solver must not be the interesting variable.
    remaining = dict(noisy_aux)
    prediction: dict[int, str] = {}
    for token, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        if not remaining:
            break
        best = min(remaining, key=lambda kw: abs(remaining[kw] - c))
        prediction[token] = best
        del remaining[best]

    recovered = sum(1 for t, kw in prediction.items() if truth.get(t) == kw)

    return CountResult(
        scheme=view.scheme,
        aux_fraction=aux_fraction,
        recovered=recovered,
        total_tokens=len(counts),
        seed=seed,
        ceiling=tie_ceiling(counts),
    )
