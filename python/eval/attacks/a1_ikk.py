"""A1 -- IKK co-occurrence attack.

Islam, Kuzu, Kantarcioglu, *Access Pattern Disclosure on Searchable Encryption*, NDSS 2012.

The foundational access-pattern leakage-abuse attack: build a co-occurrence matrix from
observed access patterns, build the expected matrix from auxiliary knowledge, and match them by
simulated annealing.

Its job here is calibration, not novelty. A1 must **succeed** somewhere, or the harness is
broken and every subsequent "Mnemos resists attack X" claim is worthless -- an attack that fails
everywhere is not evidence of security anywhere. Run it first, confirm it works, then trust the
rest.

What it is pointed at, and why not at Naive-EtS
================================================
An earlier plan aimed A1 at the ``Naive-EtS`` baseline as the calibration target. That is the
wrong target and the reason is worth recording: Naive-EtS decrypts **the whole store** on every
search, so its access pattern is identical for every query and carries no co-occurrence signal
at all. A1 would score zero against it -- not because the harness is broken but because the
scheme has no access pattern to abuse. (Naive-EtS *does* leak, badly, through unpadded record
lengths; that is a different channel, tested in Rust by
``baseline_contrast::naive_ets_leaks_record_lengths_and_mnemos_does_not``.)

So A1 targets a **partitioned index**, where a query reads a query-dependent subset of
partitions and co-occurrence is exactly the signal IKK described. Against Mnemos it feeds E4:
how much does co-occurrence over ``(k_probe + d)`` bucket reads reveal, as a function of
``k_probe``, ``d`` and ``B``?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IkkResult:
    """One trial."""

    scheme: str
    aux_fraction: float
    recovered: int
    total_queries: int
    seed: int

    @property
    def advantage(self) -> float:
        chance = 1.0 / self.total_queries if self.total_queries else 0.0
        return max(0.0, self.recovered / self.total_queries - chance)

    def as_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "aux_fraction": self.aux_fraction,
            "recovered": self.recovered,
            "total_queries": self.total_queries,
            "accuracy": self.recovered / self.total_queries if self.total_queries else 0.0,
            "advantage": self.advantage,
            "seed": self.seed,
        }


def cooccurrence_matrix(read_sets: list[set[int]], n_queries: int) -> np.ndarray:
    """Observed co-occurrence: how often each pair of queries touched the same partition.

    Consumes read sets extracted from the adversary's transcript -- the ``bucket_read`` events
    and nothing else. This is the entire input to IKK.

    Normalised by the number of observations so the matrix is comparable to one built from an
    auxiliary corpus of a different size, which is the whole point of the attack.
    """
    m = np.zeros((n_queries, n_queries), dtype=np.float64)
    for i in range(n_queries):
        for j in range(n_queries):
            if i < len(read_sets) and j < len(read_sets):
                a, b = read_sets[i], read_sets[j]
                union = len(a | b)
                m[i, j] = len(a & b) / union if union else 0.0
    return m


def _energy(perm: np.ndarray, observed: np.ndarray, expected: np.ndarray) -> float:
    """Squared distance between the observed matrix and the permuted expected matrix.

    IKK's objective. The permutation that minimises it is the attack's guess at which observed
    query is which known query.
    """
    permuted = expected[np.ix_(perm, perm)]
    return float(np.sum((observed - permuted) ** 2))


def simulated_annealing(
    observed: np.ndarray,
    expected: np.ndarray,
    seed: int,
    iterations: int = 20_000,
) -> np.ndarray:
    """Minimise :func:`_energy` over permutations, as IKK does.

    Plain Metropolis with a geometric cooling schedule and pairwise swaps. Deliberately the
    textbook version: a heavily tuned solver would make the *result* a statement about the
    solver rather than about the leakage.
    """
    rng = np.random.default_rng(seed)
    n = observed.shape[0]
    perm = rng.permutation(n)
    energy = _energy(perm, observed, expected)

    t0, t1 = 1.0, 1e-4
    for step in range(iterations):
        temp = t0 * (t1 / t0) ** (step / max(1, iterations - 1))
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        candidate = perm.copy()
        candidate[i], candidate[j] = candidate[j], candidate[i]
        cand_energy = _energy(candidate, observed, expected)
        delta = cand_energy - energy
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            perm, energy = candidate, cand_energy
    return perm


def run(
    observed_read_sets: list[set[int]],
    aux_read_sets: list[set[int]],
    truth: list[int],
    scheme: str,
    aux_fraction: float,
    seed: int,
    iterations: int = 20_000,
) -> IkkResult:
    """Run one trial.

    Parameters
    ----------
    observed_read_sets
        Per-query partition read sets, from the transcript.
    aux_read_sets
        The adversary's belief about what each *known* query would read, from a corpus it
        believes resembles the target.
    truth
        Index into ``aux_read_sets`` for each observed query. **Scoring only.**
    """
    n = min(len(observed_read_sets), len(aux_read_sets), len(truth))
    if n < 2:
        return IkkResult(scheme, aux_fraction, 0, max(1, n), seed)

    observed = cooccurrence_matrix(observed_read_sets[:n], n)
    expected = cooccurrence_matrix(aux_read_sets[:n], n)

    # Degrade the auxiliary matrix to model imperfect knowledge. An adversary holding the exact
    # target co-occurrence matrix is not realistic and would make the attack look far stronger.
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, (1.0 - aux_fraction), expected.shape)
    expected = np.clip(expected + noise * expected.std(), 0.0, 1.0)

    perm = simulated_annealing(observed, expected, seed, iterations)
    recovered = int(sum(1 for i in range(n) if perm[i] == truth[i]))

    return IkkResult(
        scheme=scheme,
        aux_fraction=aux_fraction,
        recovered=recovered,
        total_queries=n,
        seed=seed,
    )
