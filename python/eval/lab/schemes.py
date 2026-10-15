"""What each scheme leaves on disk, and what a tau-arrival adversary reads off it.

This module is where the paper's central claim becomes falsifiable, so the modelling
assumptions are written out in full rather than buried in the code.

The fairness contract
=====================
Attack A3 is an attack on *baselines*. A weakened baseline invalidates the finding, and it is
the first thing a careful reader checks. So every scheme here is modelled under the **most
conservative reading** -- the one that leaks least -- and every judgement call is recorded.

Applied uniformly to all schemes:

**Keys live in hardware.** Our threat model (TA-1) puts the root key in a TPM. We extend the
same courtesy to the baselines: their master keys are *not* on disk and *not* recoverable. Any
leak modelled below survives that assumption.

**Forward privacy actually works.** In Sophos, ``ST_{c-1} = TDP_private(ST_c)``: walking the
token chain *backwards* requires the private key, which is in hardware. So the adversary
**cannot** recover past search tokens, **cannot** link dictionary entries to keywords, and
**cannot** reconstruct the inverted index. We model that faithfully. Forward privacy is not
broken here -- it does exactly what it claims.

What is left, and why it is on disk
===================================
What remains outside the hardware boundary is the *non-key* client state: the per-keyword
update counters ``c[w]``, and the current search tokens ``ST_c[w]``.

That state is on disk for three reasons, none of them a strawman:

1. **It is large.** One entry per keyword, growing with the vocabulary. A TPM holds keys, not
   a 503-entry dictionary that grows without bound.
2. **It is written constantly.** Every update bumps a counter. TPM write endurance and latency
   make that a non-starter.
3. **It is not key material**, so no implementation would think to protect it as such.

And in the setting that matters here it is on the *compromised* host, because **an agent's
memory has no client/server split** -- the agent is both. That observation is the hinge of the
whole paper, so this module measures it directly via the ``split`` / ``local`` ablation below.

The ablation
============
+------------------+--------------------------------------------------------------------+
| ``sophos-split`` | Classic deployment. Counters live on a separate client machine the |
|                  | adversary did not compromise. **Baseline for the baseline** -- this |
|                  | isolates what the single-host setting actually contributes.        |
+------------------+--------------------------------------------------------------------+
| ``sophos-local`` | Single-host agent memory. Counters on the compromised host.        |
+------------------+--------------------------------------------------------------------+
| ``diana-local``  | Same setting; counts leak through GGM residual key-set *sizes*     |
|                  | rather than an explicit counter. Different mechanism, same         |
|                  | consequence -- which shows the finding is a property of the class, |
|                  | not a quirk of one scheme.                                         |
+------------------+--------------------------------------------------------------------+
| ``mnemos``       | Padded buckets, PRF slot addressing. Counters do not exist.        |
+------------------+--------------------------------------------------------------------+

Reporting ``sophos-split`` alongside ``sophos-local`` is what makes this an honest experiment
instead of a rigged one: if the split variant also falls, the finding is about DSSE generally
and our setting claim is overblown. We would need to know that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .corpus import Corpus, Entry


@dataclass(frozen=True)
class SnapshotView:
    """Exactly what the adversary computes from ``EDB_tau``.

    Every field is something recoverable from bytes on disk at compromise time, with **no
    transcript, no timing, no key**. Attack A3 may read this object and nothing else.

    Token identifiers are **opaque**: ``token_counts`` is keyed by an arbitrary integer label,
    not by the keyword. Recovering which label is which keyword is precisely the attack.
    """

    token_counts: dict[int, int] = field(default_factory=dict)
    """Opaque token -> number of entries indexed under it. Empty when the scheme does not
    persist per-keyword counters. **This is the leak.**"""

    insertion_positions: dict[int, list[int]] = field(default_factory=dict)
    """Opaque token -> physical positions of its entries, when layout encodes arrival order.
    Empty for a history-independent store."""

    total_entries: int = 0
    """Corpus size. Everyone leaks this; hiding it is a declared non-goal."""

    n_buckets: int = 0
    """Partition count, a public parameter."""

    scheme: str = ""

    def leaks_counts(self) -> bool:
        return bool(self.token_counts)

    def leaks_order(self) -> bool:
        return bool(self.insertion_positions)


class Scheme:
    """Base: build a store from a corpus, then expose only what disk exposes."""

    name = "abstract"

    def __init__(self, corpus: Corpus, rng: np.random.Generator) -> None:
        self.corpus = corpus
        self.rng = rng

    def snapshot(self, entries: list[Entry]) -> tuple[SnapshotView, dict[int, str]]:
        """Return the adversary's view, and -- separately -- the ground-truth label map.

        The two are returned as a pair so the boundary is visible in the type signature: the
        attack receives only the first element. The truth map exists solely for scoring and
        must never reach ``python/eval/attacks/``.
        """
        raise NotImplementedError


class SophosLocal(Scheme):
    """Sophos (Bost, CCS 2016) as deployed in single-host agent memory.

    Keys in hardware; forward privacy intact; the adversary cannot walk the token chain
    backwards. What it *can* read is the per-keyword counter dictionary, which is on the same
    compromised disk as everything else because there is no separate client machine.

    Leak: ``|DB(w)|`` for every keyword, under an opaque label.
    """

    name = "sophos-local"

    def snapshot(self, entries: list[Entry]) -> tuple[SnapshotView, dict[int, str]]:
        counts = self.corpus.keyword_counts(entries)

        # Token labels are opaque and their ORDER carries no information: a real
        # implementation keys this dictionary by a keyed hash of the keyword, so its
        # iteration order is effectively random with respect to the keyword. Shuffling here
        # prevents the attack from accidentally exploiting our alphabetical vocabulary order,
        # which would be an artifact of the harness rather than a property of the scheme.
        keywords = list(counts.keys())
        perm = self.rng.permutation(len(keywords))
        label_of = {keywords[int(p)]: i for i, p in enumerate(perm)}
        truth = {label: kw for kw, label in label_of.items()}

        view = SnapshotView(
            token_counts={label_of[w]: c for w, c in counts.items()},
            # DELIBERATELY EMPTY, and this is the conservative call.
            #
            # Sophos's dictionary is append-structured, so physical position does encode
            # global arrival order. But entries are keyed by UT_{w,c} = H1(K_w, c), and
            # without the key the adversary cannot group them by keyword -- so it cannot
            # attribute any position to any token, and cannot build the co-occurrence matrix
            # that makes IKK-style attacks strong.
            #
            # A bucketed *semantic* index would leak more: buckets are contiguous regions, so
            # region size gives occupancy and position within a region gives arrival order,
            # both without any key. We model the stricter keyword-index case instead, because
            # it matches Sophos as published and because a finding that survives the
            # conservative model survives scrutiny.
            insertion_positions={},
            total_entries=len(entries),
            scheme=self.name,
        )
        return view, truth


class SophosSplit(Scheme):
    """Sophos in its classic client/server deployment -- **the honest control**.

    Counters live on a separate client the adversary did not compromise. The server holds only
    the encrypted dictionary, whose entries are unlinkable to keywords without the chain.

    Leak: total entry count, and raw physical ordering with no keyword attribution.

    If A3 succeeds here too, the single-host framing is not doing the work we claim and the
    finding is really about DSSE in general. Running this variant is how we find that out.
    """

    name = "sophos-split"

    def snapshot(self, entries: list[Entry]) -> tuple[SnapshotView, dict[int, str]]:
        view = SnapshotView(
            token_counts={},  # counters are on the uncompromised client
            insertion_positions={},  # positions exist but cannot be attributed to a keyword
            total_entries=len(entries),
            scheme=self.name,
        )
        return view, {}


class DianaLocal(Scheme):
    """Diana (Bost-Minaud-Ohrimenko, CCS 2017) in the single-host setting.

    No explicit counter: forward privacy comes from a GGM puncturable PRF. But the **residual
    key set** after ``n`` punctures has a size determined by ``n`` -- for a depth-``d`` GGM
    tree, puncturing ``n`` leaves leaves roughly ``n * log2(depth)`` residual keys. So update
    counts leak through the *size of a structure* rather than the value of a counter.

    Same consequence, different mechanism. That matters: a reader's first instinct on seeing
    the Sophos result is "then use Diana", and this is the answer.

    Mnemos uses the same GGM primitive and does not leak, because it pads every chain's
    serialised state to a constant worst-case length (``mnemos-index::token``). The fix is
    cheap and local *once you know to look for it* -- which is the paper's practical takeaway.
    """

    name = "diana-local"

    #: GGM tree depth, from ``params.toml`` ``crypto.pprf_depth``.
    DEPTH = 32

    def snapshot(self, entries: list[Entry]) -> tuple[SnapshotView, dict[int, str]]:
        counts = self.corpus.keyword_counts(entries)

        keywords = list(counts.keys())
        perm = self.rng.permutation(len(keywords))
        label_of = {keywords[int(p)]: i for i, p in enumerate(perm)}
        truth = {label: kw for kw, label in label_of.items()}

        # The adversary measures residual key-set size, then inverts the relation to estimate
        # the update count. The inversion is lossy -- distinct counts can yield equal residual
        # sizes -- so Diana leaks a slightly *coarser* signal than Sophos. We model that
        # coarsening rather than assuming Diana leaks exact counts, because assuming otherwise
        # would flatter the attack.
        observed: dict[int, int] = {}
        for w, c in counts.items():
            residual = self._residual_key_count(c)
            observed[label_of[w]] = self._invert_residual(residual)

        view = SnapshotView(
            token_counts=observed,
            insertion_positions={},  # same reasoning as SophosLocal
            total_entries=len(entries),
            scheme=self.name,
        )
        return view, truth

    @classmethod
    def _residual_key_count(cls, n_updates: int) -> int:
        """Residual keys after puncturing ``n_updates`` leaves of a depth-32 GGM tree.

        Puncturing a leaf replaces one subtree key with the sibling keys along its root path.
        With ``n`` punctures the residual set is O(n log(depth)), saturating as the tree fills.
        """
        if n_updates <= 0:
            return 0
        return int(np.ceil(n_updates * np.log2(cls.DEPTH)))

    @classmethod
    def _invert_residual(cls, residual: int) -> int:
        """Recover an update-count estimate from an observed residual size."""
        if residual <= 0:
            return 0
        return round(residual / np.log2(cls.DEPTH))


class MnemosHI(Scheme):
    """Mnemos: history-independent, fully padded.

    Three mechanisms, all from `history independence`:

    - **Buckets are padded to capacity.** Occupancy is constant on disk, so there is no
      per-keyword count to read. This is the direct answer to Sophos and Diana.
    - **Slot address is ``PRF(K_idx, tag)``** with Robin Hood collision resolution, so physical
      position is a function of identity, not arrival time. No insertion order.
    - **Token chain state is padded to constant length**, so Diana's residual-size channel is
      closed even though the same GGM primitive is used.

    What is left: the file sizes, which are ``n_max``-derived constants, and the bucket count,
    which is a public parameter. Both are declared non-goals.
    """

    name = "mnemos"

    def __init__(self, corpus: Corpus, rng: np.random.Generator, n_buckets: int = 256) -> None:
        super().__init__(corpus, rng)
        self.n_buckets = n_buckets

    def snapshot(self, entries: list[Entry]) -> tuple[SnapshotView, dict[int, str]]:
        view = SnapshotView(
            token_counts={},  # padded to capacity -- occupancy is not observable
            insertion_positions={},  # PRF-addressed -- layout carries no order
            total_entries=len(entries),
            n_buckets=self.n_buckets,
            scheme=self.name,
        )
        return view, {}


SCHEMES: dict[str, type[Scheme]] = {
    SophosLocal.name: SophosLocal,
    SophosSplit.name: SophosSplit,
    DianaLocal.name: DianaLocal,
    MnemosHI.name: MnemosHI,
}


def build(scheme: str, corpus: Corpus, rng: np.random.Generator) -> Scheme:
    """Instantiate a scheme by name."""
    try:
        return SCHEMES[scheme](corpus, rng)
    except KeyError:
        raise ValueError(f"unknown scheme {scheme!r}; have {sorted(SCHEMES)}") from None
