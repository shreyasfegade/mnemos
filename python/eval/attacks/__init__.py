"""Attacks A1-A5.

+----+--------------------------------+------------------------------+----------------------+
| ID | Attack                         | Source                       | Target               |
+====+================================+==============================+======================+
| A1 | IKK co-occurrence              | Islam et al., NDSS 2012      | Naive-EtS            |
| A2 | Count attack                   | Cash et al., CCS 2015        | Naive-EtS, Sophos    |
| A3 | **Snapshot index (ours)**      | This work                    | Sophos, Diana, Mnemos|
| A4 | Induced-write injection        | Zhang et al., USENIX'16      | all                  |
| A5 | Volumetric / query-equality    | Xu et al., CCS 2023          | **Mnemos**           |
+----+--------------------------------+------------------------------+----------------------+

Two rules govern this package, and both are load-bearing.

**1. Attacks may not cheat.**

Every module here imports :mod:`mnemos.trace` (the ``LeakEvent`` stream) and the baselines'
``snapshot_observables()`` interface. **None of them may import :mod:`mnemos.engine` or touch
private state.** The threat model says precisely what the adversary observes; an attack with
more than that proves nothing, and the failure mode is silent -- the attack still "works", it
just no longer measures what its name claims.

**2. We run attacks we expect to lose to.**

A5 is aimed at Mnemos. Xu et al. showed volumetric and query-equality leakage survives forward
*and* backward privacy; our answer is the constant-volume invariant, and A5 is how that answer
gets tested rather than asserted. If it partially succeeds, the residual is reported. A paper
that only runs the attacks its scheme survives reads as marketing, and readers in this area
recognise the pattern immediately.
"""

from __future__ import annotations

__all__ = [
    "a1_ikk",
    "a2_count",
    "a3_snapshot",
    "a4_injection",
    "a5_volumetric",
]
