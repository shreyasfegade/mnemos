"""E5 -- run attack A5 against Mnemos itself, on a real store.

This is the self-directed attack. It builds an actual store through the compiled engine, issues
a controlled query workload, captures the ``LeakEvent`` stream the engine really emits, and
attacks it. Nothing here is simulated; if the constant-volume invariant were broken, this would
find it.

Usage::

    cd python && python -m eval.experiments.e5_volumetric

Reports two numbers and they point in opposite directions
=========================================================
**Volume leakage is zero.** Every search reads the same number of partitions and the same number
of content slots, whatever was asked and whatever matched. That is the invariant working.

**Query equality leaks completely.** The probe set is a deterministic function of the query
vector's position relative to the centroids -- no token, no epoch, no key enters that
computation -- so two searches for the same thing are byte-identical in the read pattern, and
remain so forever. Token rotation does not touch it, because tokens govern which *entries* a
search recognises, not which *partitions* it reads.

The honest consequence is that Mnemos protects *what is stored* far better than *what is asked*.
A repeated query is recognisable as repeated. Whether that matters depends on the deployment,
and the paper says so in its limitations rather than leaving a reader to discover it.

Mitigations, none of them free
==============================
* ``decoy_buckets > 0`` adds per-epoch noise, but the real probes are common to every
  observation of a query, so intersecting two epochs recovers them. It raises the observation
  count required; it does not close the channel. Decoys are *deliberately* deterministic --
  fresh ones would cancel by intersection -- which is exactly why they cannot help here.
* Probing a *random* superset each time closes the pairwise channel, at the cost of reading
  more partitions per query. **This is now implemented** (``probe_spread``,
  ``probe_retention``) and measured by :mod:`eval.experiments.e10_probe_spread`; this
  experiment establishes the ``s = 1, p = 1.0`` baseline it is measured against.
* Full ORAM over the partition table closes it properly, and is the cost this design exists to
  avoid (ADR-0002).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from eval.attacks import a5_volumetric as A5
from eval.lab import corpus as C

#: Distinct queries in the workload.
N_QUERIES = 12
#: Searches issued in total, sampled from those queries with repeats.
N_SEARCHES = 60
#: Corpus size. Small on purpose -- the leakage is structural and does not need scale to show.
N_ENTRIES = 240
SEED = 20_260_814


def _params_path() -> Path:
    return C.repo_root() / "python" / "tests" / "params-test.toml"


def run_trial(scratch: Path, seed: int) -> tuple[A5.VolumetricResult, float]:
    """Build a real store, drive it, and attack the trace it emits."""
    from mnemos.mcp_server import MnemosMemoryServer

    os.environ.setdefault("MNEMOS_I_ACCEPT_INSECURE", "1")
    rng = random.Random(seed)

    store = scratch / "store"
    trace = scratch / "trace.jsonl"

    server = MnemosMemoryServer(store, params_path=_params_path(), allow_insecure_keystore=True)
    server.open(create_if_missing=True)

    corpus = C.load()
    entries = corpus.all_entries[:N_ENTRIES]
    for e in entries:
        server.put(e.text[:200], {"topic": e.topic})

    # Distinct queries drawn from the corpus, so they are realistic rather than random noise.
    queries = [entries[rng.randrange(len(entries))].text[:120] for _ in range(N_QUERIES)]

    # Tracing starts AFTER the writes: the attack under test is about search leakage, and
    # leaving the write events in would just make the segmentation harder for no reason.
    server.record_trace(trace)

    truth: list[int] = []
    boundaries: list[int] = []
    for i in range(N_SEARCHES):
        q = rng.randrange(N_QUERIES)
        server.search(queries[q])
        truth.append(q)
        # Force an epoch boundary a third of the way in, so intra- and inter-epoch pairs both
        # exist and the "does rotation help?" question is actually asked.
        if i == N_SEARCHES // 3:
            server.put("a write to force an epoch boundary")
            boundaries.append(len(truth))

    server.close()

    result = A5.analyse(trace, truth, boundaries, seed=seed, epoch_ms=500)
    fpr = A5.false_positive_rate(trace, truth)
    return result, fpr


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        result, fpr = run_trial(Path(tmp), SEED)

    print("=== E5: A5 volumetric + query-equality, against Mnemos ===\n")
    print(f"  bucket reads per search      : {result.bucket_reads_per_search}")
    print(f"  content slot reads per search: {result.slot_reads_per_search}")
    print(f"  variance in reads per search : {result.volume_variance:.6f}   <- must be 0.0")
    print(f"  volume-based recovery        : {result.volume_recovery:.3f}")
    print()
    print(f"  distinct queries issued      : {result.distinct_queries}")
    print(f"  distinct read signatures     : {result.distinct_read_signatures}")
    print(f"  query-equality, intra-epoch  : {result.query_equality_intra_epoch:.3f}")
    print(f"  query-equality, inter-epoch  : {result.query_equality_inter_epoch:.3f}")
    print(f"  false-positive rate          : {fpr:.3f}")
    print()

    if result.volume_variance != 0.0:
        print("  !! VOLUME LEAK: reads per search are not constant. This is a BUG, not a")
        print("     finding -- some path took an early exit or skipped an empty partition.")
    else:
        print("  Volume: CLOSED. Constant-volume invariant holds on a real store.")

    if result.query_equality_inter_epoch > 0.5:
        print("  Query equality: OPEN, and not bounded by an epoch. The probe set is a")
        print("     function of the query alone, so repetition is recognisable indefinitely.")
        print("     Reported as a limitation; see this module's docstring for the options.")

    out = C.repo_root() / "python" / "eval" / "results" / "e5_volumetric.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    payload["false_positive_rate"] = fpr
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
