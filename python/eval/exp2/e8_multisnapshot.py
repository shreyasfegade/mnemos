"""E8: what a sequence of file versions reveals about writes made *before* the compromise.

The review's C2, which is the most serious factual error in the paper. The claim was: "the
construction is history-independent, so multiple snapshots reveal no more about cohort H than one
does." History independence (Naor-Teague) concerns **one** representation against the operation
sequence that produced it. It says nothing about a **sequence** of representations. Two versions
of the partition file captured at times t1 < t2 diff to exactly the regions rewritten in between,
which is the write channel applied before tau. The paper's own anchoring instantiations -- a sync
account with version history, and repeated imaging -- are precisely what supplies such a sequence.

What this measures
==================
An adversary holding versions of the store captured every ``g`` flushes, for a range of ``g``,
diffs consecutive versions and asks, for each pre-tau record, which regions could have held it.

``exposure``   the fraction of pre-tau records whose true region is inside the candidate set its
               interval produces. At fine granularity this is 1.0: the record is definitely in
               there.
``precision``  the mean of 1/|candidates|, the probability of picking the right region by guessing
               inside the set. 1/B is the null; anything above it is leakage about cohort H.

Then the mitigation: with ``rewrite_at_lock`` the whole file is re-encrypted at a session boundary,
so any diff spanning that boundary covers every region and carries nothing. It only helps if the
adversary cannot capture versions more finely than a session, so the granularity sweep is what says
whether it is worth anything, and the real-client probe below is what says which granularity is
realistic.

    cd python && python -m eval.exp2.e8_multisnapshot
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e8_multisnapshot.json"

B = 64
CAP = 16
DIM = 768
BLOCK = 256
W = 4
N_HIST = 300
GRANULARITIES = (1, 2, 4, 16, 64)
TRIALS = 20


def region_digest(path: Path, b: int, cap: int, slot_len: int, header: int = 64) -> list[bytes]:
    """A cheap per-region fingerprint of the partition file."""
    import hashlib

    raw = path.read_bytes()
    region_len = cap * slot_len
    return [
        hashlib.sha1(raw[header + r * region_len : header + (r + 1) * region_len]).digest()
        for r in range(b)
    ]


def changed(a: list[bytes], b: list[bytes]) -> set[int]:
    return {i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y}


def run(granularity: int, rewrite_at_lock: bool, vecs, cents, trial: int) -> dict:
    rng = np.random.default_rng(600 + trial)
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.merge(
            lab.base_overrides(
                dim=DIM, buckets=B, capacity=CAP, block=BLOCK, epoch_writes=W, probe_spread=1
            ),
            {"v2": {"rewrite_at_lock": rewrite_at_lock}},
        )
        eng, _, _pp = lab.create("p", tmp / "s", cents, over, {})
        st = eng.stat()
        slot_len = int(st["slot_len"])
        part = tmp / "s" / "partitions.bin"

        versions = [region_digest(part, B, CAP, slot_len)]
        record_interval: list[int] = []  # which captured interval each record's write fell in
        record_region: list[int] = []

        flushes = 0
        idx = rng.choice(len(vecs), size=N_HIST, replace=False)
        for i in idx:
            try:
                r = eng.ground_truth_region(vecs[int(i)].tolist())
                eng.put(f"h{i}", vecs[int(i)].tolist(), "{}")
            except Exception:
                continue
            eng.flush()
            flushes += 1
            record_region.append(int(r))
            record_interval.append((flushes - 1) // granularity)
            if flushes % granularity == 0:
                versions.append(region_digest(part, B, CAP, slot_len))

        # A session boundary: the adversary's next capture comes after the store was closed.
        with contextlib.suppress(Exception):
            eng.close()
        versions.append(region_digest(part, B, CAP, slot_len))

        diffs = [changed(versions[k], versions[k + 1]) for k in range(len(versions) - 1)]

        exposed = 0
        precisions = []
        for reg, iv in zip(record_region, record_interval, strict=True):
            cand = diffs[iv] if iv < len(diffs) else set(range(B))
            if not cand:
                cand = set(range(B))
            if reg in cand:
                exposed += 1
            precisions.append(1.0 / len(cand))
        n = max(len(record_region), 1)
        return {
            "granularity_flushes": granularity,
            "rewrite_at_lock": rewrite_at_lock,
            "records": n,
            "versions_captured": len(versions),
            "exposure": exposed / n,
            "precision": float(np.mean(precisions)),
            "null_precision": 1.0 / B,
            "advantage": float(np.mean(precisions)) - 1.0 / B,
            "mean_candidates": float(np.mean([1.0 / p for p in precisions])),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cross_session(vecs, cents, rewrite_at_lock: bool, trial: int) -> dict:
    """The version-diff an adversary gets when it can only capture once per session."""
    rng = np.random.default_rng(800 + trial)
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.merge(
            lab.base_overrides(
                dim=DIM, buckets=B, capacity=CAP, block=BLOCK, epoch_writes=W, probe_spread=1
            ),
            {"v2": {"rewrite_at_lock": rewrite_at_lock}},
        )
        import mnemos_native as native

        eng, _, pp = lab.create("p", tmp / "s", cents, over, {})
        st = eng.stat()
        slot_len = int(st["slot_len"])
        part = tmp / "s" / "partitions.bin"

        idx = rng.choice(len(vecs), size=40, replace=False)
        for i in idx[:20]:
            with contextlib.suppress(Exception):
                eng.put(f"a{i}", vecs[int(i)].tolist(), "{}")
        lab.drain(eng)
        eng.close()
        v1 = region_digest(part, B, CAP, slot_len)

        eng2 = native.EngineV2.open(str(tmp / "s"), str(pp), True, {})
        eng2.unlock()
        regions = []
        for i in idx[20:]:
            try:
                regions.append(int(eng2.ground_truth_region(vecs[int(i)].tolist())))
                eng2.put(f"b{i}", vecs[int(i)].tolist(), "{}")
            except Exception:
                pass
        lab.drain(eng2)
        eng2.close()
        v2 = region_digest(part, B, CAP, slot_len)

        cand = changed(v1, v2) or set(range(B))
        hit = sum(1 for r in regions if r in cand) / max(len(regions), 1)
        return {
            "rewrite_at_lock": rewrite_at_lock,
            "candidate_regions": len(cand),
            "of_total": B,
            "second_session_records": len(regions),
            "exposure": hit,
            "precision": 1.0 / len(cand),
            "null_precision": 1.0 / B,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def real_client_probe() -> dict:
    """Best-effort: what a real sync client is doing on this machine.

    This cannot establish a retention policy without authenticating to the provider, and it does
    not pretend to. It records what is locally observable, and the granularity sweep above is what
    the security claim is actually parameterised on.
    """
    out: dict = {"note": "locally observable only; retention policy needs provider API access"}
    roots = {
        "OneDrive": os.environ.get("ONEDRIVE") or os.environ.get("ONEDRIVECONSUMER"),
        "OneDriveCommercial": os.environ.get("ONEDRIVECOMMERCIAL"),
        "Dropbox": str(Path.home() / "Dropbox"),
    }
    out["clients_present"] = {k: v for k, v in roots.items() if v and Path(v).exists()}
    root = next((Path(v) for v in out["clients_present"].values()), None)
    if root is None:
        out["measured"] = False
        return out
    # Rewrite one file repeatedly inside the synced tree and watch how the client reacts locally.
    probe = root / ".mnemos-sync-probe"
    try:
        probe.mkdir(exist_ok=True)
        f = probe / "churn.bin"
        sizes = []
        for _ in range(10):
            f.write_bytes(os.urandom(64 * 1024))
            time.sleep(0.4)
            sizes.append(f.stat().st_mtime)
        out["measured"] = True
        out["writes"] = len(sizes)
        out["distinct_mtimes"] = len(set(sizes))
        out["comment"] = (
            "Each local rewrite is a candidate upload. Whether the provider stores each as a "
            "retained version is a policy question this probe cannot answer, which is why the "
            "attack above treats capture granularity as an explicit parameter."
        )
    except Exception as e:
        out["measured"] = False
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args()
    t0 = time.time()

    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ][:4000]
    vecs = embed_cached([r["text"] for r in rows])
    cents = lab.kmeans_centroids(vecs, B, seed=0)
    print(f"B={B} C={CAP} w={W}; null precision {1.0 / B:.4f}", flush=True)

    # Empirical entropy of the region prior (records placed by content are region-skewed by topic
    # frequency), so the ledger's Fano bound uses the true H(X) rather than log2 B. a reader.
    import math

    reg = np.argmax(vecs @ cents.T, axis=1)
    rc = np.bincount(reg, minlength=B)
    tot = int(rc.sum())
    region_entropy = -sum((c / tot) * math.log2(c / tot) for c in rc if c > 0) if tot else 0.0

    results = {
        "experiment": "E8",
        "buckets": B,
        "region_entropy_bits": region_entropy,
        "epoch_writes": W,
        "claim_under_test": (
            "history independence does not bound what a SEQUENCE of versions reveals"
        ),
        "granularity_sweep": [],
        "cross_session": [],
    }

    for g in GRANULARITIES:
        runs = [run(g, False, vecs, cents, t) for t in range(args.trials)]
        m, lo, hi = lab.boot_ci([r["precision"] for r in runs])
        e, _, _ = lab.boot_ci([r["exposure"] for r in runs])
        cand = float(np.mean([r["mean_candidates"] for r in runs]))
        row = {
            "granularity_flushes": g,
            "precision": m,
            "lo": lo,
            "hi": hi,
            "exposure": e,
            "mean_candidate_regions": cand,
            "null_precision": 1.0 / B,
            "advantage": m - 1.0 / B,
        }
        results["granularity_sweep"].append(row)
        print(
            f"capture every {g:>3} flushes: exposure {e:.3f}  precision {m:.3f} "
            f"[{lo:.3f},{hi:.3f}]  ({cand:.1f} candidate regions of {B})",
            flush=True,
        )

    for ral in (False, True):
        runs = [cross_session(vecs, cents, ral, t) for t in range(args.trials)]
        m, lo, hi = lab.boot_ci([r["precision"] for r in runs])
        e, _, _ = lab.boot_ci([r["exposure"] for r in runs])
        cand = float(np.mean([r["candidate_regions"] for r in runs]))
        row = {
            "rewrite_at_lock": ral,
            "precision": m,
            "lo": lo,
            "hi": hi,
            "exposure": e,
            "mean_candidate_regions": cand,
            "null_precision": 1.0 / B,
        }
        results["cross_session"].append(row)
        print(
            f"cross-session diff, rewrite_at_lock={ral!s:>5}: exposure {e:.3f} "
            f"precision {m:.4f} ({cand:.1f} candidate regions of {B})",
            flush=True,
        )

    results["real_client"] = real_client_probe()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
