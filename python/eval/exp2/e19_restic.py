"""E19: the multi-version channel through a real backup tool.

E8 diffs file versions it captures itself. A reader asked whether a real versioning tool keeps
versions at a granularity the attack can use, and what its own chunking does to the diff. This backs
the store up with restic 0.19 (content-defined chunking, encrypted repository) after every flush or
every few flushes, then plays two adversaries against the repository:

* **key holder** -- holds the repository and its password (a compromised backup host or account
  whose credentials sit beside the repository). It restores each snapshot through restic's own
  ``dump`` interface and runs E8's region diff on consecutive restored versions. Prediction: the
  channel survives intact, because restic restores byte-exact files.
* **bucket holder** -- holds only the encrypted repository. It sees how much new data each snapshot
  added. Recorded descriptively: restic chunks at about a megabyte, so a few rewritten regions
  upload whole chunks, which is also the upload amplification a backup of this store pays.

Run against Mnemos-P (content placement, where a diff names a region whose topic matters) and
Mnemos-M (sweep flush, where the diff names a cursor range that carries no topic); for Mnemos-M
only the upload volume is recorded.

The restic password is random per run and never written to disk.

    cd python && python -m eval.exp2.e19_restic
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e19_restic.json"
RESTIC = C.repo_root() / "tools" / "bin" / "restic.exe"

B = 64
CAP = 16
DIM = 768
BLOCK = 256
W = 4
N_HIST = 60
GRANULARITIES = (1, 4)
TRIALS = 5


def restic(env: dict, *args: str, stdin: bytes | None = None) -> bytes:
    r = subprocess.run([str(RESTIC), *args], env=env, input=stdin, capture_output=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"restic {args[0]} failed: {r.stderr.decode(errors='replace')[:400]}")
    return r.stdout


def backup(env: dict, data: bytes) -> dict:
    """Back the file up from stdin; return restic's JSON summary line."""
    out = restic(
        env, "backup", "--json", "--quiet", "--stdin", "--stdin-filename", "store.bin", stdin=data
    )
    lines = [json.loads(x) for x in out.decode().splitlines() if x.strip().startswith("{")]
    return next(x for x in lines if x.get("message_type") == "summary")


def region_digests(raw: bytes, slot_len: int, header: int = 64) -> list[bytes]:
    region_len = CAP * slot_len
    return [
        hashlib.sha1(raw[header + r * region_len : header + (r + 1) * region_len]).digest()
        for r in range(B)
    ]


def run(kind: str, g: int, vecs, cents, trial: int) -> dict:
    rng = np.random.default_rng(1900 + trial)
    tmp = Path(tempfile.mkdtemp())
    env = {
        **os.environ,
        "RESTIC_REPOSITORY": str(tmp / "repo"),
        "RESTIC_PASSWORD": secrets.token_hex(16),
        "RESTIC_CACHE_DIR": str(tmp / "cache"),
    }
    try:
        restic(env, "init", "--quiet")
        over = lab.base_overrides(
            dim=DIM, buckets=B, capacity=CAP, block=BLOCK, epoch_writes=W, probe_spread=1
        )
        eng, _, _ = lab.create(kind, tmp / "s", cents, over, {})
        slot_len = int(eng.stat().get("slot_len", 0))
        part = tmp / "s" / ("partitions.bin" if kind == "p" else "slab.bin")

        summaries = [backup(env, part.read_bytes())]
        rewritten_bytes = []
        regions: list[int] = []
        intervals: list[int] = []
        flushes = 0
        prev = part.read_bytes()
        for i in rng.choice(len(vecs), size=N_HIST, replace=False):
            v = vecs[int(i)].tolist()
            try:
                r = eng.ground_truth_region(v) if kind == "p" else -1
                eng.put(f"h{i}", v, "{}")
            except Exception:
                continue
            eng.flush()
            flushes += 1
            regions.append(int(r))
            intervals.append((flushes - 1) // g)
            if flushes % g == 0:
                now = part.read_bytes()
                rewritten_bytes.append(sum(1 for a, b in zip(prev, now, strict=True) if a != b))
                prev = now
                summaries.append(backup(env, now))
        with contextlib.suppress(Exception):
            eng.close()

        added = [s["data_added"] for s in summaries[1:]]
        blobs = [s["data_blobs"] for s in summaries[1:]]
        row = {
            "kind": kind,
            "granularity_flushes": g,
            "file_bytes": len(prev),
            "snapshots": len(summaries),
            "mean_data_added": float(np.mean(added)),
            "cv_data_added": float(np.std(added) / max(np.mean(added), 1)),
            "mean_new_blobs": float(np.mean(blobs)),
            "mean_rewritten_bytes": float(np.mean(rewritten_bytes)),
            "upload_amplification": float(np.mean(added) / max(np.mean(rewritten_bytes), 1)),
        }
        if kind != "p":
            return row

        # Key holder: restore every snapshot through restic and diff consecutive versions.
        ids = [s["snapshot_id"] for s in summaries]
        versions = [region_digests(restic(env, "dump", sid, "/store.bin"), slot_len) for sid in ids]
        diffs = [
            {k for k, (x, y) in enumerate(zip(a, b, strict=True)) if x != y}
            for a, b in itertools.pairwise(versions)
        ]
        prec, exposed = [], 0
        for reg, iv in zip(regions, intervals, strict=True):
            cand = (diffs[iv] if iv < len(diffs) else set()) or set(range(B))
            exposed += reg in cand
            prec.append(1.0 / len(cand))
        row.update(
            {
                "precision": float(np.mean(prec)),
                "exposure": exposed / max(len(regions), 1),
                "null_precision": 1.0 / B,
                "restored_byte_exact": restic(env, "dump", ids[-1], "/store.bin") == prev,
            }
        )
        return row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
    ver = restic({**os.environ}, "version").decode().split()[1]
    out: dict = {"experiment": "E19", "restic": ver, "buckets": B, "rows": []}
    for kind in ("p", "m"):
        for g in GRANULARITIES:
            runs = [run(kind, g, vecs, cents, t) for t in range(args.trials)]
            agg = {"kind": kind, "granularity_flushes": g, "trials": len(runs)}
            for k in (
                "mean_data_added",
                "mean_rewritten_bytes",
                "upload_amplification",
                "cv_data_added",
                "mean_new_blobs",
                "file_bytes",
            ):
                agg[k] = float(np.mean([r[k] for r in runs]))
            if kind == "p":
                m, lo, hi = lab.boot_ci([r["precision"] for r in runs])
                agg.update(
                    precision=m,
                    lo=lo,
                    hi=hi,
                    null_precision=1.0 / B,
                    exposure=float(np.mean([r["exposure"] for r in runs])),
                    restored_byte_exact=all(r["restored_byte_exact"] for r in runs),
                )
            out["rows"].append(agg)
            msg = (
                f" precision {agg['precision']:.3f} [{agg['lo']:.3f},{agg['hi']:.3f}]"
                if (kind == "p")
                else ""
            )
            print(
                f"{kind} g={g}: added {agg['mean_data_added'] / 1024:.0f} KiB/snapshot for "
                f"{agg['mean_rewritten_bytes'] / 1024:.0f} KiB rewritten "
                f"(x{agg['upload_amplification']:.1f}){msg}",
                flush=True,
            )
    out["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())} in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
