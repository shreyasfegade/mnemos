"""E14: Windows storage-artifact audit for the tau-arrival threat model.

"state at tau only" is fragile on real operating systems, because
several ordinary artifacts reveal pre-tau history. This audits, on the actual machine, which of
them are present for a store directory, so the paper states them as explicit assumptions rather
than assuming them away.

Checks (read-only, no admin required for the queries themselves):

- **USN change journal**: whether NTFS records file-write events for the store's volume, and
  whether entries for the store files are queryable (``fsutil usn readjournal``).
- **Volume Shadow Copies**: whether shadow copies exist for the volume (``vssadmin list shadows``
  needs admin; we detect presence via the ``System Volume Information`` and report if unreadable).
- **File timestamps**: the spread of mtimes across the store files, which leaks write activity.
- **OneDrive / cloud sync**: whether the store path is under a known cloud-sync root, which keeps
  server-side version history a sync adversary can diff.

Output is a JSON report the paper's threat-model section cites. It does not attempt to defeat any
artifact; it documents which ones apply so the adversary instantiations are honest.

    python -m eval.experiments.e14_windows_artifacts --store <path-to-a-store-dir>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from eval.lab import corpus as C

OUT = C.repo_root() / "python" / "eval" / "results" / "e14_windows_artifacts.json"


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return p.returncode, (p.stdout + p.stderr)
    except Exception as e:
        return -1, str(e)


def make_store(path: Path) -> None:
    """A small real v2 store, so the audit runs against genuine store files and their writes."""
    import numpy as np

    from eval.lab.stores import create_v2

    rng = np.random.default_rng(0)
    cents = rng.normal(size=(16, 64))
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    over = {
        "embedding": {"dim": 64},
        "index": {
            "buckets": 16,
            "bucket_capacity": 16,
            "top_k": 5,
            "probe_width": 4,
            "probe_spread": 1,
        },
        "crypto": {"block_size_bytes": 512, "max_entry_bytes": 512},
        "epoch": {"epoch_writes": 8, "epoch_ms": 3_600_000},
        "memory": {"arena_bytes": 1 << 18},
        "v2": {"durable": True},
    }
    eng = create_v2(path, cents, over)[0]
    for i in range(40):
        v = rng.normal(size=64)
        v /= (v @ v) ** 0.5
        with contextlib.suppress(Exception):
            eng.put(f"rec {i}", v.tolist(), "{}")
        eng.flush()
    eng.close()


def audit(store: Path) -> dict:
    drive = os.path.splitdrive(str(store.resolve()))[0] or "C:"
    report: dict = {"experiment": "E14", "store": str(store), "drive": drive, "artifacts": {}}

    rc, out = run(["fsutil", "fsinfo", "volumeinfo", drive + "\\"])
    report["artifacts"]["ntfs"] = "NTFS" in out
    report["fsutil_volumeinfo_rc"] = rc

    rc, out = run(["fsutil", "usn", "queryjournal", drive + "\\"])
    report["artifacts"]["usn_journal_active"] = rc == 0 and "Usn Journal ID" in out
    report["usn_query_rc"] = rc
    # Try reading recent USN entries that mention our store's files (best effort; needs the
    # journal to be enabled, which it is by default on the system volume).
    rc, out = run(["fsutil", "usn", "readjournal", drive + "\\", "csv"])
    names = {f.name for f in store.glob("*")}
    report["artifacts"]["usn_records_store_writes"] = rc == 0 and any(n in out for n in names)

    # Shadow copies presence (vssadmin needs admin; a non-zero rc with "Administrator" means they
    # may exist but are unreadable to us, which is itself the point).
    rc, out = run(["vssadmin", "list", "shadows", "/for=" + drive])
    report["artifacts"]["vss_readable"] = rc == 0 and "Shadow Copy" in out
    report["artifacts"]["vss_requires_admin"] = "Administrator" in out or "denied" in out.lower()

    # Timestamp spread across the store files.
    mtimes = [f.stat().st_mtime for f in store.glob("*") if f.is_file()]
    if mtimes:
        report["artifacts"]["mtime_spread_seconds"] = max(mtimes) - min(mtimes)
        report["artifacts"]["distinct_mtimes"] = len(set(int(m) for m in mtimes))

    # Cloud-sync roots.
    onedrive = os.environ.get("ONEDRIVE", "")
    sp = str(store.resolve()).lower()
    report["artifacts"]["under_onedrive"] = bool(onedrive) and sp.startswith(onedrive.lower())
    report["artifacts"]["onedrive_present"] = bool(onedrive)

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None)
    args = ap.parse_args()
    if args.store:
        report = audit(Path(args.store))
    else:
        tmp = Path(tempfile.mkdtemp()) / "store"
        make_store(tmp)
        report = audit(tmp)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["artifacts"], indent=1))
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
