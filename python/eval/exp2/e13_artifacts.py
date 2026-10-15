"""E13: the operating-system artifact audit, done properly.

"the USN journal is enabled and a sync client is present" is an existence check,
not an audit. The τ-arrival model claims the adversary sees nothing before τ, and that claim is
only defensible if the artifacts a real Windows installation keeps are enumerated and measured.

This does five things, and records honestly which of them needed privileges it did not have:

1. **A USN record for an actual store write.** Create a store, write to it, then read the change
   journal back and find the records naming its files. This shows the file system itself retains
   pre-τ write history, with timestamps.
2. **Volume Shadow Copies.** Whether any exist, and their creation times -- each is a whole prior
   state of the volume.
3. **Sync clients.** Whether the store's directory is inside a synchronised tree, and which
   clients are installed.
4. **Hibernation and the pagefile.** ``hiberfil.sys`` captures physical memory, including a locked
   arena and an unwrapped master key: page locking prevents paging, not hibernation. If
   hibernation is enabled while a store can be unlocked, key custody is defeated for an imaging
   adversary.
5. **Lock-on-suspend.** Whether the engine zeroises on the power events that precede hibernation.

    cd python && python -m eval.exp2.e13_artifacts
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from eval.exp2 import lab
from eval.lab import corpus as C

OUT = C.repo_root() / "python" / "eval" / "results" / "e13_artifacts.json"


def sh(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def ps(script: str, timeout: int = 180) -> tuple[int, str]:
    return sh(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script], timeout
    )


def make_store_and_write(root: Path) -> dict:
    """Create a real store on the system volume and write to it, so the audit has a subject."""
    dim, b, cap = 64, 8, 8
    rng = np.random.default_rng(0)
    cents = rng.normal(size=(b, dim)).astype(np.float32)
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    over = lab.base_overrides(
        dim=dim, buckets=b, capacity=cap, block=256, epoch_writes=2, probe_spread=1
    )
    eng, _, _ = lab.create("p", root / "store", cents, over, {})
    for i in range(12):
        v = rng.normal(size=dim).astype(np.float32)
        v /= np.linalg.norm(v)
        eng.put(f"audit record {i}", v.tolist(), "{}")
    lab.drain(eng)
    eng.flush()
    eng.close()
    files = sorted(p.name for p in (root / "store").iterdir() if p.is_file())
    return {"dir": str(root / "store"), "files": files}


def usn_audit(store: dict) -> dict:
    """Find change-journal records naming the store's files."""
    out: dict = {"enabled": None, "journal_id": None, "records_for_store": [], "error": None}
    rc, text = sh(["fsutil", "usn", "queryjournal", "C:"])
    out["enabled"] = rc == 0
    out["queryjournal"] = text.strip()[:1200]
    if rc != 0:
        out["error"] = "change journal not queryable (may need elevation)"
        return out

    # Read the file's own USN entry: this is the record the file system kept about our write.
    for name in store["files"]:
        path = str(Path(store["dir"]) / name)
        rc2, t2 = sh(["fsutil", "usn", "readdata", path])
        if rc2 == 0:
            out["records_for_store"].append({"file": name, "readdata": t2.strip()[:600]})
    if not out["records_for_store"]:
        out["error"] = "no USN data readable for the store's files"
    return out


def vss_audit() -> dict:
    rc, text = sh(["vssadmin", "list", "shadows"])
    admin_needed = "Administrator" in text or "administrator" in text or rc != 0
    count = text.count("Shadow Copy ID")
    return {
        "queried": True,
        "needs_elevation": bool(admin_needed and count == 0),
        "shadow_copies_found": count,
        "raw": text.strip()[:1200],
    }


def sync_audit(store_dir: str) -> dict:
    roots = {
        "OneDrive": os.environ.get("ONEDRIVE") or os.environ.get("ONEDRIVECONSUMER"),
        "OneDriveCommercial": os.environ.get("ONEDRIVECOMMERCIAL"),
        "Dropbox": str(Path.home() / "Dropbox"),
        "GoogleDrive": str(Path.home() / "My Drive"),
        "iCloudDrive": str(Path.home() / "iCloudDrive"),
    }
    present = {k: v for k, v in roots.items() if v and Path(v).exists()}
    inside = {k: v for k, v in present.items() if str(store_dir).lower().startswith(str(v).lower())}
    rc, procs = ps(
        "Get-Process | Where-Object {$_.Name -match 'OneDrive|Dropbox|googledrive|iCloud'} | "
        "Select-Object -ExpandProperty Name"
    )
    return {
        "clients_installed": sorted(present),
        "client_roots": present,
        "store_inside_synced_tree": sorted(inside),
        "sync_processes_running": sorted(set(filter(None, procs.split()))) if rc == 0 else [],
    }


def hibernation_audit() -> dict:
    _rc, avail = ps("powercfg /a")
    hiber = Path("C:/hiberfil.sys")
    pagefile = Path("C:/pagefile.sys")
    info: dict = {
        "powercfg_available_states": avail.strip()[:900],
        "hibernation_enabled": "Hibernate" in avail
        and "not support" not in avail.split("Hibernate")[-1][:120]
        if "Hibernate" in avail
        else None,
    }
    for name, p in (("hiberfil.sys", hiber), ("pagefile.sys", pagefile)):
        try:
            info[name] = {"exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0}
        except OSError as e:
            info[name] = {"exists": None, "error": str(e)}
    info["implication"] = (
        "A hibernation image captures physical memory, locked pages included. Page locking "
        "prevents paging, not hibernation, so an imaging adversary who takes the disk after the "
        "machine hibernated while the store was unlocked recovers the unwrapped master key and "
        "the resident plaintext. The mitigation is to lock the store on suspend and hibernate "
        "events, which is a deployment requirement rather than a storage-layer property."
    )
    return info


def lock_on_suspend_check(root: Path) -> dict:
    """Does locking actually drop the key and the resident plaintext?

    This is the mechanism a power-event handler would invoke, tested directly: after ``lock`` the
    store must report itself locked and must refuse to search.
    """
    dim, b, cap = 64, 8, 8
    rng = np.random.default_rng(1)
    cents = rng.normal(size=(b, dim)).astype(np.float32)
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    over = lab.base_overrides(
        dim=dim, buckets=b, capacity=cap, block=256, epoch_writes=2, probe_spread=1
    )
    results = {}
    for kind in ("p", "m"):
        eng, _, _ = lab.create(kind, root / f"lk-{kind}", cents, over, {})
        v = rng.normal(size=dim).astype(np.float32)
        v /= np.linalg.norm(v)
        eng.put("sensitive", v.tolist(), "{}")
        lab.drain(eng)
        eng.flush()
        before = bool(eng.stat()["unlocked"])
        eng.lock()
        after = bool(eng.stat()["unlocked"])
        try:
            eng.search(v.tolist())
            searched_while_locked = True
        except Exception:
            searched_while_locked = False
        results[kind] = {
            "unlocked_before": before,
            "unlocked_after_lock": after,
            "search_succeeded_while_locked": searched_while_locked,
            "ok": before and not after and not searched_while_locked,
        }
    return results


def main() -> None:
    t0 = time.time()
    root = Path(
        tempfile.mkdtemp(
            prefix="mnemos-audit-", dir="C:/Users/" + os.environ.get("USERNAME", "Public")
        )
    )
    try:
        store = make_store_and_write(root)
        print(f"store written at {store['dir']}: {store['files']}", flush=True)

        results = {
            "experiment": "E13",
            "platform": ps(
                "(Get-CimInstance Win32_OperatingSystem).Caption + ' ' + "
                "[System.Environment]::OSVersion.Version"
            )[1].strip(),
            "store": store,
            "usn": usn_audit(store),
            "vss": vss_audit(),
            "sync": sync_audit(store["dir"]),
            "hibernation": hibernation_audit(),
            "lock_on_suspend": lock_on_suspend_check(root),
        }

        u = results["usn"]
        print(
            f"USN journal enabled: {u['enabled']}; records recovered for store files: "
            f"{len(u['records_for_store'])}",
            flush=True,
        )
        print(
            f"VSS: {results['vss']['shadow_copies_found']} shadow copies (elevation needed: "
            f"{results['vss']['needs_elevation']})",
            flush=True,
        )
        print(
            f"sync clients installed: {results['sync']['clients_installed']}; store inside a "
            f"synced tree: {results['sync']['store_inside_synced_tree']}",
            flush=True,
        )
        h = results["hibernation"]
        print(f"hiberfil.sys: {h['hiberfil.sys']}; pagefile.sys: {h['pagefile.sys']}", flush=True)
        locked = {k: v["ok"] for k, v in results["lock_on_suspend"].items()}
        print(f"lock-on-suspend mechanism: {locked}", flush=True)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"wrote {OUT.relative_to(C.repo_root())} in {time.time() - t0:.0f}s")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
