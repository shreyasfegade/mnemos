"""E33: what a real synchronization client's version history gives the multi-version channel.

E8 sweeps the capture granularity ``g`` as a parameter and E19 backs the store up with restic on a
schedule the harness chooses. A synchronization client chooses its own schedule: Syncthing waits
for its file-system watcher (10 s by default) before syncing a change, and its staggered versioning
then thins the archived versions (one per 30 s for the first hour, one per hour for a day, ...).
This runs two Syncthing 2.x instances on loopback -- a send-only "laptop" and a receive-only
"backup" with staggered versioning -- while an agent-paced workload writes a Mnemos-P store into the
synchronized folder, and then reads what the backup retained.

Every retained version is matched to the flush whose whole-file hash it equals (the harness logs
one hash per flush), which gives the realized capture granularity exactly. The version diff is then
scored as in E19: for each record, the regions that changed between the two retained versions
around its flush are the adversary's candidates, ``precision`` is one over their number (the
``1/B`` null when a diff covers the file) and ``exposure`` the fraction of records whose region is
among them.

The versioner's cleanup runs every 60 s here (its default is hourly), so the retention measured is
the thinned schedule; with the default, more versions survive between cleanups.

    cd python && python -m eval.exp2.e33_syncthing --minutes 30
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import secrets
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from eval.exp2 import lab
from eval.exp2.oracle_common import write_json
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

RES = C.repo_root() / "python" / "eval" / "results"
OUT = RES / "e33_syncthing.json"
ST = C.repo_root() / "tools" / "bin" / "syncthing.exe"

B, CAP, DIM, BLOCK, W = 64, 16, 768, 256, 4


def free_port() -> int:
    """A port the OS hands out right now, so Windows' excluded ranges are never chosen."""
    import socket

    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return int(sk.getsockname()[1])


class Node:
    def __init__(self, name: str, home: Path) -> None:
        self.name, self.home = name, home
        self.gui, self.listen = free_port(), free_port()
        self.key = secrets.token_hex(16)
        self.proc: subprocess.Popen | None = None
        subprocess.run(
            [str(ST), "generate", f"--home={home}", "--no-port-probing"],
            check=True,
            capture_output=True,
        )
        self.id = (
            subprocess.run(
                [str(ST), "device-id", f"--home={home}"], check=True, capture_output=True
            )
            .stdout.decode()
            .strip()
        )
        # Loopback only from the first start: no discovery, relays, NAT or default 22000 listener.
        cfg = home / "config.xml"
        x = cfg.read_text(encoding="utf-8")
        for a, b in (
            (
                "<listenAddress>default</listenAddress>",
                f"<listenAddress>tcp://127.0.0.1:{self.listen}</listenAddress>",
            ),
            (
                "<globalAnnounceEnabled>true</globalAnnounceEnabled>",
                "<globalAnnounceEnabled>false</globalAnnounceEnabled>",
            ),
            (
                "<localAnnounceEnabled>true</localAnnounceEnabled>",
                "<localAnnounceEnabled>false</localAnnounceEnabled>",
            ),
            ("<relaysEnabled>true</relaysEnabled>", "<relaysEnabled>false</relaysEnabled>"),
            ("<natEnabled>true</natEnabled>", "<natEnabled>false</natEnabled>"),
        ):
            x = x.replace(a, b)
        cfg.write_text(x, encoding="utf-8")

    def start(self, log: Path) -> None:
        self.proc = subprocess.Popen(
            [
                str(ST),
                "serve",
                f"--home={self.home}",
                "--no-browser",
                "--no-restart",
                "--no-upgrade",
                f"--gui-address=http://127.0.0.1:{self.gui}",
                f"--gui-apikey={self.key}",
                f"--log-file={log}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        t0 = time.time()
        while True:
            try:
                self.api("GET", "/rest/system/ping")
                return
            except Exception:
                if time.time() - t0 > 60 or self.proc.poll() is not None:
                    tail = log.read_text(errors="replace").splitlines()[-8:] if log.exists() else []
                    raise RuntimeError(f"syncthing {self.name} did not start: {tail}") from None
                time.sleep(0.5)

    def api(self, method: str, path: str, body=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.gui}{path}",
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={"X-API-Key": self.key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
        return json.loads(raw) if raw.strip() else None

    def configure(self, peer: Node, folder: Path, kind: str) -> None:
        self.api(
            "PATCH",
            "/rest/config/options",
            {
                "listenAddresses": [f"tcp://127.0.0.1:{self.listen}"],
                "globalAnnounceEnabled": False,
                "localAnnounceEnabled": False,
                "relaysEnabled": False,
                "natEnabled": False,
                "urAccepted": -1,
                "autoUpgradeIntervalH": 0,
                "crashReportingEnabled": False,
                "startBrowser": False,
            },
        )
        self.api(
            "PUT",
            f"/rest/config/devices/{peer.id}",
            {
                "deviceID": peer.id,
                "name": peer.name,
                "addresses": [f"tcp://127.0.0.1:{peer.listen}"],
                "autoAcceptFolders": False,
            },
        )
        with contextlib.suppress(Exception):
            self.api("DELETE", "/rest/config/folders/default")
        cfg = {
            "id": "mnemos",
            "label": "mnemos",
            "path": str(folder),
            "type": kind,
            "devices": [{"deviceID": self.id}, {"deviceID": peer.id}],
            "rescanIntervalS": 3600,
            "fsWatcherEnabled": True,
            "fsWatcherDelayS": 10,
        }
        if kind == "receiveonly":
            cfg["versioning"] = {
                "type": "staggered",
                "params": {"maxAge": "31536000"},
                "cleanupIntervalS": 60,
            }
        self.api("PUT", "/rest/config/folders/mnemos", cfg)

    def stop(self) -> None:
        if self.proc is not None:
            with contextlib.suppress(Exception):
                self.api("POST", "/rest/system/shutdown")
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def retry(fn, attempts: int = 200, wait: float = 0.05):
    """Call ``fn`` until it stops failing with a sharing violation (about ten seconds at most)."""
    for i in range(attempts):
        try:
            return fn()
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(wait)
    return None


def region_digests(raw: bytes, slot_len: int, header: int = 64) -> list[bytes]:
    rl = CAP * slot_len
    return [hashlib.sha1(raw[header + r * rl : header + (r + 1) * rl]).digest() for r in range(B)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0, help="duration of the workload")
    ap.add_argument("--every", type=float, default=5.0, help="seconds between agent writes")
    ap.add_argument("--settle", type=float, default=90.0, help="seconds to wait for the last sync")
    ap.add_argument(
        "--out", type=Path, default=None, help="result file (a smoke test writes elsewhere)"
    )
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the working directory for inspection")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
        return
    if args.out is None:
        args.out = RES / "e33" / f"every{int(args.every)}s.json"

    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ][:4000]
    vecs = embed_cached([r["text"] for r in rows])  # the cached slice E8 and E19 use
    cents = lab.kmeans_centroids(vecs, B, seed=0)
    rng = np.random.default_rng(33_000)

    tmp = Path(tempfile.mkdtemp(prefix="e33-"))
    a = Node("a", tmp / "home-a")
    b = Node("b", tmp / "home-b")
    fa, fb = tmp / "laptop", tmp / "backup"
    fa.mkdir()
    fb.mkdir()
    t0 = time.time()
    try:
        a.start(tmp / "a.log")
        b.start(tmp / "b.log")
        a.configure(b, fa, "sendonly")
        b.configure(a, fb, "receiveonly")
        # Wait for the two devices to connect before writing anything.
        deadline = time.time() + 120
        while time.time() < deadline:
            conns = a.api("GET", "/rest/system/connections")["connections"]
            if conns.get(b.id, {}).get("connected"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("the two syncthing instances never connected")

        over = lab.base_overrides(
            dim=DIM, buckets=B, capacity=CAP, block=BLOCK, epoch_writes=W, probe_spread=1
        )
        eng, _, _ = lab.create("p", fa / "store", cents, over, {})
        slot_len = int(eng.stat().get("slot_len", 0))
        part = fa / "store" / "partitions.bin"

        flush_hash: dict[str, int] = {hashlib.sha1(part.read_bytes()).hexdigest(): 0}
        records: list[tuple[int, int]] = []  # (flush index, region)
        flushes = 0
        end = time.time() + 60 * args.minutes
        nxt = time.time()
        while time.time() < end:
            time.sleep(max(0.0, nxt - time.time()))
            nxt += args.every
            v = vecs[int(rng.integers(0, len(vecs)))].tolist()
            try:
                reg = int(eng.ground_truth_region(v))
                eng.put("r", v, "{}")
            except Exception:
                continue
            # Syncthing opens the file without write sharing while it hashes it, and on Windows the
            # flush then fails with "Access is denied". The application would retry; so do we.
            retry(eng.flush)
            flushes += 1
            records.append((flushes, reg))
            flush_hash[hashlib.sha1(retry(part.read_bytes)).hexdigest()] = flushes
        with contextlib.suppress(Exception):
            eng.close()
        # Mnemos-P re-encrypts the whole file at lock; that final state is one more version.
        flush_hash[hashlib.sha1(part.read_bytes()).hexdigest()] = flushes + 1
        # Let the last change sync, then wait for one cleanup pass.
        time.sleep(args.settle)

        vdir = fb / ".stversions" / "store"
        retained = []  # (flush index, region digests), from archived versions and the live copy
        unmatched = 0
        files = sorted(vdir.glob("partitions~*.bin")) if vdir.exists() else []
        live = fb / "store" / "partitions.bin"
        for f in [*files, *([live] if live.exists() else [])]:
            raw = f.read_bytes()
            idx = flush_hash.get(hashlib.sha1(raw).hexdigest())
            if idx is None:
                unmatched += 1
                continue
            retained.append((idx, region_digests(raw, slot_len)))
        retained.sort(key=lambda x: x[0])
        idxs = [i for i, _ in retained]

        prec, exposed, covered = [], 0, 0
        for f_idx, reg in records:
            # The two retained versions that bracket this record's flush.
            before = [k for k, i in enumerate(idxs) if i < f_idx]
            after = [k for k, i in enumerate(idxs) if i >= f_idx]
            if not before or not after:
                continue
            x, y = retained[before[-1]][1], retained[after[0]][1]
            cand = {r for r in range(B) if x[r] != y[r]} or set(range(B))
            covered += 1
            exposed += reg in cand
            prec.append(1.0 / len(cand))
        gaps = np.diff(idxs) if len(idxs) > 1 else np.array([])
        m, lo, hi = lab.boot_ci(prec) if prec else (float("nan"),) * 3
        out = {
            "experiment": "E33",
            "syncthing": subprocess.run([str(ST), "--version"], capture_output=True)
            .stdout.decode()
            .split()[1],
            "versioning": "staggered, cleanup every 60 s",
            "fs_watcher_delay_s": 10,
            "minutes": args.minutes,
            "write_every_s": args.every,
            "pace": "agent-paced" if args.every >= 20 else "fast",
            "flushes": flushes,
            "versions_retained": len(retained),
            "versions_unmatched": unmatched,
            "mean_flushes_per_version": float(np.mean(gaps)) if len(gaps) else None,
            "median_flushes_per_version": float(np.median(gaps)) if len(gaps) else None,
            "records_covered": covered,
            "precision": m,
            "lo": lo,
            "hi": hi,
            "null_precision": 1.0 / B,
            "exposure": exposed / max(covered, 1),
            "seconds": time.time() - t0,
        }
        write_json(args.out, out)
        print(json.dumps(out, indent=1), flush=True)
    finally:
        a.stop()
        b.stop()
        if args.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def aggregate() -> None:
    runs = {}
    for f in sorted((RES / "e33").glob("every*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        runs[d["pace"]] = d
    write_json(OUT, {"experiment": "E33", "runs": runs})
    print(f"wrote {OUT.relative_to(C.repo_root())} with {sorted(runs)}")


if __name__ == "__main__":
    main()
