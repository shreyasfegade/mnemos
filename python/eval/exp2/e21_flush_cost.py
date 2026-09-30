"""E21: what Mnemos-M's clocked flush costs on disk, per flush and per day.

The first submission never stated Delta. It was 500 ms, and a flush rewrote ``w + 1`` whole
regions whether or not anything was written, which at the shipped geometry is about 5.4 MB per
flush -- close to a terabyte a day for a store left unlocked. The revision replaces that flush with
a sweep (``m.flush_mode = "sweep"``): the next ``sweep_slots`` slots after a cursor, a fixed-size
sealed journal, and the manifest. This experiment measures both, on the real engine, at the shipped
geometry and at a 131k-slot one, and reports:

* bytes written per flush, with nothing staged and with one record staged (they must be equal);
* wall time per durable flush;
* bytes per day as a function of Delta, for a store unlocked all day and for eight hours;
* what a block-level sync client would upload per flush, at Syncthing's 128 KiB and Dropbox's
  4 MiB block sizes, computed from the byte ranges the flush actually wrote;
* the cost of Mnemos-P's full re-encryption at lock, which is the whole partition file;
* years to exhaust a drive rated at 600 TBW;
* the sweep's one residual: how often, at a given write rate and load, more records are staged at
  lock than the journal holds, which is when locking needs extra flushes.

    cd python && python -m eval.exp2.e21_flush_cost
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from eval.exp2 import lab
from eval.lab import corpus as C

OUT = C.repo_root() / "python" / "eval" / "results" / "e21_flush_cost.json"
DIM = 768
BLOCK = 4096
W = 16
DELTAS = (0.5, 1.0, 5.0, 10.0, 30.0, 60.0)
TBW = 600e12  # rated endurance of the 1 TB drive the measurements ran on (Micron 2500)
SYNC_BLOCKS = {"syncthing_128k": 128 * 1024, "dropbox_4m": 4 * 1024 * 1024}

GEOMETRIES = {
    "shipped": (256, 64),
    "131k": (1024, 128),
}

# The earlier shipped configuration of [m], and the revision's.
MODES = {
    "regions": {"flush_mode": "regions", "tombstone_capacity": 4096},
    "sweep": {
        "flush_mode": "sweep",
        "sweep_slots": 8,
        "journal_slots": 4,
        "tombstone_capacity": 512,
    },
}


def unit_vecs(n: int, dim: int, rng) -> np.ndarray:
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def written_ranges(trace: Path, slot_len: int, cap: int, region_len: int) -> list[tuple[int, int]]:
    """Byte ranges of ``slab.bin`` written by the events in ``trace``."""
    out = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e["kind"] != "region_write":
            continue
        r = int(e["region"])
        slots = e.get("slots") or []
        if not slots:
            out.append((64 + r * region_len, region_len))
        else:
            for s in slots:
                out.append((64 + (r * cap + int(s)) * slot_len, slot_len))
    return out


def sync_upload(ranges: list[tuple[int, int]], block: int) -> int:
    touched = set()
    for off, ln in ranges:
        touched.update(range(off // block, (off + ln - 1) // block + 1))
    return len(touched) * block


def measure(mode: str, b: int, c: int, rng) -> dict:
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.merge(
            lab.base_overrides(dim=DIM, buckets=b, capacity=c, block=BLOCK, epoch_writes=W),
            {"m": {**MODES[mode], "durable": True}},
        )
        cents = unit_vecs(b, DIM, rng)
        eng, _, _ = lab.create("m", tmp / "s", cents, over)
        st = eng.stat()
        slot_len, region_len = int(st["slot_len"]), int(st["region_len"])
        # Warm the store to a realistic occupancy, a quarter full.
        n_warm = b * c // 4
        vecs = unit_vecs(min(n_warm, 4096), DIM, rng)
        for i in range(n_warm):
            eng.put(f"warm {i}", vecs[i % len(vecs)].tolist(), "{}")
            if i % 256 == 255:
                lab.drain(eng)
        lab.drain(eng)

        def one_flush(staged: int) -> tuple[int, float, list[tuple[int, int]]]:
            for j in range(staged):
                eng.put(f"x{j}", vecs[j].tolist(), "{}")
            trace = tmp / f"t{time.monotonic_ns()}.jsonl"
            eng.set_trace_sink(str(trace))
            s0 = eng.stat()
            t0 = time.perf_counter()
            eng.flush()
            dt = time.perf_counter() - t0
            s1 = eng.stat()
            eng.set_trace_sink(str(tmp / "null.jsonl"))
            by = sum(
                int(s1[k]) - int(s0[k])
                for k in ("region_bytes_written", "manifest_bytes_written", "journal_bytes_written")
            )
            return by, dt, written_ranges(trace, slot_len, c, region_len)

        empty = [one_flush(0) for _ in range(12)]
        staged = [one_flush(1) for _ in range(12)]
        lab.drain(eng)
        bpf = int(np.median([x[0] for x in empty]))
        bpf1 = int(np.median([x[0] for x in staged]))
        small_files = bpf - int(np.median([sum(ln for _, ln in x[2]) for x in empty]))
        uploads = {
            name: int(np.median([sync_upload(x[2], blk) for x in empty])) + small_files
            for name, blk in SYNC_BLOCKS.items()
        }
        with_contextlib_close(eng)
        slab = (tmp / "s" / "slab.bin").stat().st_size
        per_day = {str(d): bpf * 86400 / d / 1e9 for d in DELTAS}
        return {
            "mode": mode,
            "buckets": b,
            "capacity": c,
            "slots": b * c,
            "slot_len": slot_len,
            "file_bytes": slab,
            "bytes_per_flush": bpf,
            "bytes_per_flush_one_staged": bpf1,
            "flush_ms_p50": float(np.median([x[1] for x in empty]) * 1e3),
            "flush_ms_p95": float(np.percentile([x[1] for x in empty + staged], 95) * 1e3),
            "gb_per_day": per_day,
            "gb_per_8h_day": {str(d): bpf * 8 * 3600 / d / 1e9 for d in DELTAS},
            "sync_upload_per_flush": uploads,
            "sync_upload_gb_per_day_at_10s": {k: v * 8640 / 1e9 for k, v in uploads.items()},
            "drive_years_at_600tbw": {str(d): TBW / (bpf * 86400 / d * 365) for d in DELTAS},
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def with_contextlib_close(eng) -> None:
    with contextlib.suppress(Exception):
        eng.close()


def p_lock_cost(b: int, c: int, rng) -> dict:
    """Mnemos-P re-encrypts its whole partition file at lock: the bytes are the file."""
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.base_overrides(dim=DIM, buckets=b, capacity=c, block=BLOCK, epoch_writes=W)
        eng, _, _ = lab.create("p", tmp / "s", unit_vecs(b, DIM, rng), over)
        size = sum(f.stat().st_size for f in (tmp / "s").rglob("*") if f.is_file())
        t0 = time.perf_counter()
        with_contextlib_close(eng)
        return {
            "buckets": b,
            "capacity": c,
            "store_bytes": size,
            "lock_seconds": time.perf_counter() - t0,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def lock_residual(rate: float, load: float, rng, sessions: int = 40, length: int = 200) -> dict:
    """How often a session ends with more staged records than the journal holds.

    The sweep places a staged record only in a free slot the cursor passes, so it places at most
    ``sweep_slots * (1 - load)`` records per flush -- its *placement capacity*. A write rate below
    that is sustained with a backlog that stays at or below ``journal_slots``; a rate above it makes
    the backlog grow without bound, and locking then needs extra flushes to drain it. A real agent
    writes seconds to minutes apart, so at Delta = 10 s its rate is a small fraction of a write per
    flush, far below the placement capacity at any reasonable load.

    The store is small (512 slots) on purpose: a full cursor sweep takes 64 flushes, well inside a
    200-flush session, so a deleted record's slot is purged and reused within the session rather
    than lingering as a tombstone until a much later sweep -- the earlier harness used a store so
    large that no session ever swept it, which drove occupancy to the global bound and measured
    capacity saturation instead of the journal backlog.
    """
    tmp = Path(tempfile.mkdtemp())
    b, c, dim = 16, 32, 32
    sweep_slots = int(MODES["sweep"]["sweep_slots"])
    journal = int(MODES["sweep"]["journal_slots"])
    placement_capacity = sweep_slots * (1.0 - load)
    try:
        over = lab.merge(
            lab.base_overrides(dim=dim, buckets=b, capacity=c, block=256, epoch_writes=W),
            {"m": MODES["sweep"]},
        )
        eng, _, _ = lab.create("m", tmp / "s", unit_vecs(b, dim, rng), over)
        vecs = unit_vecs(256, dim, rng)
        live: list[str] = []
        for i in range(int(load * b * c)):
            with contextlib.suppress(Exception):
                live.append(eng.put(f"w{i}", vecs[i % 256].tolist(), "{}"))
        lab.drain(eng)
        residual = 0
        backlog_max = 0
        backlog_at_lock = []
        for _ in range(sessions):
            for _t in range(length):
                for _k in range(int(rng.poisson(rate))):
                    # Delete first, so churn keeps occupancy flat rather than climbing.
                    if live:
                        with contextlib.suppress(Exception):
                            eng.delete(live.pop(int(rng.integers(0, len(live)))))
                    with contextlib.suppress(Exception):
                        live.append(eng.put("n", vecs[int(rng.integers(0, 256))].tolist(), "{}"))
                eng.flush()
                backlog_max = max(backlog_max, int(eng.stat()["staged"]))
            b_lock = int(eng.stat()["staged"])
            backlog_at_lock.append(b_lock)
            if b_lock > journal:
                residual += 1
        with_contextlib_close(eng)
        return {
            "rate": rate,
            "load": load,
            "placement_capacity_per_flush": placement_capacity,
            "sustainable": rate < placement_capacity,
            "sessions": sessions,
            "flushes_per_session": length,
            "residual_sessions": residual,
            "residual_rate": residual / sessions,
            "max_backlog": backlog_max,
            "mean_backlog_at_lock": float(np.mean(backlog_at_lock)),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    rng = np.random.default_rng(20260928)
    t0 = time.time()
    res: dict = {
        "experiment": "E21",
        "tbw_assumed": TBW,
        "deltas_s": list(DELTAS),
        "geometries": {},
    }
    for name, (b, c) in GEOMETRIES.items():
        res["geometries"][name] = {m: measure(m, b, c, rng) for m in MODES}
        res["geometries"][name]["p_lock"] = p_lock_cost(b, c, rng)
        g = res["geometries"][name]
        print(
            f"{name}: regions {g['regions']['bytes_per_flush'] / 2**20:.2f} MiB/flush, "
            f"sweep {g['sweep']['bytes_per_flush'] / 1024:.1f} KiB/flush "
            f"(one staged: {g['sweep']['bytes_per_flush_one_staged'] / 1024:.1f}); "
            f"P lock rewrites {g['p_lock']['store_bytes'] / 2**20:.0f} MiB",
            flush=True,
        )
    res["lock_residual"] = [
        lock_residual(rate, load, rng) for load in (0.5, 0.9) for rate in (0.1, 0.5, 1.0, 2.0)
    ]
    for r in res["lock_residual"]:
        print(
            f"lock residual: rate {r['rate']}/flush, load {r['load']}: "
            f"{r['residual_rate']:.3f} (max backlog {r['max_backlog']})",
            flush=True,
        )
    res["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
