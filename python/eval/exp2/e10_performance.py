"""E10: what the three designs cost, at five sizes, against a Rust plaintext baseline.

Three things were wrong with the previous performance run and all three
are fixed here.

1. **The baseline was numpy**, and its ``add`` rebuilt the whole array, so the plaintext write path
   was *slower* than the encrypted one at 10^4 records and the ratio meant nothing. The baseline is
   now ``mnemos-baselines::plain::PlaintextIvf``, the in-tree Rust index that shares ``QuantVec``
   and the same partitioner with the encrypted engines, reached through the same FFI. A ratio
   against it is a statement about the security machinery, not about two libraries.
2. **Two sizes is not a curve.** Five sizes, so the shape is visible.
3. **The sparse regime was never timed.** An agent writes seconds apart, so nearly every flush
   carries one record and pays for ``w + 1`` whole-region rewrites on its own. That per-put cost is
   the one a deployment actually sees, and it is reported beside the amortised one.

It also measures what decides between Mnemos-M and Mnemos-P: resident memory, unlock time, and
bytes written per memory. That is the crossover.

    cd python && python -m eval.exp2.e10_performance
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import device, embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e10_performance.json"
SIZES = (1_000, 3_000, 10_000, 30_000, 100_000)
DIM = 768
BLOCK = 1024
W = 16
N_SEARCH = 150
N_SPARSE_PUTS = 40


def geometry(n: int) -> tuple[int, int]:
    b = max(16, 1 << int(np.ceil(np.log2(max(4.0, np.sqrt(n))))))
    c = max(16, int(np.ceil(n / b * 3)))
    return b, c


def pct(xs: list[float]) -> dict:
    a = np.array(xs) * 1e3  # milliseconds
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
    }


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# Named variants of an engine: the engine a variant runs on, and the parameter overrides it adds.
#
# ``m-ivf`` is Mnemos-M with its in-memory partition index (params ``m.search = "ivf"``). Its disk
# layout is Mnemos-M's -- uniformly random -- and its probing happens in the locked arena, so it
# has Mnemos-M's leakage; the question E10 answers is whether it removes the crossover in search
# latency that exhaustive in-memory ranking produces. ``probe_width`` 8 matches the number of
# regions Mnemos-P reads per search at k_probe = 4 and s = 2, so the two do comparable work.
VARIANTS: dict[str, tuple[str, dict]] = {
    "m-ivf": ("m", {"m": {"search": "ivf", "probe_width": 8}}),
}


def bench_engine(kind: str, n: int, vecs: np.ndarray, cents: np.ndarray, rng) -> dict:
    base_kind, variant_over = VARIANTS.get(kind, (kind, {}))
    b, cap = geometry(n)
    tmp = Path(tempfile.mkdtemp())
    row: dict = {"engine": kind, "n": n, "buckets": b, "capacity": cap}
    try:
        if kind == "plaintext":
            import mnemos_native as native

            ivf = native.PlaintextIvf([list(map(float, c)) for c in cents], 4, 10)
            t0 = time.perf_counter()
            for i in range(n):
                ivf.put(f"r{i}", vecs[i].tolist(), "{}")
            row["bulk_load_s"] = time.perf_counter() - t0
            row["added"] = n
            qs = rng.integers(0, n, N_SEARCH)
            lat = []
            for qi in qs:
                t = time.perf_counter()
                ivf.search(vecs[qi].tolist())
                lat.append(time.perf_counter() - t)
            row["search_ms"] = pct(lat)
            lat = []
            for j in range(N_SPARSE_PUTS):
                t = time.perf_counter()
                ivf.put(f"w{j}", vecs[j % n].tolist(), "{}")
                lat.append(time.perf_counter() - t)
            row["put_sparse_ms"] = pct(lat)
            row["put_amortised_ms"] = row["put_sparse_ms"]
            row["resident_bytes"] = 0
            row["disk_bytes"] = 0
            row["unlock_s"] = 0.0
            # The in-tree plaintext IVF keeps nothing on disk: bytes written is not applicable,
            # and a 0 beside the encrypted designs' write costs would read as a measurement.
            row["bytes_per_memory"] = None
            return row

        over = lab.base_overrides(
            dim=DIM,
            buckets=b,
            capacity=cap,
            block=BLOCK,
            epoch_writes=W,
            probe_spread=2,
            arena_bytes=4 << 20,
        )
        if kind == "rs":
            over = lab.merge(over, {"index": {"n_max": b * cap, "overflow_depth": 2}})
        over = lab.merge(over, variant_over)
        t_create = time.perf_counter()
        eng, _, _ = lab.create(base_kind, tmp / "s", cents, over, {})
        row["create_s"] = time.perf_counter() - t_create

        t0 = time.perf_counter()
        added = 0
        for i in range(n):
            try:
                eng.put(f"r{i}", vecs[i].tolist(), "{}")
                added += 1
            except Exception:
                break
        lab.drain(eng)
        eng.flush()
        row["bulk_load_s"] = time.perf_counter() - t0
        row["added"] = added
        st1 = eng.stat()
        row["resident_bytes"] = int(st1.get("resident_bytes", 0))
        row["disk_bytes"] = dir_bytes(tmp / "s")
        # Only engines that count their own bytes report this. Mnemos-RS does not, and reading a
        # missing counter as 0 once put "0.0 kB written per memory" in the table for the one design
        # that writes on every put: an absent measurement must stay absent.
        if "region_bytes_written" in st1:
            written = int(st1["region_bytes_written"]) + int(st1.get("manifest_bytes_written", 0))
            row["bytes_per_memory"] = written / max(added, 1)
        else:
            row["bytes_per_memory"] = None

        qs = rng.integers(0, max(added, 1), N_SEARCH)
        lat = []
        for qi in qs:
            t = time.perf_counter()
            eng.search(vecs[qi].tolist())
            lat.append(time.perf_counter() - t)
        row["search_ms"] = pct(lat)

        # The sparse regime: one record, one flush. This is what an agent writing seconds apart
        # actually pays, and it is the number a deployment feels.
        lat = []
        for j in range(N_SPARSE_PUTS):
            t = time.perf_counter()
            with contextlib.suppress(Exception):
                eng.put(f"s{j}", vecs[j % max(added, 1)].tolist(), "{}")
            eng.flush()
            lat.append(time.perf_counter() - t)
        row["put_sparse_ms"] = pct(lat)

        # Amortised: a full batch of w puts, the flush inside the timed region.
        lat = []
        for _ in range(max(4, N_SPARSE_PUTS // W)):
            t = time.perf_counter()
            for j in range(W):
                with contextlib.suppress(Exception):
                    eng.put(f"a{j}", vecs[j % max(added, 1)].tolist(), "{}")
            lab.drain(eng)
            lat.append((time.perf_counter() - t) / W)
        row["put_amortised_ms"] = pct(lat)

        # Unlock: close, reopen, and time the scan that rebuilds whatever the design keeps.
        with contextlib.suppress(Exception):
            eng.close()
        import mnemos_native as native

        cls = {"m": native.EngineM, "p": native.EngineV2, "rs": native.Engine}[base_kind]
        pp = str((tmp / "s").parent / "s.params.toml")
        t = time.perf_counter()
        if kind == "rs":
            e2 = cls.open(str(tmp / "s"), pp, True)
        else:
            e2 = cls.open(str(tmp / "s"), pp, True, {})
        e2.unlock()
        row["unlock_s"] = time.perf_counter() - t
        with contextlib.suppress(Exception):
            e2.lock()
        return row
    except Exception as e:
        # A design that cannot run at a size is a result, not an omission: Mnemos-M's resident
        # index is the thing that eventually stops it, and where it stops is the crossover.
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--engines", nargs="+", default=["plaintext", "m", "m-ivf", "p", "rs"])
    args = ap.parse_args()
    t0 = time.time()

    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    base = embed_cached([r["text"] for r in rows])
    need = max(args.sizes)
    reps = int(np.ceil(need / len(base)))
    vecs = np.tile(base, (reps, 1))[:need].astype(np.float32)
    print(f"{len(base)} base embeddings tiled to {need}; device {device()}", flush=True)

    results = {
        "experiment": "E10",
        "device": device(),
        "dim": DIM,
        "block": BLOCK,
        "w": W,
        "rows": [],
    }
    rng = np.random.default_rng(0)
    for n in args.sizes:
        cents = lab.kmeans_centroids(base, geometry(n)[0], seed=0)
        for kind in args.engines:
            r = bench_engine(kind, n, vecs, cents, rng)
            results["rows"].append(r)
            if "error" in r:
                print(f"n={n:>7} {kind:>9}: FAILED {r['error']}", flush=True)
            else:
                print(
                    f"n={n:>7} {kind:>9}: search p50 {r['search_ms']['p50']:7.3f} ms | "
                    f"put(sparse) p50 {r['put_sparse_ms']['p50']:8.3f} ms | "
                    f"RAM {r['resident_bytes'] / 1e6:7.1f} MB | disk {r['disk_bytes'] / 1e6:7.1f} "
                    f"MB | "
                    f"unlock {r['unlock_s']:6.2f} s | "
                    + (
                        "     n/a kB/memory"
                        if r["bytes_per_memory"] is None
                        else f"{r['bytes_per_memory'] / 1e3:8.1f} kB/memory"
                    ),
                    flush=True,
                )
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"wrote {OUT.relative_to(C.repo_root())} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
