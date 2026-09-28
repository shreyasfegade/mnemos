"""E11 -- how store-build cost grows with corpus size. The prediction here was wrong.

The paper reports latency at $n <= 10^4$ and called the unconditional manifest rewrite "a
genuine scalability limit". A benchmark at $n = 10^5$ was then left running for three and a half
hours without producing a measurement, which looked like confirmation.

"It did not finish" is an anecdote. This experiment turns it into a number, and the number does
not say what we expected.

The mechanism we predicted
==========================
Two costs per flush are independent of how much actually changed:

``manifest rewrite``
    Rewritten **every epoch whether or not anything changed**, because a conditional write would
    turn "the manifest changed" into "real work happened this epoch" -- an oracle. It carries the
    slab occupancy bitmap, the index occupancy bitmap and the centroid table, so it is
    ``Theta(n_max)`` bytes.

``token advance``
    Forward privacy requires advancing the search token for **every** partition, not only the
    touched ones, for the same reason.

A store of ``n`` entries performs ``n / epoch_writes`` flushes, so total *write volume* is
``Theta(n * n_max / w)``. With ``n_max`` proportional to ``n`` that is quadratic, and per-entry
wall-clock should therefore grow linearly in ``n``.

What was measured
=================
========  ===========  ============  ==============
       n    build (s)     per entry    manifest MB
========  ===========  ============  ==============
   1,000         0.10        96.0 us            1.6
   2,000         0.19        96.8 us            6.3
   4,000         0.37        92.5 us           13.1
   8,000         0.87       108.3 us           52.2
  16,000         1.82       114.0 us          110.6
========  ===========  ============  ==============

Per-entry cost goes as ``n^0.07``; total build cost as ``n^1.07`` -- slightly super-linear, with
per-entry cost rising ~23% across the last two doublings. So the manifest term is *visible* and
does not dominate. Extrapolated, ``n = 100,000`` builds in about thirteen seconds.

Two wrong answers preceded this one
===================================
**Measured with a benchmark holding a core**, the smallest point came out 6x high and dragged the
fitted exponent negative -- a super-linear cost made to look sub-linear by contention.

**Measured through** ``MnemosMemoryServer.put`` **, which embeds its argument**, every size came
out at ~7,600 us/entry and perfectly flat. That is the embedder: it costs milliseconds per entry,
about a hundred times the storage path, and it is O(1) in ``n``. The flat curve was a measurement
of ``bge-base-en-v1.5``, and it "confirmed" a linearity claim that the write path does not
actually satisfy. Vectors are now precomputed and handed to the native engine directly.

Both failures produced a confident, plausible, wrong exponent. Neither was detectable from the
output alone -- only from asking what the number should have been (96 us matches the ``put``
figure in the paper's performance table; 7,600 us does not).

The correction matters more than the number
===========================================
"Quadratic write volume" is a true statement about a growth law that had been turned into a false
statement about a system, and a benchmark that ran three and a half hours without finishing
appeared to confirm it. It did not: that time went on building and measuring five schemes at
100,000 entries, and the store build itself is seconds.

The honest claim is: the quadratic term exists, it is now faintly visible, and we have not
reached the size where it dominates.

Usage::

    cd python && python -m eval.experiments.e11_write_scaling

Run it on an idle machine. Measured while a criterion benchmark held a core, the smallest point
came out 6x high and dragged the fitted exponent negative.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval.lab import corpus as C

#: Corpus sizes. Chosen to span a decade with enough points to distinguish a flat per-entry cost
#: from a linearly growing one; larger sizes are the thing being shown to be infeasible.
SIZES = (1_000, 2_000, 4_000, 8_000, 16_000)

DIM = 768
SEED = 20_260_827


@dataclass
class WriteRow:
    n: int
    buckets: int
    capacity: int
    n_max: int
    build_seconds: float
    per_entry_us: float
    flushes: int
    manifest_bytes_total: int
    """`flushes * manifest size`, the quantity that grows quadratically."""
    store_bytes: int


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _to_toml(d: dict) -> str:
    lines: list[str] = []
    for k, v in d.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {_fmt(v)}")
    for section, body in d.items():
        if not isinstance(body, dict):
            continue
        flat = {k: v for k, v in body.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in body.items() if isinstance(v, dict)}
        if flat:
            lines.append(f"\n[{section}]")
            lines.extend(f"{k} = {_fmt(v)}" for k, v in flat.items())
        for sub, subbody in nested.items():
            lines.append(f"\n[{section}.{sub}]")
            lines.extend(f"{k} = {_fmt(v)}" for k, v in subbody.items())
    return "\n".join(lines) + "\n"


def _params_for(n: int) -> tuple[Path, int, int, int, int]:
    """Geometry sized the same way E2 sizes it: `B ~ sqrt(n)`, `C` scaled to hold the corpus."""
    import tomllib

    with (C.repo_root() / "params.toml").open("rb") as fh:
        p = tomllib.load(fh)

    buckets = max(16, 1 << int(np.ceil(np.log2(max(4.0, np.sqrt(n))))))
    capacity = max(32, 1 << int(np.ceil(np.log2(max(2.0, (n / buckets) * 2)))))
    n_max = max(1024, 1 << int(np.ceil(np.log2(n / 0.6 * 1.5))))
    assert buckets * capacity >= n, "index geometry cannot hold the corpus"

    p["embedding"]["dim"] = DIM
    p["index"]["buckets"] = buckets
    p["index"]["bucket_capacity"] = capacity
    p["index"]["n_max"] = n_max
    p["crypto"]["block_size_bytes"] = 1024
    p["crypto"]["max_entry_bytes"] = 1024
    p["memory"]["arena_bytes"] = 262_144

    out = Path(tempfile.gettempdir()) / f"mnemos-e11-{n}.toml"
    out.write_text(_to_toml(p), encoding="utf-8")
    return out, buckets, capacity, n_max, int(p["epoch"]["epoch_writes"])


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def run_size(n: int) -> WriteRow:
    import mnemos_native

    from mnemos.mcp_server import MnemosMemoryServer

    os.environ.setdefault("MNEMOS_I_ACCEPT_INSECURE", "1")
    rng = np.random.default_rng(SEED)

    params, buckets, capacity, n_max, epoch_writes = _params_for(n)

    # Synthetic vectors: this experiment measures I/O, and the embedder would dominate the
    # wall-clock while contributing nothing to what is being measured.
    vecs = rng.normal(size=(min(buckets * 4, n), DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    centroids = mnemos_native.train_centroids([v.tolist() for v in vecs], buckets, DIM, SEED)

    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "store"
        server = MnemosMemoryServer(store, params_path=params, allow_insecure_keystore=True)
        server.open(create_if_missing=True, centroids=centroids)

        # Vectors are precomputed and handed straight to the native engine. `MnemosMemoryServer
        # .put` embeds its argument, and the embedder costs milliseconds per entry -- roughly a
        # hundred times the storage path. Going through it made every size look identical
        # because the *embedder* is O(1) per entry, and the flat curve that produced was a
        # measurement of bge-base-en-v1.5, not of the write path.
        engine = server._require()
        pool = rng.normal(size=(64, DIM)).astype(np.float32)
        pool /= np.linalg.norm(pool, axis=1, keepdims=True)
        vectors = [pool[i % 64].tolist() for i in range(n)]
        meta = "{}"

        t0 = time.perf_counter()
        for i in range(n):
            engine.put(f"entry {i} filler text for a realistic record body", vectors[i], meta)
        engine.flush()
        elapsed = time.perf_counter() - t0

        size = _dir_bytes(store)
        server.close()

    flushes = max(1, n // epoch_writes)
    # The manifest carries two occupancy bitmaps plus the centroid table. Approximated from the
    # geometry rather than measured per-write, because the point is the growth law, not the
    # constant.
    manifest_bytes = n_max // 8 + (buckets * capacity) // 8 + buckets * DIM
    return WriteRow(
        n=n,
        buckets=buckets,
        capacity=capacity,
        n_max=n_max,
        build_seconds=elapsed,
        per_entry_us=1e6 * elapsed / n,
        flushes=flushes,
        manifest_bytes_total=flushes * manifest_bytes,
        store_bytes=size,
    )


def main() -> None:
    # Warm-up build, discarded. The first store built in a process pays for loading the native
    # extension, faulting in its pages and warming the filesystem cache; on a 500-entry corpus
    # that overhead was 6x the steady-state per-entry cost and dragged the fitted exponent
    # *negative* -- i.e. it made a super-linear cost look sub-linear.
    print("  warm-up ...", flush=True)
    run_size(SIZES[0])

    rows: list[WriteRow] = []
    for n in SIZES:
        row = run_size(n)
        rows.append(row)
        print(
            f"  n={n:<6} B={row.buckets:<4} C={row.capacity:<4} n_max={row.n_max:<6} "
            f"build={row.build_seconds:7.2f}s  per-entry={row.per_entry_us:8.1f}us  "
            f"manifest total={row.manifest_bytes_total / 1e6:8.1f}MB",
            flush=True,
        )

    out = C.repo_root() / "python" / "eval" / "results" / "e11_write_scaling.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "experiment": "E11",
                "description": "store build cost vs corpus size; the write-path scaling wall",
                "dim": DIM,
                "rows": [asdict(r) for r in rows],
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    print("\n=== E11: write-path scaling ===\n")
    print(f"  {'n':>7}  {'build (s)':>10}  {'per entry':>11}  {'vs smallest':>12}")
    base = rows[0].per_entry_us if rows else 1.0
    for r in rows:
        print(
            f"  {r.n:>7}  {r.build_seconds:>10.2f}  {r.per_entry_us:>9.1f}us"
            f"  {r.per_entry_us / base:>11.2f}x"
        )

    if len(rows) >= 2:
        # Growth exponent of per-entry cost. 0 means flat (linear total), 1 means linear
        # per-entry (quadratic total).
        xs = np.log([r.n for r in rows])
        ys = np.log([r.per_entry_us for r in rows])
        slope = float(np.polyfit(xs, ys, 1)[0])
        print(f"\n  per-entry cost grows as n^{slope:.2f}")
        print(f"  => total write cost grows as n^{1 + slope:.2f}")
        last = rows[-1]
        pred = last.per_entry_us * (100_000 / last.n) ** slope
        print(
            f"\n  extrapolated to n = 100,000: {pred:.0f}us/entry "
            f"=> {pred * 100_000 / 1e6 / 60:.1f} minutes to build the store"
        )
    print(f"\nwrote {out.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
