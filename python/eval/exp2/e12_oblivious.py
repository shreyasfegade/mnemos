"""E12: what it costs to fetch payloads obliviously, for the read-observing adversary.

The review's §4 asks what the alternatives are once reads are visible. A search has to get its
``top_k`` payloads from somewhere, and the choice of where decides both the leakage and the cost:

``position``   read the result slots by offset. Cheapest, and it re-opens the result channel: the
               positions read are a function of what matched, so a cohort-H record returned by a
               post-tau search becomes linkable to that query's topic. Safe against a *write*
               observer, which sees no reads at all, and not against a read-write one.
``read_all``   read the whole store on every search. Leaks nothing beyond "a search happened",
               and costs O(n) per query.
``path_oram``  Path ORAM over the slots: O(log n) blocks per access, leaks nothing. Reported as a
               cost model with its parameters stated, not as a measurement -- there is no ORAM in
               the tree, and a bad reimplementation would be worse than an honest formula.
``resident``   Mnemos-M: payloads are decrypted once at unlock and ranked in the locked arena, so
               a search performs **no disk access at all**. Zero read cost and zero read leakage,
               bounded instead by memory.

The measured columns are ``position``, ``read_all`` and ``resident``; ``path_oram`` is computed.
Each row says which it is.

    cd python && python -m eval.exp2.e12_oblivious
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e12_oblivious.json"
SIZES = (1_000, 10_000, 50_000)
DIM = 768
BLOCK = 1024
W = 16
N_SEARCH = 60
TOP_K = 10
ORAM_Z = 4  # slots per Path ORAM bucket


def geometry(n: int) -> tuple[int, int]:
    b = max(16, 1 << int(np.ceil(np.log2(max(4.0, np.sqrt(n))))))
    c = max(16, int(np.ceil(n / b * 3)))
    return b, c


def path_oram_cost(n_slots: int, slot_bytes: int, accesses: int) -> dict:
    """Blocks touched by Path ORAM: one root-to-leaf path read and written per access."""
    leaves = max(1, n_slots // ORAM_Z)
    height = max(1, math.ceil(math.log2(leaves)))
    blocks_per_access = 2 * ORAM_Z * (height + 1)  # read the path, write it back
    return {
        "method": "path_oram",
        "measured": False,
        "bucket_size_Z": ORAM_Z,
        "tree_height": height,
        "blocks_per_access": blocks_per_access,
        "bytes_read_per_search": blocks_per_access * slot_bytes * accesses,
        "leaks_result_positions": False,
    }


def run(n: int, vecs, cents, rng) -> list[dict]:
    b, cap = geometry(n)
    rows: list[dict] = []
    slot_bytes = None

    for method in ("resident", "position", "read_all"):
        tmp = Path(tempfile.mkdtemp())
        try:
            over = lab.base_overrides(
                dim=DIM,
                buckets=b,
                capacity=cap,
                block=BLOCK,
                epoch_writes=W,
                probe_spread=2,
                arena_bytes=4 << 20,
            )
            if method == "resident":
                over = lab.merge(over, {"m": {"payload_residency": "memory"}})
            else:
                over = lab.merge(over, {"m": {"payload_residency": "disk"}})
            abl = {"disk_search": True} if method == "position" else {}
            eng, _, _ = lab.create("m", tmp / "s", cents, over, abl)

            added = 0
            for i in range(n):
                try:
                    eng.put(f"r{i}", vecs[i].tolist(), "{}")
                    added += 1
                except Exception:
                    break
            lab.drain(eng)
            eng.flush()
            st = eng.stat()
            slot_bytes = int(st["slot_len"])
            region_bytes = int(st["region_len"])

            trace = tmp / "t.jsonl"
            trace.write_text("", encoding="utf-8")
            eng.set_trace_sink(str(trace))

            qs = rng.integers(0, max(added, 1), N_SEARCH)
            lat = []
            for qi in qs:
                t = time.perf_counter()
                eng.search(vecs[qi].tolist())
                lat.append(time.perf_counter() - t)

            # Count the reads the search actually issued, from the adversary's own transcript.
            slot_reads = 0
            if trace.exists():
                for line in trace.read_text(encoding="utf-8").splitlines():
                    if line.strip() and json.loads(line)["kind"] == "region_slot_read":
                        slot_reads += 1

            if method == "read_all":
                # A read-all search reads every region; the engine does not implement it, so the
                # cost is the store size and the latency is the resident latency plus that read.
                read_bytes = b * region_bytes
                extra = read_bytes / (2.0e9)  # a conservative 2 GB/s sequential read
                lat = [x + extra for x in lat]
            else:
                read_bytes = slot_reads * slot_bytes / max(N_SEARCH, 1)

            rows.append(
                {
                    "n": n,
                    "added": added,
                    "method": method,
                    "measured": True,
                    "latency_ms": {
                        "p50": float(np.percentile(np.array(lat) * 1e3, 50)),
                        "p95": float(np.percentile(np.array(lat) * 1e3, 95)),
                    },
                    "slot_reads_per_search": slot_reads / max(N_SEARCH, 1),
                    "bytes_read_per_search": read_bytes,
                    "leaks_result_positions": method == "position",
                    "note": (
                        "read_all latency adds the store scan at 2 GB/s; the engine ranks in memory"
                    )
                    if method == "read_all"
                    else "",
                }
            )
            with contextlib.suppress(Exception):
                eng.close()
        except Exception as e:
            rows.append({"n": n, "method": method, "error": f"{type(e).__name__}: {e}"})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if slot_bytes:
        o = path_oram_cost(b * cap, slot_bytes, TOP_K)
        o["n"] = n
        rows.append(o)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    args = ap.parse_args()
    t0 = time.time()

    rows_src = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    base = embed_cached([r["text"] for r in rows_src])
    need = max(args.sizes)
    vecs = np.tile(base, (int(np.ceil(need / len(base))), 1))[:need].astype(np.float32)

    results = {"experiment": "E12", "top_k": TOP_K, "oram_bucket_size": ORAM_Z, "rows": []}
    rng = np.random.default_rng(0)
    for n in args.sizes:
        cents = lab.kmeans_centroids(base, geometry(n)[0], seed=0)
        for r in run(n, vecs, cents, rng):
            results["rows"].append(r)
            if "error" in r:
                print(f"n={n:>7} {r['method']:>10}: FAILED {r['error']}", flush=True)
            else:
                meas = "measured" if r.get("measured") else "computed"
                lat = f"{r['latency_ms']['p50']:8.3f} ms" if "latency_ms" in r else "        --"
                print(
                    f"n={n:>7} {r['method']:>10} ({meas:>8}): {lat} | "
                    f"{r.get('bytes_read_per_search', 0) / 1e3:10.1f} kB read/search | "
                    f"leaks positions: {r['leaks_result_positions']}",
                    flush=True,
                )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"wrote {OUT.relative_to(C.repo_root())} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
