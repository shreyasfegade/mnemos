"""Shared construction for the three designs, so an experiment names a design and not a shape.

The paper evaluates a design space rather than one construction, which means every experiment
has to be able to build all three engines at *one* geometry and drive them through one interface:

``rs``  Mnemos-RS -- the two-file record-store design (the crate is ``mnemos-engine``).
``p``   Mnemos-P  -- the partitioned single sealed file (the crate is ``mnemos-v2``).
``m``   Mnemos-M  -- the memory-resident design (the crate is ``mnemos-mem``).

The engine names in the crate tree are historical; the names here are the ones the paper uses.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from eval.lab.stores import write_params

# Mnemos-M and Mnemos-P share their geometry keys, so a caller states the shape once.
_SHARED = ("embedding", "index", "crypto", "epoch", "memory")


def base_overrides(
    *,
    dim: int,
    buckets: int,
    capacity: int,
    top_k: int = 10,
    probe_width: int = 4,
    probe_spread: int = 2,
    block: int = 1024,
    epoch_writes: int = 16,
    epoch_ms: int = 3_600_000,
    arena_bytes: int = 1 << 20,
) -> dict:
    """One geometry, expressed once, usable by any of the three engines."""
    return {
        "embedding": {"dim": dim},
        "index": {
            "buckets": buckets,
            "bucket_capacity": capacity,
            "top_k": top_k,
            "probe_width": probe_width,
            "probe_spread": probe_spread,
            "n_max": buckets * capacity,
            "overflow_depth": 0,
        },
        "crypto": {"block_size_bytes": block, "max_entry_bytes": block},
        "epoch": {
            "epoch_writes": epoch_writes,
            "epoch_ms": epoch_ms,
            "compaction_slots": 64,
            "slot_write_budget": 64,
            "max_load_factor": 0.95,
        },
        "memory": {"arena_bytes": arena_bytes},
        "v2": {"durable": False, "flush_policy": "on_demand"},
        "m": {"durable": False, "payload_blocks": 1, "flush_ms": epoch_ms},
    }


def merge(base: dict, extra: dict) -> dict:
    out = {k: dict(v) for k, v in base.items()}
    for sec, kv in extra.items():
        out.setdefault(sec, {}).update(kv)
    return out


def create(kind: str, dir_: Path, centroids, overrides: dict, ablation: dict | None = None):
    """Build one engine. Returns ``(engine, params_sha256, params_path)``.

    ``kind`` is ``"m"``, ``"p"`` or ``"rs"``. Mnemos-M needs an arena large enough for its
    resident index, which is a function of the geometry, so it is sized here rather than left to
    a caller to get wrong.
    """
    import mnemos_native as native

    os.environ.setdefault("MNEMOS_I_ACCEPT_INSECURE", "1")
    over = {k: dict(v) for k, v in overrides.items()}

    if kind == "m":
        # The resident index is slots * (header + payload); the arena must hold it plus scratch.
        b = over["index"]["buckets"]
        c = over["index"]["bucket_capacity"]
        dim = over["embedding"]["dim"]
        blocks = over.get("m", {}).get("payload_blocks", 1)
        block = over["crypto"]["block_size_bytes"]
        resident = over.get("m", {}).get("payload_residency", "memory")
        hdr_resident = over.get("m", {}).get("header_residency", "memory") == "memory"
        ivf = over.get("m", {}).get("search", "exact") == "ivf"
        per = (
            (16 + dim + 4 if hdr_resident else 0)
            + (blocks * block if resident == "memory" else 0)
            + (2 if ivf else 0)
        )
        need = b * c * per + (16 + dim + 4) + blocks * block + (1 << 16)
        over["memory"] = dict(over.get("memory", {}))
        over["memory"]["arena_bytes"] = max(over["memory"].get("arena_bytes", 0), need)

    pp = dir_.parent / f"{dir_.name}.params.toml"
    h = write_params(pp, over)
    cents = [list(map(float, c)) for c in centroids]
    cls = {"m": native.EngineM, "p": native.EngineV2, "rs": native.Engine}[kind]
    if kind == "rs":
        eng = cls.create(str(dir_), cents, str(pp), True)
    else:
        eng = cls.create(str(dir_), cents, str(pp), True, ablation or {})
    return eng, h, pp


def drain(eng, limit: int = 1_000_000) -> int:
    """Flush until nothing is staged. Returns the number of flushes performed.

    Mnemos-M's flush is **clocked**, so a single ``flush()`` writes at most ``w`` records and a
    bulk load leaves the rest staged. Mnemos-P auto-flushes at ``w`` on ``put``, so a single
    trailing flush is enough for it, but calling this is correct for both and a bulk-load that
    forgets it silently loses records.
    """
    n = 0
    while n < limit and int(eng.stat().get("staged", 0)) > 0:
        eng.flush()
        n += 1
    return n


def kmeans_centroids(vecs: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=2).fit(vecs)
    c = km.cluster_centers_
    return (c / np.linalg.norm(c, axis=1, keepdims=True)).astype(np.float32)


def search_ids(eng, kind: str, q) -> list[str]:
    """Result ids from any engine. Mnemos-RS returns no id, so it returns texts instead."""
    res = eng.search(list(map(float, q)))
    if kind == "rs":
        return [r[0] for r in res]
    return [r[0] for r in res]


SMALL_N = 20


def boot_ci(values, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and a 95% interval for it.

    Below ``SMALL_N`` independent values the interval is Student-t, not bootstrap. A percentile
    bootstrap of a mean over three to five trials can produce only a handful of distinct resample
    means, and its nominal 95% interval is too narrow -- it undercovers, and at n = 3 it can
    collapse to a point that reads as perfect precision. Several experiments here ran three to
    five trials, which is how E8 came to report precision 0.207 "[0.207, 0.207]". The t interval
    is exact for normal trial means and conservative in the right direction when they are not.
    """
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return 0.0, 0.0, 0.0
    if len(v) < SMALL_N:
        m = float(v.mean())
        if len(v) < 2:
            return m, float("-inf"), float("inf")
        from scipy import stats

        half = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
        return m, m - half, m + half
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson interval for a proportion -- correct near 0 and 1, where the normal one is not."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)
