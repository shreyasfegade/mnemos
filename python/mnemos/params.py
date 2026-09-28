"""Read ``params.toml`` -- the same file Rust reads.

One file, one set of numbers, read by the implementation, the evaluation harness, and the
paper's configuration table. A paper and an implementation that disagree about ``k_probe`` is
a finding a reviewer makes for you.
"""

from __future__ import annotations

import hashlib
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def find_params_file(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: cwd) to the workspace root."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        p = candidate / "params.toml"
        if p.is_file():
            return p
    raise FileNotFoundError("params.toml not found in any ancestor directory")


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    """Load and validate the parameter set."""
    path = find_params_file()
    with path.open("rb") as fh:
        params = tomllib.load(fh)
    _validate(params)
    return params


def params_hash() -> str:
    """SHA-256 of the raw file, hex.

    Stamped onto every experiment result so a figure can always be traced to the exact
    configuration that produced it. Hashing the raw bytes rather than the parsed structure
    means comment changes also bump it -- slightly noisy, and the right trade: it is better to
    over-invalidate than to have two results share a hash while differing.
    """
    return hashlib.sha256(find_params_file().read_bytes()).hexdigest()


def slots_read_per_search(params: dict[str, Any] | None = None) -> int:
    """``(k_probe * s + d) * C`` -- the constant-volume invariant, in one place.

    Mirrors ``Params::slots_read_per_search()`` in Rust. A5 checks the observed count against
    this; if they ever disagree, one side has an optimisation the other does not know about.

    ``probe_spread`` was missing here after the query-equality mitigation landed, so this
    function understated read volume by exactly the mitigation's factor -- and understated it
    *silently*, because the Rust side had been updated and nothing compares the two. It is
    checked against Rust now (``test_params_agree_with_rust``); duplicating a formula is
    defensible only while something asserts the copies agree.
    """
    p = params or load()
    idx = p["index"]
    spread = idx.get("probe_spread", 1)
    return int((idx["probe_width"] * spread + idx["decoy_buckets"]) * idx["bucket_capacity"])


def _validate(p: dict[str, Any]) -> None:
    """The same cross-field invariants Rust enforces.

    Duplicated deliberately. The Python harness can construct configurations the Rust engine
    never sees -- sweep cells, ablations -- and an invalid sweep cell that silently produces a
    plausible number is worse than one that crashes.
    """
    if p.get("schema_version") != 2:
        raise ValueError(f"unsupported schema_version {p.get('schema_version')}; expected 2")

    if p["epoch"]["wal_records"] != p["epoch"]["epoch_writes"]:
        raise ValueError(
            "epoch.wal_records must equal epoch.epoch_writes, or WAL fill level leaks "
            "pending-write count"
        )

    if p["index"]["collision_policy"] != "robin-hood":
        raise ValueError(
            f"index.collision_policy = {p['index']['collision_policy']!r} is not uniquely "
            "representing; history independence would be violated (docs/adr/0003)"
        )

    spread = p["index"].get("probe_spread", 1)
    if p["index"]["probe_width"] * spread + p["index"]["decoy_buckets"] > p["index"]["buckets"]:
        raise ValueError(
            "probe_width * probe_spread + decoy_buckets exceeds bucket count; constant-volume "
            "reads cannot be maintained"
        )

    if p["index"].get("overflow_depth", 0) >= p["index"]["probe_width"]:
        raise ValueError(
            f"index.overflow_depth = {p['index'].get('overflow_depth')} must be less than "
            f"probe_width = {p['index']['probe_width']}; a record spilled beyond the probe "
            "width cannot be found by any query"
        )

    if not p["memory"]["lock_pages"]:
        raise ValueError(
            "memory.lock_pages = false violates threat-model assumption TA-4 "
            "(plaintext must never reach swap)"
        )
