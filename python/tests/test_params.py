"""Tests for the parameter loader.

These look trivial and are not. Each asserts a cross-field invariant that, if violated,
silently breaks a security property while leaving the system apparently working -- the worst
failure mode this project has. A configuration bug produces plausible numbers, plausible
figures, and a wrong paper.
"""

from __future__ import annotations

import pytest

from mnemos import params


def test_shipped_params_are_valid() -> None:
    p = params.load()
    assert p["schema_version"] == 2


def test_constant_volume_formula_matches_the_rust_side() -> None:
    """``(k_probe * s + d) * C``. Mirrors ``Params::slots_read_per_search()``.

    If the two implementations disagree, A5 is checking observed read counts against the
    wrong expectation and would pass while the invariant is broken.

    ``probe_spread`` belongs in the formula: widening multiplies the count by a constant, which
    is exactly what keeps it independent of the query. This assertion was written against the
    pre-mitigation formula and started failing the moment ``params.toml`` shipped ``s = 2`` ---
    which is the correct behaviour for it to have. Its Rust twin failed for the same reason on
    the same commit.
    """
    p = params.load()
    expected = (
        p["index"]["probe_width"] * p["index"]["probe_spread"] + p["index"]["decoy_buckets"]
    ) * p["index"]["bucket_capacity"]
    assert params.slots_read_per_search(p) == expected


def test_wal_length_equals_batch_size() -> None:
    """A WAL longer than one batch would let its fill level count pending writes."""
    p = params.load()
    assert p["epoch"]["wal_records"] == p["epoch"]["epoch_writes"]


def test_collision_policy_is_uniquely_representing() -> None:
    """Plain linear probing is not history-independent -- see the paper, Section 5."""
    p = params.load()
    assert p["index"]["collision_policy"] == "robin-hood"


def test_page_locking_is_required() -> None:
    """TA-4: plaintext must never reach swap."""
    p = params.load()
    assert p["memory"]["lock_pages"] is True


def test_probe_plus_decoys_fits_in_the_bucket_space() -> None:
    p = params.load()
    widened = p["index"]["probe_width"] * p["index"]["probe_spread"]
    assert widened + p["index"]["decoy_buckets"] <= p["index"]["buckets"]


def test_validation_rejects_a_leaky_wal_length() -> None:
    p = params.load()
    broken = {**p, "epoch": {**p["epoch"], "wal_records": p["epoch"]["epoch_writes"] + 1}}
    with pytest.raises(ValueError, match="wal_records"):
        params._validate(broken)


def test_validation_rejects_linear_probing() -> None:
    p = params.load()
    broken = {**p, "index": {**p["index"], "collision_policy": "linear"}}
    with pytest.raises(ValueError, match="uniquely representing"):
        params._validate(broken)


def test_params_hash_is_stable_within_a_run() -> None:
    assert params.params_hash() == params.params_hash()


def test_slots_read_per_search_agrees_with_rust() -> None:
    """The formula exists in two languages. Something has to assert the copies agree.

    They did not. `probe_spread` landed in the Rust formula with the query-equality mitigation
    and never reached the Python one, so this function understated read volume by exactly the
    mitigation's factor -- silently, because nothing compared them. The A5 harness checks
    observed read counts against the Python number, so the disagreement would have shown up as
    the *engine* reading twice as much as its own invariant allowed.
    """
    native = pytest.importorskip("mnemos_native")
    assert params.slots_read_per_search() == native.slots_read_per_search()


def test_validation_rejects_a_spread_that_overflows_the_bucket_space() -> None:
    p = params.load()
    broken = {**p, "index": {**p["index"], "probe_spread": p["index"]["buckets"]}}
    with pytest.raises(ValueError, match="constant-volume"):
        params._validate(broken)


def test_validation_rejects_unfindable_overflow_depth() -> None:
    """A record spilled past the probe width is not found even by a search for its own vector.

    That would replace a silent drop with a silent misplacement, which is the same defect
    wearing a hat.
    """
    p = params.load()
    broken = {**p, "index": {**p["index"], "overflow_depth": p["index"]["probe_width"]}}
    with pytest.raises(ValueError, match="overflow_depth"):
        params._validate(broken)


def test_the_shipped_default_has_the_query_equality_mitigation_on() -> None:
    """Pinned so a revert to `s = 1` breaks CI rather than a deployment.

    E10 measures pairwise query-equality recovery at Youden's J = 1.000 with `s = 1` and 0.000
    with `s = 2`. Shipping the measurable configuration rather than the safe one is defensible
    for a benchmark and not for a store.
    """
    assert params.load()["index"]["probe_spread"] >= 2
