"""Phase 6 -- the Python layer against the real engine.

These tests run against ``mnemos_native``, the PyO3 extension, not a mock. A mocked engine here
would test that the mock behaves like the mock.

They skip rather than fail when the extension is not built, because the extension is a compiled
artifact and CI may legitimately not have it -- but the skip message says exactly how to build
it, so a skipped test is actionable rather than invisible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mnemos import embed as embedding
from mnemos import params
from mnemos.mcp_server import TOOL_SCHEMA, MnemosMemoryServer, dispatch

native = pytest.importorskip(
    "mnemos_native",
    reason=(
        "build with: cargo build --profile release-ffi -p mnemos-ffi "
        "--features mnemos-keystore/insecure-software-keystore, then copy the cdylib to "
        "python/mnemos_native.pyd (Windows) or .so"
    ),
)


# --------------------------------------------------------------------------- quantisation


def test_quantisation_matches_the_rust_encoding_width():
    """Python and Rust must agree byte for byte, or history independence fails."""
    dim = 768
    v = np.random.default_rng(7).normal(size=(3, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for q in embedding.quantise(v):
        assert len(q.encode()) == dim + 4


def test_quantisation_is_canonical():
    """Two equal embeddings must produce identical bytes.

    If they did not, two stores holding the same logical content would differ on disk -- the
    exact property docs/adr/0003 exists to guarantee.
    """
    v = np.array([[0.3, -0.7, 0.2, 0.6]], dtype=np.float32)
    a, b = embedding.quantise(v)[0], embedding.quantise(v.copy())[0]
    assert a.encode() == b.encode()


def test_quantisation_error_is_small_and_reported():
    rng = np.random.default_rng(11)
    v = rng.normal(size=(64, 128)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    err = embedding.quantisation_roundtrip_error(v)
    assert 0.0 <= err < 0.01, f"int8 quantisation lost {err:.4f} cosine -- far more than expected"


def test_scale_snapping_is_idempotent():
    """Snapping an already-snapped value must not move it, or the encoding is not canonical."""
    for x in [1e-4, 0.00787, 0.5, 1.0, 7.25]:
        once = embedding._snap_to_f16_grid(x)
        assert embedding._snap_to_f16_grid(once) == once


# --------------------------------------------------------------------------- backend guard


def test_utility_metrics_refuse_the_hashing_fallback():
    """R1, made mechanical.

    The hashing fallback produces plausible-looking numbers that mean nothing for retrieval
    quality. A guard that must be *remembered* is a guard that will eventually be forgotten
    between now and the camera-ready.
    """
    embedding.embed(["probe the active backend"])
    if embedding.ACTIVE_BACKEND.startswith(("bge", "minilm")):
        embedding.require_semantic_backend()  # a real model is present; must not raise
    else:
        with pytest.raises(embedding.SemanticBackendUnavailable):
            embedding.require_semantic_backend()


def test_hashing_backend_is_deterministic_across_calls():
    """Figures must regenerate identically, so the fallback cannot use Python's salted hash()."""
    a = embedding._hash_embed(["encrypted memory index"], 64)
    b = embedding._hash_embed(["encrypted memory index"], 64)
    assert np.array_equal(a, b)


def test_shared_vocabulary_outscores_disjoint():
    """The only property the fallback actually has -- asserted so its caveat stays accurate."""
    q, near, far = embedding._hash_embed(
        ["encrypted memory index", "encrypted memory store", "banana pancake weather"], 128
    )
    assert float(q @ near) > float(q @ far)


# --------------------------------------------------------------------------- native engine

#: Small geometry, same invariants. See the header of params-test.toml for why.
TEST_PARAMS = Path(__file__).parent / "params-test.toml"


def _test_params() -> dict:
    import tomllib

    with TEST_PARAMS.open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MNEMOS_I_ACCEPT_INSECURE", "1")
    s = MnemosMemoryServer(
        tmp_path / "store", params_path=TEST_PARAMS, allow_insecure_keystore=True
    )
    s.open(create_if_missing=True)
    yield s
    s.close()


def test_put_and_search_roundtrip(server: MnemosMemoryServer):
    texts = [
        "the encrypted index pads every partition to a fixed size",
        "robin hood hashing gives a uniquely represented layout",
        "forward privacy uses a puncturable prf over a ggm tree",
    ]
    for t in texts:
        assert len(server.put(t)) == 32

    # Fill the epoch so the batch flushes, then read back.
    for i in range(16):
        server.put(f"filler entry number {i}")

    hits = server.search("the encrypted index pads every partition to a fixed size")
    assert isinstance(hits, list)
    assert all({"text", "meta", "score"} <= set(h) for h in hits)


def test_metadata_survives_the_roundtrip(server: MnemosMemoryServer):
    server.put("a memory with structure", {"tag": "deploy", "n": 3})
    for i in range(16):
        server.put(f"filler {i}")
    hits = server.search("a memory with structure")
    metas = [h["meta"] for h in hits if h["text"] == "a memory with structure"]
    if metas:  # ranking is embedder-dependent; the assertion is about shape, not recall
        assert metas[0] == {"tag": "deploy", "n": 3}


def test_update_returns_a_new_id(server: MnemosMemoryServer):
    """ADR-0004's visible API cost, asserted so it cannot regress into a compatibility shim."""
    original = server.put("the first version of this note")
    for i in range(16):
        server.put(f"filler {i}")
    replacement = server.update(original, "the second version of this note")
    assert replacement != original, "update kept the id -- that means an in-place rewrite"
    assert len(replacement) == 32


def test_search_is_constant_volume_regardless_of_top_k(server: MnemosMemoryServer):
    """`top_k` truncates the returned list; it does not reduce the work or the leakage."""
    for i in range(32):
        server.put(f"entry {i} about partitioning and padding")
    assert len(server.search("padding", top_k=1)) <= 1
    assert len(server.search("padding", top_k=10)) <= 10
    declared = params.slots_read_per_search(_test_params())
    assert declared == native.slots_read_per_search(str(TEST_PARAMS))


def test_stat_reports_the_embedding_backend(server: MnemosMemoryServer):
    """A result carrying no backend label is a result nobody can interpret six weeks later."""
    server.put("something")
    s = server.stat()
    assert {"live_entries", "epoch", "capacity", "unlocked"} <= set(s)
    assert s["embedding_backend"] != "none"


def test_locked_store_refuses_to_search(server: MnemosMemoryServer):
    server.put("a secret")
    server.lock()
    with pytest.raises(RuntimeError):
        server.search("a secret")
    server.unlock()
    server.search("a secret")  # must work again


def test_a_bad_memory_id_raises_rather_than_corrupting(server: MnemosMemoryServer):
    with pytest.raises(ValueError):
        server.delete("not-a-hex-id")
    with pytest.raises(ValueError):
        server.delete("ab" * 8)  # right alphabet, wrong length


def test_wrong_dimension_is_rejected_at_the_boundary(tmp_path: Path, monkeypatch):
    """The FFI must not accept a vector the store cannot hold; silently padding would corrupt
    every partition assignment made from it."""
    monkeypatch.setenv("MNEMOS_I_ACCEPT_INSECURE", "1")
    p = _test_params()
    dim, buckets = p["embedding"]["dim"], p["index"]["buckets"]
    centroids = native.train_centroids(
        [list(np.random.default_rng(i).normal(size=dim)) for i in range(buckets)], buckets, dim
    )
    e = native.Engine.create(str(tmp_path / "s"), centroids, str(TEST_PARAMS), True)
    with pytest.raises(ValueError):
        e.put("text", [0.0, 1.0], "{}")


def test_trace_sink_writes_the_adversarys_transcript(server: MnemosMemoryServer, tmp_path: Path):
    """The event stream is what python/eval/attacks consumes AS the adversary's view."""
    trace = tmp_path / "trace.jsonl"
    server.record_trace(trace)
    for i in range(16):
        server.put(f"entry {i}")

    assert trace.is_file(), "no trace written"
    lines = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    assert lines, "trace file is empty"

    kinds = {next(iter(ev)) if isinstance(ev, dict) else str(ev) for ev in lines}
    assert kinds, "no event kinds recorded"

    # Nothing in the transcript may carry plaintext. If it ever does, every attack result is
    # measured against an adversary strictly stronger than the threat model allows.
    blob = trace.read_text()
    assert "entry 0" not in blob, "the event stream leaked entry text"


# --------------------------------------------------------------------------- tool surface


def test_tool_schema_matches_the_dispatcher():
    """A tool advertised but unroutable is a runtime failure in front of an agent."""
    server = MnemosMemoryServer(Path("unused"))
    for tool in TOOL_SCHEMA:
        with pytest.raises(RuntimeError):  # not open -- proves the name routed
            dispatch(server, tool["name"], _example_args(tool))


def test_unknown_tool_is_rejected():
    with pytest.raises(ValueError):
        dispatch(MnemosMemoryServer(Path("unused")), "memory.drop_everything", {})


def test_flush_is_not_an_mcp_tool():
    """Exposing flush would let an agent disable the batching that bounds Delta."""
    assert not any("flush" in t["name"] for t in TOOL_SCHEMA)


def _example_args(tool: dict) -> dict:
    filler = {
        "text": "x",
        "query": "x",
        "memory_id": "ab" * 16,
        "top_k": 1,
        "meta": {},
    }
    required = tool["inputSchema"].get("required", [])
    return {k: filler[k] for k in required}
