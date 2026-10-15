"""Embedding and int8 quantisation.

Model: ``BAAI/bge-base-en-v1.5``, 768-d, cosine. ``all-MiniLM-L6-v2`` (384-d) for speed sweeps.
Both from ``params.toml``.

The model is public and the adversary knows it (assumption TA-5). We claim **no** security from
its secrecy -- a reader will ask, and "none, deliberately" is the only defensible answer.
Anything else would be security through obscurity dressed up as a design choice.

Quantisation must match ``mnemos_core::types::QuantVec`` **exactly**, including the f16 scale
grid. Two representations of the same vector that differ by a rounding step would break history
independence: two stores holding the same logical set would no longer be byte-identical, which
is precisely the property history independence exists to guarantee and the ``proptest`` in
``mnemos-store::slab`` exists to check.

The backend is recorded, never assumed
--------------------------------------
``sentence-transformers`` is a heavy optional dependency and it is not always installed. Rather
than fail at import -- which would take the whole attack lab down with it -- this module falls
back to a **deterministic hashing embedder** and records which backend produced every vector in
:data:`ACTIVE_BACKEND`.

That distinction is load-bearing, so it is enforced rather than documented:
:func:`require_semantic_backend` raises unless a real model is loaded, and every caller that
computes a *utility* number (Recall, nDCG) must call it first. Security results -- A3, A4,
leakage volume, history independence -- do not depend on where vectors come from and are valid
under either backend.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from . import params

#: Which backend produced the vectors this process has been handing out.
#: ``"bge"``/``"minilm"`` for a real model, ``"hashing-INVALID-FOR-UTILITY"`` for the fallback.
ACTIVE_BACKEND: str = "none"


@dataclass(frozen=True)
class QuantVec:
    """int8 scalar-quantised embedding. Mirrors the Rust type byte for byte."""

    codes: np.ndarray  # dtype int8, shape (dim,)
    scale: float  # drawn from the f16 grid so the encoding is canonical

    def encode(self) -> bytes:
        """Serialise exactly as ``QuantVec::encode`` does: codes then a little-endian f32."""
        return self.codes.astype(np.int8).tobytes() + struct.pack("<f", self.scale)

    def cosine(self, other: QuantVec) -> float:
        """Integer dot product scaled once at the end -- the Rust hot loop, in numpy."""
        n = min(len(self.codes), len(other.codes))
        acc = int(np.dot(self.codes[:n].astype(np.int32), other.codes[:n].astype(np.int32)))
        return acc * self.scale * other.scale


class SemanticBackendUnavailableError(RuntimeError):
    """Raised when a utility measurement is attempted without a real embedding model."""


# The earlier name, kept so existing callers and tests keep working.
SemanticBackendUnavailable = SemanticBackendUnavailableError


@lru_cache(maxsize=2)
def _load_model(name: str) -> Any:
    """Load and cache a sentence-transformer, or return ``None`` if unavailable.

    Returning ``None`` rather than raising lets the caller decide. The attack lab genuinely does
    not need semantics; the utility experiments genuinely do.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        return SentenceTransformer(name)
    except Exception:
        return None


def _hash_embed(texts: list[str], dim: int) -> np.ndarray:
    """Deterministic hashing embedder. **Not semantic.**

    Signed feature hashing over whitespace tokens, then L2-normalised. Documents sharing
    vocabulary score higher than documents that do not, which is enough to exercise
    partitioning, retrieval mechanics and every leakage property -- and nowhere near enough to
    support a recall number.

    Deterministic across processes and platforms (BLAKE2b, not :func:`hash`), so a figure
    regenerates identically.
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in text.lower().split():
            h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            v = int.from_bytes(h, "little")
            out[i, v % dim] += -1.0 if (v >> 63) & 1 else 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    result: np.ndarray = out / norms
    return result


def embed(texts: list[str], speed: bool = False, dim: int | None = None) -> np.ndarray:
    """Embed a batch. Returns float32 ``(n, dim)``, L2-normalised.

    Normalised at this boundary so cosine similarity is a plain dot product downstream --
    which lets the Rust hot loop stay in integer arithmetic and keeps the search latency
    budget reachable.

    ``dim`` overrides the width from ``params.toml``. A store created from an override parameter
    file expects vectors of *its* width, and a mismatch is rejected at the FFI boundary rather
    than silently padded -- so the override has to reach the embedder too. A real model cannot
    change its output width, so this is only honoured by the hashing fallback; asking a real
    model for the wrong width raises.
    """
    global ACTIVE_BACKEND

    p = params.load()
    model_name = p["embedding"]["speed_model" if speed else "model"]
    want = dim if dim is not None else p["embedding"]["speed_dim" if speed else "dim"]

    model = _load_model(model_name)
    if model is None:
        ACTIVE_BACKEND = "hashing-INVALID-FOR-UTILITY"
        return _hash_embed(list(texts), want)

    ACTIVE_BACKEND = "minilm" if speed else "bge"
    vectors = np.asarray(
        model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True),
        dtype=np.float32,
    )
    if vectors.shape[1] != want:
        raise ValueError(
            f"{model_name} produced {vectors.shape[1]} dimensions but {want} were requested"
        )
    return vectors


def require_semantic_backend() -> None:
    """Refuse to proceed unless a real embedding model is loaded.

    Call this at the top of anything that computes Recall, nDCG, or any other *utility* number.
    The hashing fallback makes those meaningless while leaving them plausible-looking, which is
    the worst possible failure mode -- so it is turned into a loud one.
    """
    if not ACTIVE_BACKEND.startswith(("bge", "minilm")):
        raise SemanticBackendUnavailableError(
            f"active embedding backend is {ACTIVE_BACKEND!r}. Utility metrics (Recall, nDCG) "
            "require a real model: pip install sentence-transformers. Security metrics do not "
            "and may proceed."
        )


def _snap_to_f16_grid(x: float) -> float:
    """Round a positive float32 to the nearest binary16-representable value.

    Mirrors ``mnemos_core::types::snap_to_f16_grid``: truncate the f32 mantissa from 23 bits to
    10, rounding to nearest. Canonicalisation, not compression -- see the module docstring.
    """
    if not np.isfinite(x) or x <= 0.0:
        return 1.0
    bits = struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]
    discard = 13
    rounded = (bits + (1 << (discard - 1))) & ~((1 << discard) - 1)
    out = struct.unpack("<f", struct.pack("<I", rounded))[0]
    return out if np.isfinite(out) and out > 0.0 else float(x)


def quantise(vectors: np.ndarray) -> list[QuantVec]:
    """Quantise to int8 with a per-vector f16-grid scale.

    Scale is ``max(abs(v)) / 127`` rounded to the nearest representable f16. The rounding is
    not an optimisation -- it makes the encoding canonical, which history independence
    requires (see the module docstring).
    """
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    out: list[QuantVec] = []
    for v in vectors:
        peak = float(np.max(np.abs(v))) if v.size else 0.0
        scale = _snap_to_f16_grid(peak / 127.0 if peak > 0.0 else 1.0)
        codes = np.clip(np.round(v / scale), -127, 127).astype(np.int8)
        out.append(QuantVec(codes=codes, scale=scale))
    return out


def quantisation_roundtrip_error(vectors: np.ndarray) -> float:
    """Mean cosine error introduced by quantisation.

    Reported in the paper alongside recall, so a reader can separate loss caused by
    quantisation from loss caused by the security machinery. Conflating the two would let us
    take credit for a cheap scheme or take blame for an expensive quantiser.
    """
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    quantised = quantise(vectors)
    errors = []
    for original, q in zip(vectors, quantised, strict=True):
        recovered = q.codes.astype(np.float32) * q.scale
        denom = float(np.linalg.norm(original) * np.linalg.norm(recovered))
        if denom < 1e-12:
            continue
        errors.append(1.0 - float(np.dot(original, recovered)) / denom)
    return float(np.mean(errors)) if errors else 0.0
