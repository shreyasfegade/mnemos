"""Embedding cache shared by every experiment.

All experiments embed the **full text** with ``BAAI/bge-base-en-v1.5`` (512-token window, the
model's own limit) through :func:`mnemos.embed.embed`, the same path the store uses. Earlier
experiments truncated engine input to 400 characters while the partitioning was computed on the
full text; one embedding path for both removes that mismatch.

Vectors are cached by the SHA-256 of the model name and the exact text list, so a cache hit is
only ever the same computation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from eval.lab.corpus import repo_root

CACHE = repo_root() / "python" / "eval" / "cache"


def _key(texts: list[str], model: str) -> str:
    h = hashlib.sha256(model.encode())
    for t in texts:
        h.update(hashlib.sha256(t.encode("utf-8")).digest())
    return h.hexdigest()[:32]


def embed_cached(texts: list[str], model: str = "BAAI/bge-base-en-v1.5") -> np.ndarray:
    """L2-normalised float32 ``(n, 768)`` embeddings of ``texts``."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"emb-{_key(texts, model)}.npy"
    if path.exists():
        return np.load(path)
    from mnemos import embed as E

    vecs = E.embed(list(texts))
    E.require_semantic_backend()
    vecs = np.asarray(vecs, dtype=np.float32)
    np.save(path, vecs)
    return vecs


def device() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def cache_path_for(texts: list[str], model: str = "BAAI/bge-base-en-v1.5") -> Path:
    return CACHE / f"emb-{_key(texts, model)}.npy"
