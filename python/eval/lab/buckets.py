"""Semantic partitioning -- the model that actually matches Mnemos's design.

Why this file exists
====================
The first pass of ``schemes.py`` modelled a *keyword* index (Sophos as published) and found
attack A3 essentially powerless: count-only matching over a 500-term vocabulary is capped near
12% even under a classic Zipfian distribution, because rare keywords have small integer counts
and small integers collide en masse. That negative result is real and is kept.

But it is a result about the wrong index. Mnemos is a **semantic** store: memories are embedded
and assigned to IVF partitions. The fair forward-private baseline is therefore a
forward-private index *over partitions*, and partitions differ from keywords in a way that
matters enormously:

    A partition is a contiguous region of a file. Its size is visible to anyone who can read
    the directory entry. No key is involved, no token chain, no cryptanalysis -- just ``ls``.

That is why the paper (Section 6) pads every bucket to capacity and why `history independence`
treats padding as load-bearing rather than as a storage tax. The padding defends against
occupancy leakage, which means occupancy leakage was always the threat the design had in mind.
The keyword model simply was not it.

What this module provides
=========================
A deterministic semantic partitioning of the corpus, so ``schemes.py`` can model what an
unpadded partitioned index leaves on disk.

Embedding substitution
----------------------
``params.toml`` specifies ``BAAI/bge-base-en-v1.5``, but ``sentence-transformers`` pulls in
torch and is not installed here. We substitute **TF-IDF -> truncated SVD -> spherical
k-means**, which is a legitimate stand-in *for this experiment specifically* because attack A3
consumes only partition occupancy and arrival order -- never embedding quality.

It would **not** be legitimate for the utility numbers (Recall@10, nDCG), which depend
directly on embedding quality. Those must wait for the real model, and this substitution is
recorded ino the two never get conflated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .corpus import Corpus, Entry


@dataclass(frozen=True)
class Partitioning:
    """Assignment of every entry to an IVF partition."""

    bucket_of: dict[int, int]
    """``doc_id -> bucket``."""
    n_buckets: int
    embedding_backend: str
    """Recorded in results so a figure can never silently mix backends."""

    def occupancy(self, entries: list[Entry]) -> dict[int, int]:
        """Entries per partition -- **what an unpadded index's region sizes reveal**."""
        occ: dict[int, int] = {}
        for e in entries:
            b = self.bucket_of[e.doc_id]
            occ[b] = occ.get(b, 0) + 1
        return occ

    def arrival_order(self, entries: list[Entry]) -> dict[int, list[int]]:
        """Per partition, the authoring ranks of its entries, ascending.

        An append-structured partition stores entries in arrival order, so their physical
        offsets within the region reveal this directly -- again with no key involved.
        """
        out: dict[int, list[int]] = {}
        for rank, e in enumerate(entries):
            out.setdefault(self.bucket_of[e.doc_id], []).append(rank)
        return out

    def topic_profile(self, entries: list[Entry]) -> dict[int, dict[str, int]]:
        """Per partition, its topic composition. **Scoring only** -- never exposed to A3."""
        prof: dict[int, dict[str, int]] = {}
        for e in entries:
            b = self.bucket_of[e.doc_id]
            prof.setdefault(b, {})
            prof[b][e.topic] = prof[b].get(e.topic, 0) + 1
        return prof

    def dominant_topic(self, entries: list[Entry]) -> dict[int, str]:
        """The majority topic of each partition. Scoring only."""
        return {
            b: max(counts.items(), key=lambda kv: kv[1])[0]
            for b, counts in self.topic_profile(entries).items()
        }


def partition(
    corpus: Corpus,
    n_buckets: int = 256,
    seed: int = 0,
    svd_dim: int = 128,
    backend: str = "auto",
) -> Partitioning:
    """Embed and partition the whole corpus.

    Partitioning is fitted over **all** entries, historical and post together, because a real
    store builds its centroids once and keeps using them. Fitting only on the historical
    cohort would give the post cohort a different partition geometry and quietly confound the
    H-vs-P comparison.

    Parameters
    ----------
    backend
        ``"bge"`` uses the real ``bge-base-en-v1.5`` sentence embedder -- the model an actual
        deployment would use, and the one the paper names. ``"tfidf"`` uses the TF-IDF/SVD
        substitute. ``"auto"`` prefers the real model and falls back, which keeps the harness
        runnable on a machine without the weights.

        The backend is recorded in :attr:`Partitioning.embedding_backend` either way, so a
        result can never be silently misattributed. This matters because the substitute is
        adequate for *structural* attacks -- A3 and A4 consume partition occupancy and arrival
        order, never embedding quality -- and inadequate for anything that measures retrieval
        quality.
    """
    entries = corpus.all_entries
    texts = [e.text for e in entries]

    if backend in ("bge", "auto"):
        dense = _bge_embed(texts)
        if dense is not None:
            km = KMeans(n_clusters=n_buckets, random_state=seed, n_init=4)
            labels = km.fit_predict(dense)
            return Partitioning(
                bucket_of={e.doc_id: int(labels[i]) for i, e in enumerate(entries)},
                n_buckets=n_buckets,
                embedding_backend="bge-base-en-v1.5 (REAL)",
            )
        if backend == "bge":
            raise RuntimeError(
                "backend='bge' requested but sentence-transformers or the weights are "
                "unavailable. Refusing to silently substitute -- pass backend='tfidf' if that "
                "is what you actually want."
            )

    tfidf = TfidfVectorizer(
        max_features=20_000,
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
    )
    x = tfidf.fit_transform(texts)

    # Reduce before clustering: k-means on raw high-dimensional sparse TF-IDF is dominated by
    # vocabulary overlap rather than topic, which would produce partitions that do not
    # correspond to anything semantic and would make the experiment meaningless in a way that
    # is hard to notice.
    dim = min(svd_dim, x.shape[1] - 1)
    dense = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(x)

    # L2-normalise so Euclidean k-means is equivalent to spherical (cosine) k-means, matching
    # `params.toml` embedding.metric = "cosine".
    dense = normalize(dense)

    km = KMeans(n_clusters=n_buckets, random_state=seed, n_init=4)
    labels = km.fit_predict(dense)

    return Partitioning(
        bucket_of={e.doc_id: int(labels[i]) for i, e in enumerate(entries)},
        n_buckets=n_buckets,
        embedding_backend=f"tfidf-svd{dim}-kmeans (SUBSTITUTE for bge-base-en-v1.5)",
    )


def _bge_embed(texts: list[str]):
    """Embed with the real model, or return ``None`` if it is not available.

    Returns ``None`` rather than raising so ``backend="auto"`` can fall back; the caller
    decides whether a fallback is acceptable. Embeddings are already L2-normalised, so
    Euclidean k-means over them is spherical k-means, matching ``params.toml``
    ``embedding.metric = "cosine"``.
    """
    try:
        from mnemos import embed as embedding
    except ImportError:
        return None

    vectors = embedding.embed(list(texts))
    if not embedding.ACTIVE_BACKEND.startswith("bge"):
        return None
    return normalize(np.asarray(vectors, dtype=np.float64))
