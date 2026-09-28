"""E7 -- does an agent retrieve as well through Mnemos as through a plaintext store?

The security results say what an attacker cannot learn. This says what the *user* gives up, and
without it the paper is only half an argument: a store that leaks nothing because it returns
nothing would score perfectly on every attack in this repository.

Design
======
One corpus, embedded **once**, with the real ``bge-base-en-v1.5`` model. The identical vectors
are then fed to three retrievers:

* **Exact brute force** -- ranks every entry by cosine. This is the ground truth, not a
  competitor: Recall and nDCG are measured against it.
* **Plaintext IVF** -- same partitioning, same ``probe_width``, no encryption, no padding, no
  epochs. The utility ceiling for an approximate index.
* **Mnemos** -- the real compiled engine, through the MCP surface an agent would actually use.

Because all three see the same vectors and the same partition geometry, any gap between
Plaintext IVF and Mnemos is attributable to the security machinery and nothing else. A run that
embedded twice, or partitioned differently, would be measuring the embedder.

Why the backend guard is not optional here
==========================================
:func:`mnemos.embed.require_semantic_backend` is called before anything else. The hashing
fallback produces vectors with no semantic structure, so Recall@10 computed over them is a
number with no meaning that nonetheless *looks* like a number. This experiment refuses to run
without the real model, which is research debt R1 turned into a precondition instead of a note.

Metrics
=======
* **Recall@10** -- share of the true top-10 that the retriever returned. Set overlap, not rank
  correlation: returning the right ten in a different order has not lost recall.
* **nDCG@10** -- rank-sensitive, so a retriever that finds the right entries but buries them is
  penalised where Recall would not notice.
* **Self-retrieval** -- given an entry's own text as the query, is that entry in the top 10?
  The closest thing to "can the agent find the note it wrote", and the one an agent's task
  success actually depends on.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval.lab import corpus as C

#: Corpus size. Large enough for the partitioning to matter, small enough to run in minutes.
N_ENTRIES = 800
#: Queries drawn from the corpus.
N_QUERIES = 150
TOP_K = 10
SEED = 20_260_814


@dataclass
class UtilityRow:
    retriever: str
    recall_at_10: float
    ndcg_at_10: float
    self_retrieval: float
    n_queries: int
    embedding_backend: str


class _CachedEmbedder:
    """Serves precomputed vectors by text.

    Exists so Mnemos and the plaintext baseline provably receive **identical** vectors: the
    model is called once, and both retrievers read from the same table. Re-embedding per
    retriever would leave open the possibility that a difference in results came from
    non-determinism in the model rather than from the index.
    """

    def __init__(self, table: dict[str, np.ndarray], dim: int) -> None:
        self._table = table
        self._dim = dim

    def embed(self, texts: list[str], speed: bool = False, dim: int | None = None) -> np.ndarray:
        del speed, dim
        missing = [t for t in texts if t not in self._table]
        if missing:
            raise KeyError(f"{len(missing)} texts were not pre-embedded; the cache is incomplete")
        return np.vstack([self._table[t] for t in texts])


def _ndcg(returned: list[int], ideal: list[int], k: int) -> float:
    """nDCG@k with binary relevance against the exact top-k."""
    rel = set(ideal[:k])
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(returned[:k]) if d in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(rel))))
    return float(dcg / idcg) if idcg else 0.0


def _recall(returned: list[int], ideal: list[int], k: int) -> float:
    rel = set(ideal[:k])
    if not rel:
        return 1.0
    return len(rel & set(returned[:k])) / len(rel)


def _write_params(n: int, dim: int) -> tuple[Path, int, int]:
    """Write a parameter file sized for `n` entries, and return it with its geometry.

    Sizing is not a detail. Two independent bounds have to hold and both bit during
    development:

    * ``buckets * bucket_capacity >= n`` -- otherwise the index saturates, entries are silently
      never indexed, and every recall number is measured against a store that does not contain
      the corpus.
    * ``n <= n_max * max_load_factor`` -- otherwise the engine correctly refuses to flush,
      because Robin Hood cascades past that occupancy would make the per-flush write count
      data-dependent again.

    Both are asserted rather than assumed, so a future change to N_ENTRIES fails loudly instead
    of quietly producing a flattering number.
    """
    import tomllib

    base = C.repo_root() / "params.toml"
    with base.open("rb") as fh:
        p = tomllib.load(fh)

    buckets = max(16, 1 << int(np.ceil(np.log2(max(4.0, np.sqrt(n))))))
    capacity = max(32, 1 << int(np.ceil(np.log2(max(2.0, (n / buckets) * 2)))))
    n_max = max(1024, 1 << int(np.ceil(np.log2(n / 0.6 * 1.5))))

    assert buckets * capacity >= n, "index geometry cannot hold the corpus"
    assert n <= n_max * p["epoch"]["max_load_factor"], "corpus exceeds the safe load factor"

    p["embedding"]["dim"] = dim
    p["index"]["buckets"] = buckets
    p["index"]["bucket_capacity"] = capacity
    p["index"]["n_max"] = n_max
    p["index"]["decoy_buckets"] = 0
    p["index"]["top_k"] = TOP_K
    # Pinned to the pre-mitigation configuration. Unlike E9 this genuinely matters: probe spread
    # widens the read set, so it *raises* recall (E10 measures +2.5 pp on its corpus). Measuring
    # utility with the mitigation on would therefore report a smaller gap to plaintext than the
    # security machinery alone is responsible for, and the number quoted in the abstract is
    # meant to be the price of that machinery. The shipped default does slightly better than
    # this figure, which is the direction an under-claim should run in.
    p["index"]["probe_spread"] = 1
    p["crypto"]["block_size_bytes"] = 1024
    p["crypto"]["max_entry_bytes"] = 1024
    p["memory"]["arena_bytes"] = 262_144

    out = Path(tempfile.gettempdir()) / "mnemos-e7-params.toml"
    out.write_text(_to_toml(p), encoding="utf-8")
    print(
        f"store geometry: buckets={buckets} capacity={capacity} "
        f"index_slots={buckets * capacity} n_max={n_max}"
    )
    return out, buckets, p["index"]["probe_width"]


def _to_toml(d: dict) -> str:
    """Minimal TOML writer -- the stdlib reads TOML but does not write it."""
    lines: list[str] = []
    scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
    for k, v in scalars.items():
        lines.append(f"{k} = {_fmt(v)}")
    for section, body in d.items():
        if not isinstance(body, dict):
            continue
        nested = {k: v for k, v in body.items() if isinstance(v, dict)}
        flat = {k: v for k, v in body.items() if not isinstance(v, dict)}
        if flat:
            lines.append(f"\n[{section}]")
            lines.extend(f"{k} = {_fmt(v)}" for k, v in flat.items())
        for sub, subbody in nested.items():
            lines.append(f"\n[{section}.{sub}]")
            lines.extend(f"{k} = {_fmt(v)}" for k, v in subbody.items())
    return "\n".join(lines) + "\n"


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def main(corpus_path: Path | None = None) -> None:
    """Compare retrieval quality through Mnemos against a plaintext IVF.

    ``corpus_path`` selects the corpus. The default is D3, which
    ``eval.lab.corpus`` explicitly warns is the *wrong* corpus for utility metrics: it is
    template-assembled, so it separates more cleanly in embedding space than real prose and
    Recall@10 measured there is optimistic. Pass ``data/d2-beir/notes.jsonl`` for the public
    corpus, which is what the paper should quote.
    """
    from mnemos import embed as embedding
    from mnemos.mcp_server import MnemosMemoryServer

    os.environ.setdefault("MNEMOS_I_ACCEPT_INSECURE", "1")
    rng = np.random.default_rng(SEED)

    corpus = C.load(corpus_path) if corpus_path else C.load()
    entries = corpus.all_entries[:N_ENTRIES]
    print(f"corpus: {corpus_path.name if corpus_path else 'D3 (default, synthetic)'}")
    texts = [e.text[:400] for e in entries]

    # --- embed ONCE, with the real model ------------------------------------------------
    vectors = embedding.embed(texts)
    embedding.require_semantic_backend()  # refuses to proceed on the hashing fallback
    backend = embedding.ACTIVE_BACKEND
    dim = vectors.shape[1]
    print(f"embedding backend: {backend} (dim={dim}), {len(texts)} entries")

    table = {t: vectors[i] for i, t in enumerate(texts)}
    by_text = {t: i for i, t in enumerate(texts)}

    query_idx = rng.choice(len(texts), size=min(N_QUERIES, len(texts)), replace=False)

    # --- ground truth: exact nearest neighbours ------------------------------------------
    # Vectors are L2-normalised, so the inner product IS the cosine.
    sims = vectors @ vectors.T
    exact = {int(q): list(np.argsort(-sims[q])[:TOP_K]) for q in query_idx}

    rows: list[UtilityRow] = []

    # --- ONE partition table, shared by both retrievers ----------------------------------
    # Trained with the engine's own spherical k-means, then handed to BOTH sides. Letting each
    # retriever train its own table would make the measured gap a difference between two
    # k-means runs rather than the cost of encryption, padding and batching.
    import mnemos_native

    params_path, buckets, probe = _write_params(len(texts), dim)

    centroid_list = mnemos_native.train_centroids(
        [list(map(float, v)) for v in vectors], buckets, dim
    )
    centroids = np.asarray(centroid_list, dtype=np.float64)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    # Assignment rule identical to `EncryptedIvf::assign`: nearest centroid by cosine.
    assign = np.argmax(vectors @ centroids.T, axis=1)
    print(f"partition table: {buckets} centroids, probe_width={probe} (shared by both)")

    plain_recall, plain_ndcg, plain_self = [], [], []
    for q in query_idx:
        order = np.argsort(-(centroids @ vectors[q]))[:probe]
        cand = np.where(np.isin(assign, order))[0]
        if cand.size == 0:
            ranked: list[int] = []
        else:
            ranked = list(cand[np.argsort(-(vectors[cand] @ vectors[q]))][:TOP_K])
        plain_recall.append(_recall(ranked, exact[int(q)], TOP_K))
        plain_ndcg.append(_ndcg(ranked, exact[int(q)], TOP_K))
        plain_self.append(1.0 if int(q) in ranked else 0.0)

    rows.append(
        UtilityRow(
            "plaintext-ivf",
            float(np.mean(plain_recall)),
            float(np.mean(plain_ndcg)),
            float(np.mean(plain_self)),
            len(query_idx),
            backend,
        )
    )

    # --- Mnemos, through the real engine -------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        server = MnemosMemoryServer(
            Path(tmp) / "store",
            embedder=_CachedEmbedder(table, dim),
            params_path=params_path,
            allow_insecure_keystore=True,
        )
        server.open(create_if_missing=True, centroids=centroid_list)
        for t in texts:
            server.put(t)
        server.close()

        server = MnemosMemoryServer(
            Path(tmp) / "store",
            embedder=_CachedEmbedder(table, dim),
            params_path=params_path,
            allow_insecure_keystore=True,
        )
        server.open()

        m_recall, m_ndcg, m_self = [], [], []
        for q in query_idx:
            hits = server.search(texts[int(q)], top_k=TOP_K)
            ranked = [by_text[h["text"]] for h in hits if h["text"] in by_text]
            m_recall.append(_recall(ranked, exact[int(q)], TOP_K))
            m_ndcg.append(_ndcg(ranked, exact[int(q)], TOP_K))
            m_self.append(1.0 if int(q) in ranked else 0.0)
        server.close()

    rows.append(
        UtilityRow(
            "mnemos",
            float(np.mean(m_recall)),
            float(np.mean(m_ndcg)),
            float(np.mean(m_self)),
            len(query_idx),
            backend,
        )
    )

    # --- report ---------------------------------------------------------------------------
    print(f"\n{'retriever':<16}{'Recall@10':>12}{'nDCG@10':>10}{'self-retr':>12}")
    for r in rows:
        print(
            f"{r.retriever:<16}{r.recall_at_10:12.3f}{r.ndcg_at_10:10.3f}{r.self_retrieval:12.3f}"
        )

    plain, mnem = rows[0], rows[1]
    d_recall = (mnem.recall_at_10 - plain.recall_at_10) * 100
    d_self = (mnem.self_retrieval - plain.self_retrieval) * 100
    print(
        f"\nMnemos vs plaintext IVF: Recall@10 {d_recall:+.1f} pp, self-retrieval {d_self:+.1f} pp"
    )
    print(
        "Target (params.toml targets.utility.agent_success_delta_pp): within 2.0 pp -> "
        f"{'MET' if abs(d_self) <= 2.0 else 'MISSED'}"
    )

    out = C.repo_root() / "python" / "eval" / "results" / "e7_agent_utility.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(C.repo_root())}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus",
        default=None,
        help="corpus jsonl; omit for D3, or pass ../data/d2-beir/notes.jsonl for D2",
    )
    args = ap.parse_args()
    main(Path(args.corpus) if args.corpus else None)
