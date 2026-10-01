"""E26: retrieval with a second embedding model.

E11 measures retrieval with bge-base-en-v1.5 only. This repeats E11's core comparison on two BEIR
domains with a second, independently trained 768-dimensional model, nomic-embed-text (served by a
local Ollama), using E11's own functions: full corpora, the matched-probe plaintext IVF, Mnemos-M's
int8 exact search, Mnemos-P at s = 2, and Mnemos-M's in-memory index at the same eight partitions.

Embeddings are cached under the model name, like ``embcache``.

    cd python && python -m eval.exp2.e26_second_model
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request

import numpy as np

from eval.exp2 import e11_retrieval as E11
from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import CACHE

OUT = C.repo_root() / "python" / "eval" / "results" / "e26_second_model.json"
MODEL = "nomic-embed-text"
URL = "http://localhost:11434/api/embed"
DOMAINS = ("scifact", "nfcorpus")
KPROBE = 4


def embed(texts: list[str], prefix: str) -> np.ndarray:
    """L2-normalised embeddings from Ollama, cached by model, prefix and exact text list."""
    h = hashlib.sha256(f"{MODEL}|{prefix}".encode())
    for t in texts:
        h.update(hashlib.sha256(t.encode("utf-8")).digest())
    path = CACHE / f"emb-ollama-{h.hexdigest()[:32]}.npy"
    if path.exists():
        return np.load(path)
    out = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t[:6000] for t in texts[i : i + 32]]
        req = urllib.request.Request(
            URL,
            data=json.dumps({"model": MODEL, "input": batch, "truncate": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        out.extend(json.loads(urllib.request.urlopen(req, timeout=600).read())["embeddings"])
    v = np.asarray(out, dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, v)
    return v


def main() -> None:
    results = {"experiment": "E26", "model": MODEL, "kprobe": KPROBE, "domains": []}
    for domain in DOMAINS:
        t0 = time.time()
        cids, ctexts, qids, qtexts, qrels = E11.load_beir(domain)
        print(f"{domain}: {len(cids)} documents, {len(qids)} queries; embedding", flush=True)
        cvecs = embed(ctexts, "search_document: ")
        qvecs = embed(qtexts, "search_query: ")
        b = max(16, 1 << int(np.ceil(np.log2(max(4.0, np.sqrt(len(cids)))))))
        cents = lab.kmeans_centroids(cvecs, b, seed=0)
        row = {"domain": domain, "n_corpus": len(cids), "n_queries": len(qids), "buckets": b}
        row["plaintext_exhaustive"] = E11.plaintext_exhaustive(cvecs, cids, qvecs, qids, qrels)
        row["plaintext_ivf_matched"] = E11.plaintext_ivf(
            cvecs, cids, qvecs, qids, qrels, cents, KPROBE
        )
        row["m_int8_exhaustive"] = E11.engine_run(
            "m", cvecs, cids, qvecs, qids, qrels, cents, spread=1
        )
        row["p_s2"] = E11.engine_run("p", cvecs, cids, qvecs, qids, qrels, cents, spread=2)
        row["m_ivf_k8"] = E11.engine_run(
            "m",
            cvecs,
            cids,
            qvecs,
            qids,
            qrels,
            cents,
            spread=1,
            extra={"m": {"search": "ivf", "probe_width": min(8, b)}},
        )
        row["seconds"] = time.time() - t0
        results["domains"].append(row)
        print(
            "  "
            + " | ".join(
                f"{k} {row[k]['ndcg']:.3f}"
                for k in (
                    "plaintext_exhaustive",
                    "plaintext_ivf_matched",
                    "m_int8_exhaustive",
                    "p_s2",
                    "m_ivf_k8",
                )
            ),
            flush=True,
        )
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
