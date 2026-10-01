"""E24: what condition S4 costs in retrieval, and what a rebuild costs on disk.

Theorem 1's S4 asks that the partition table be independent of cohort H, which a table trained on
public text satisfies and a table trained on the store's own records does not. The price of S4 is
therefore the retrieval gap between the two. This measures it on SciFact (full corpus, E11's
functions and bge embeddings): a *public* table is k-means over D2's notes, a different corpus; a
*corpus-trained* table is k-means over SciFact itself. Both are scored with the matched plaintext
IVF and with Mnemos-P at s = 2, and Mnemos-M's in-memory index at eight partitions.

A rebuild under a new table rewrites every region of Mnemos-P's file, so its write trace is the
whole file and discloses nothing beyond its size; the byte count is reported from the engine.
Mnemos-M's table is logical and could be retrained in memory with no disk write; the engine does
not implement retraining, so that half is stated, not measured.

    cd python && python -m eval.exp2.e24_table_source
"""

from __future__ import annotations

import json
import time

import numpy as np

from eval.exp2 import e11_retrieval as E11
from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e24_table_source.json"
DOMAIN = "scifact"
KPROBE = 4


def main() -> None:
    t0 = time.time()
    cids, ctexts, qids, qtexts, qrels = E11.load_beir(DOMAIN)
    cvecs, qvecs = embed_cached(ctexts), embed_cached(qtexts)
    pub_rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ][:4000]
    pub = embed_cached([r["text"] for r in pub_rows])
    b = max(16, 1 << int(np.ceil(np.log2(max(4.0, np.sqrt(len(cids)))))))
    tables = {
        "public": lab.kmeans_centroids(pub, b, seed=0),
        "corpus_trained": lab.kmeans_centroids(cvecs, b, seed=0),
    }
    out: dict = {"experiment": "E24", "domain": DOMAIN, "buckets": b, "tables": {}}
    for name, cents in tables.items():
        part = np.argmax(cvecs @ cents.T, axis=1)
        load = np.bincount(part, minlength=b)
        row = {
            "max_partition_share": float(load.max() / len(cids)),
            "empty_partitions": int((load == 0).sum()),
            "plaintext_ivf": E11.plaintext_ivf(cvecs, cids, qvecs, qids, qrels, cents, KPROBE),
            "p_s2": E11.engine_run("p", cvecs, cids, qvecs, qids, qrels, cents, spread=2),
            "m_ivf_k8": E11.engine_run(
                "m",
                cvecs,
                cids,
                qvecs,
                qids,
                qrels,
                cents,
                spread=1,
                extra={"m": {"search": "ivf", "probe_width": min(8, b)}},
            ),
        }
        out["tables"][name] = row
        print(
            f"{name:>15}: plaintext IVF {row['plaintext_ivf']['ndcg']:.3f} | P s=2 "
            f"{row['p_s2']['ndcg']:.3f} (refused {row['p_s2']['refused_during_indexing']}) | "
            f"M ivf k8 {row['m_ivf_k8']['ndcg']:.3f} | max share {row['max_partition_share']:.3f}",
            flush=True,
        )
    out["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
