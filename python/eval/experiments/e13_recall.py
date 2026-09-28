"""E13: retrieval quality against BEIR qrels, nDCG@10 and Recall@10, at the shipped s = 2.

The earlier recall number used corpus passages as queries, which guarantees a self-retrieval hit
of 1.000. Here we use the BEIR test queries with their relevance judgments (qrels), so the metric
measures real retrieval, and we report the shipped configuration (probe_spread = 2), not only
s = 1. Plaintext float IVF, Mnemos v1 (int8), and Mnemos v2 (int8) are compared on the same
queries so quantisation and the security machinery are separated.

    cd python && python -m eval.experiments.e13_recall --domain scifact
"""

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from eval.lab import corpus as C
from eval.lab.embcache import embed_cached
from eval.lab.stores import create_v1, create_v2

D2 = C.repo_root() / "data" / "d2-beir"
OUT = C.repo_root() / "python" / "eval" / "results" / "e13_recall.json"


def load_beir(domain: str, max_corpus: int = 5000, max_queries: int = 300):
    z = zipfile.ZipFile(D2 / "_zips" / f"{domain}.zip")
    prefix = f"{domain}/"
    corpus = {}
    for line in z.read(prefix + "corpus.jsonl").decode("utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            corpus[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()
    queries = {}
    for line in z.read(prefix + "queries.jsonl").decode("utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    qrels = {}
    name = [n for n in z.namelist() if n.endswith("test.tsv")]
    for line in z.read(name[0]).decode("utf-8").splitlines()[1:]:
        qid, did, score = line.split("\t")
        if int(score) > 0:
            qrels.setdefault(qid, set()).add(did)
    # keep queries that have judged docs present in our (possibly truncated) corpus
    cids = list(corpus)[:max_corpus]
    cset = set(cids)
    for qid in list(qrels):
        qrels[qid] &= cset
        if not qrels[qid]:
            del qrels[qid]
    qids = [q for q in queries if q in qrels][:max_queries]
    return cids, [corpus[c] for c in cids], qids, [queries[q] for q in qids], qrels


def ndcg_at_k(ranked_ids, relevant, k=10):
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if d in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, relevant, k=10):
    return len(set(ranked_ids[:k]) & relevant) / len(relevant) if relevant else 0.0


def eval_plaintext(cvecs, cids, qvecs, qids, qrels):
    ndcgs, recs = [], []
    for qi, qid in enumerate(qids):
        sims = cvecs @ qvecs[qi]
        order = np.argsort(-sims)[:10]
        ranked = [cids[i] for i in order]
        ndcgs.append(ndcg_at_k(ranked, qrels[qid]))
        recs.append(recall_at_k(ranked, qrels[qid]))
    return float(np.mean(ndcgs)), float(np.mean(recs))


def eval_engine(kind, cvecs, cids, qvecs, qids, qrels, s):
    tmp = Path(tempfile.mkdtemp())
    B = max(16, int(np.sqrt(len(cids))))
    Cap = max(64, int(len(cids) / B * 3))
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=B, random_state=0, n_init=2).fit(cvecs)
    cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    over = {
        "embedding": {"dim": cvecs.shape[1]},
        "index": {
            "buckets": B,
            "bucket_capacity": Cap,
            "top_k": 10,
            "probe_width": 4,
            "probe_spread": s,
            "n_max": B * Cap,
            "overflow_depth": 2 if kind == "v1" else 0,
        },
        "crypto": {"block_size_bytes": 2048, "max_entry_bytes": 2048},
        "epoch": {
            "epoch_writes": 16,
            "epoch_ms": 3_600_000,
            "compaction_slots": 64,
            "slot_write_budget": 64,
            "max_load_factor": 0.95,
        },
        "memory": {"arena_bytes": 1 << 20},
        "v2": {"durable": False, "flush_policy": "on_demand", "payload_blocks": 1},
    }
    eng = (create_v2 if kind == "v2" else create_v1)(tmp / "s", cents, over)[0]
    for i, cid in enumerate(cids):
        with contextlib.suppress(Exception):
            eng.put(f"{cid}", cvecs[i].tolist(), "{}")
    eng.flush()
    ndcgs, recs = [], []
    for qi, qid in enumerate(qids):
        res = eng.search(qvecs[qi].tolist())
        ranked = []
        for r in res:
            text = r[1] if kind == "v2" else r[0]
            ranked.append(text.split("")[0])
        ndcgs.append(ndcg_at_k(ranked, qrels[qid]))
        recs.append(recall_at_k(ranked, qrels[qid]))
    with contextlib.suppress(Exception):
        eng.close()
    return float(np.mean(ndcgs)), float(np.mean(recs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["scifact", "nfcorpus", "fiqa"])
    args = ap.parse_args()
    results = {"experiment": "E13", "domains": []}
    for domain in args.domains:
        cids, ctexts, qids, qtexts, qrels = load_beir(domain)
        if not qids:
            print(f"{domain}: no usable queries, skipping")
            continue
        cvecs = embed_cached(ctexts)
        qvecs = embed_cached(qtexts)
        row = {"domain": domain, "n_corpus": len(cids), "n_queries": len(qids)}
        row["plaintext"] = dict(
            zip(("ndcg", "recall"), eval_plaintext(cvecs, cids, qvecs, qids, qrels), strict=True)
        )
        for s in (1, 2):
            row[f"v1_s{s}"] = dict(
                zip(
                    ("ndcg", "recall"),
                    eval_engine("v1", cvecs, cids, qvecs, qids, qrels, s),
                    strict=True,
                )
            )
            row[f"v2_s{s}"] = dict(
                zip(
                    ("ndcg", "recall"),
                    eval_engine("v2", cvecs, cids, qvecs, qids, qrels, s),
                    strict=True,
                )
            )
        results["domains"].append(row)
        print(
            f"{domain}: plaintext nDCG {row['plaintext']['ndcg']:.3f} | "
            f"v2 s=2 nDCG {row['v2_s2']['ndcg']:.3f} recall {row['v2_s2']['recall']:.3f} | "
            f"v1 s=2 nDCG {row['v1_s2']['ndcg']:.3f}",
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
