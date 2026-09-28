"""E11: retrieval quality on the FULL BEIR corpora, with the losses separated.

What was wrong before 
=================================
The earlier run truncated every corpus to 5,000 documents. FiQA has 57,638, so its plaintext
nDCG@10 of 0.579 was measured on a ninth of the corpus, which is why it sat far above the ~0.40
that ``bge`` models score on the full one. It also compared against an *exhaustive* plaintext
search rather than an IVF at the same ``kprobe``, so the reported 6-8 point gap folded together
three different things, and it swallowed capacity refusals during indexing, which silently drop
relevant documents.

What this run does
==================
Full corpora, and the loss decomposed into its parts by adding rungs to the ladder:

``plaintext float, exhaustive``   the reference; what the embedding model can do.
``plaintext float, IVF@kprobe``   the same probing the encrypted engines do, in the clear.
                                  The difference from the row above is the *probing* cost.
``Mnemos-M (int8, exhaustive)``   exact search over the quantised vectors.
                                  The difference from the first row is the *quantisation* cost.
``Mnemos-P (int8, IVF@kprobe)``   both costs together, at s = 1 and the shipped s = 2.
``Mnemos-P grouped probes``       the alternative probe mechanism, whose recall was never reported.

Refusals are counted, not swallowed, and the capacity is reported, so a reader can see whether any
document failed to be indexed.

    cd python && python -m eval.exp2.e11_retrieval --domains scifact nfcorpus fiqa
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

D2 = C.repo_root() / "data" / "d2-beir"
OUT = C.repo_root() / "python" / "eval" / "results" / "e11_retrieval.json"
SEP = "\x00"


def load_beir(domain: str, max_queries: int = 400):
    """The whole corpus. No truncation."""
    z = zipfile.ZipFile(D2 / "_zips" / f"{domain}.zip")
    prefix = f"{domain}/"
    corpus: dict[str, str] = {}
    for line in z.read(prefix + "corpus.jsonl").decode("utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            corpus[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()
    queries: dict[str, str] = {}
    for line in z.read(prefix + "queries.jsonl").decode("utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    qrels: dict[str, set[str]] = {}
    name = [n for n in z.namelist() if n.endswith("test.tsv")]
    for line in z.read(name[0]).decode("utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        qid, did, score = parts[0], parts[1], parts[2]
        if int(score) > 0 and did in corpus:
            qrels.setdefault(qid, set()).add(did)
    cids = list(corpus)
    qids = [q for q in queries if q in qrels][:max_queries]
    return cids, [corpus[c] for c in cids], qids, [queries[q] for q in qids], qrels


def ndcg_at_k(ranked, relevant, k=10):
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, relevant, k=10):
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def score_all(ranked_by_q, qids, qrels):
    n = [ndcg_at_k(ranked_by_q[q], qrels[q]) for q in qids]
    r = [recall_at_k(ranked_by_q[q], qrels[q]) for q in qids]
    nm, nlo, nhi = lab.boot_ci(n)
    rm, rlo, rhi = lab.boot_ci(r)
    return {
        "ndcg": nm,
        "ndcg_lo": nlo,
        "ndcg_hi": nhi,
        "recall": rm,
        "recall_lo": rlo,
        "recall_hi": rhi,
    }


def plaintext_exhaustive(cvecs, cids, qvecs, qids, qrels):
    out = {}
    for i, q in enumerate(qids):
        sims = cvecs @ qvecs[i]
        out[q] = [cids[j] for j in np.argsort(-sims)[:10]]
    return score_all(out, qids, qrels)


def plaintext_ivf(cvecs, cids, qvecs, qids, qrels, cents, kprobe):
    """The matched baseline: same centroids, same probe count, float vectors, no encryption."""
    part = np.argmax(cvecs @ cents.T, axis=1)
    members = [np.where(part == b)[0] for b in range(len(cents))]
    out = {}
    for i, q in enumerate(qids):
        probes = np.argsort(-(cents @ qvecs[i]))[:kprobe]
        idx = np.concatenate([members[b] for b in probes]) if len(probes) else np.array([], int)
        if idx.size == 0:
            out[q] = []
            continue
        sims = cvecs[idx] @ qvecs[i]
        out[q] = [cids[j] for j in idx[np.argsort(-sims)[:10]]]
    return score_all(out, qids, qrels)


def engine_run(
    kind,
    cvecs,
    cids,
    qvecs,
    qids,
    qrels,
    cents,
    *,
    spread,
    grouped=False,
    capacity_mult=3.0,
    extra=None,
):
    tmp = Path(tempfile.mkdtemp())
    try:
        n, dim = cvecs.shape
        b = len(cents)
        cap = int(np.ceil(n / b * capacity_mult))
        over = lab.base_overrides(
            dim=dim,
            buckets=b,
            capacity=cap,
            block=2048,
            epoch_writes=64,
            probe_spread=spread,
            arena_bytes=1 << 21,
        )
        if grouped:
            over = lab.merge(over, {"v2": {"probe_mode": "grouped", "group_size": 8}})
        if extra:
            over = lab.merge(over, extra)
        eng, _, _ = lab.create(kind, tmp / "s", cents, over, {})

        refused = 0
        for i, cid in enumerate(cids):
            try:
                eng.put(cid, cvecs[i].tolist(), "{}")
            except Exception:
                refused += 1
        # Mnemos-M drains w records per clocked flush, so a bulk load needs draining.
        lab.drain(eng)
        eng.flush()

        out = {}
        for i, q in enumerate(qids):
            res = eng.search(qvecs[i].tolist())
            out[q] = [r[1] for r in res]
        s = score_all(out, qids, qrels)
        s["refused_during_indexing"] = refused
        s["capacity_per_partition"] = cap
        s["slots"] = b * cap
        with contextlib.suppress(Exception):
            eng.close()
        return s
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["scifact", "nfcorpus", "fiqa"])
    ap.add_argument("--kprobe", type=int, default=4)
    args = ap.parse_args()

    results = {"experiment": "E11", "kprobe": args.kprobe, "domains": []}
    for domain in args.domains:
        t0 = time.time()
        cids, ctexts, qids, qtexts, qrels = load_beir(domain)
        if not qids:
            print(f"{domain}: no usable queries, skipping")
            continue
        print(f"{domain}: {len(cids)} documents (full corpus), {len(qids)} queries", flush=True)
        cvecs = embed_cached(ctexts)
        qvecs = embed_cached(qtexts)
        # A power of two, so the grouped-probe arm's group_size divides the bucket count. With
        # sqrt(n) rounded naively, SciFact gives 71 buckets and the engine rejects group_size 8.
        b = max(16, 1 << int(np.ceil(np.log2(max(4.0, np.sqrt(len(cids)))))))
        cents = lab.kmeans_centroids(cvecs, b, seed=0)

        row = {"domain": domain, "n_corpus": len(cids), "n_queries": len(qids), "buckets": b}
        row["plaintext_exhaustive"] = plaintext_exhaustive(cvecs, cids, qvecs, qids, qrels)
        row["plaintext_ivf_matched"] = plaintext_ivf(
            cvecs, cids, qvecs, qids, qrels, cents, args.kprobe
        )
        row["m_int8_exhaustive"] = engine_run("m", cvecs, cids, qvecs, qids, qrels, cents, spread=1)
        row["p_s1"] = engine_run("p", cvecs, cids, qvecs, qids, qrels, cents, spread=1)
        row["p_s2"] = engine_run("p", cvecs, cids, qvecs, qids, qrels, cents, spread=2)
        row["p_grouped"] = engine_run(
            "p", cvecs, cids, qvecs, qids, qrels, cents, spread=1, grouped=True
        )
        # Mnemos-M with its in-memory partition index, at three probe widths. Its leakage is
        # Mnemos-M's whatever the width, because probing never reaches the disk; what the width
        # trades is recall against search time, and these rows are the recall half.
        for k in (4, 8, 16):
            row[f"m_ivf_k{k}"] = engine_run(
                "m",
                cvecs,
                cids,
                qvecs,
                qids,
                qrels,
                cents,
                spread=1,
                extra={"m": {"search": "ivf", "probe_width": min(k, b)}},
            )
        row["seconds"] = time.time() - t0
        results["domains"].append(row)

        pe = row["plaintext_exhaustive"]["ndcg"]
        pi = row["plaintext_ivf_matched"]["ndcg"]
        me = row["m_int8_exhaustive"]["ndcg"]
        p2 = row["p_s2"]["ndcg"]
        print(
            f"  plaintext exhaustive {pe:.3f} | plaintext IVF@{args.kprobe} {pi:.3f} "
            f"| M int8 exact {me:.3f} | P s=2 {p2:.3f} | grouped {row['p_grouped']['ndcg']:.3f} "
            f"| M ivf k4/8/16 {row['m_ivf_k4']['ndcg']:.3f}/{row['m_ivf_k8']['ndcg']:.3f}/"
            f"{row['m_ivf_k16']['ndcg']:.3f}",
            flush=True,
        )
        print(
            f"  quantisation cost {pe - me:+.3f} | probing cost {pe - pi:+.3f} | "
            f"refusals P s=2 {row['p_s2']['refused_during_indexing']}",
            flush=True,
        )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
