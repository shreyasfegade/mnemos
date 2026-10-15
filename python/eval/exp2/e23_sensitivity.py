"""E23: sensitivity of the capacity/refusal behaviour to geometry and skew.

The question is how the designs depend on the parameters. The
capacity--confidentiality trade-off (prop:capacity) is the one most sensitive to geometry and
workload skew, and the cheapest to sweep: it needs only puts, no attack. This measures, for
Mnemos-P and Mnemos-M, the utilisation at the first refusal as a function of the partition count
B, the per-partition capacity C, and the Zipf skew of the topic workload. It confirms that Mnemos-P
leaves the admissible regime early under skew --- the point of sec:eval-refusal --- and that
Mnemos-M, with only a global bound, admits essentially every slot regardless of geometry or skew.

    cd python && python -m eval.exp2.e23_sensitivity
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e23_sensitivity.json"
DIM = 768
BLOCK = 256
W = 16

BUCKETS = (64, 128, 256)
CAPS = (16, 32, 64)
ZIPFS = (0.8, 1.0, 1.2)


def skewed(topics: np.ndarray, n_topics: int, n: int, zipf_a: float, rng) -> np.ndarray:
    wts = 1.0 / np.arange(1, n_topics + 1) ** zipf_a
    wts /= wts.sum()
    order = rng.permutation(n_topics)
    tw = np.zeros(n_topics)
    tw[order] = wts
    by_topic = {t: np.where(topics == t)[0] for t in range(n_topics)}
    picks = []
    for _ in range(n):
        t = int(rng.choice(n_topics, p=tw))
        if len(by_topic[t]):
            picks.append(int(rng.choice(by_topic[t])))
    return np.array(picks)


def first_refusal(kind: str, b: int, c: int, zipf_a: float, vecs, topics, n_topics, cents) -> float:
    rng = np.random.default_rng(23)
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.base_overrides(dim=DIM, buckets=b, capacity=c, block=BLOCK, epoch_writes=W)
        eng, _, _ = lab.create(kind, tmp / "s", cents, over, {})
        order = skewed(topics, n_topics, b * c * 2, zipf_a, rng)
        admitted = 0
        refusal_at = None
        for n, i in enumerate(order):
            try:
                eng.put(f"m{n}", vecs[i].tolist(), "{}")
                admitted += 1
            except Exception:
                if refusal_at is None:
                    refusal_at = admitted
                    break
            if n % (W * 4) == 0:
                lab.drain(eng)
        lab.drain(eng)
        with contextlib.suppress(Exception):
            eng.close()
        total = b * c
        return (refusal_at if refusal_at is not None else admitted) / total
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    t0 = time.time()
    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ][:4000]
    # The same 4,000 records as E22, whose embeddings are cached; the workload samples from them
    # with replacement, so the corpus size does not bound the number of puts.
    vecs = embed_cached([r["text"] for r in rows])
    # Topic labels: cluster into a modest number of topics for the skew to act on.
    n_topics = 12
    tcents = lab.kmeans_centroids(vecs, n_topics, seed=1)
    topics = np.argmax(vecs @ tcents.T, axis=1)

    out: dict = {"experiment": "E23", "zipf_values": list(ZIPFS), "grid": []}
    # Default geometry held while sweeping each axis in turn, plus the skew sweep at the default.
    for b in BUCKETS:
        for c in CAPS:
            cents = lab.kmeans_centroids(vecs, b, seed=0)
            p = first_refusal("p", b, c, 1.0, vecs, topics, n_topics, cents)
            m = first_refusal("m", b, c, 1.0, vecs, topics, n_topics, cents)
            out["grid"].append(
                {"buckets": b, "capacity": c, "zipf": 1.0, "p_util": p, "m_util": m}
            )
            print(f"B={b:>4} C={c:>3} zipf=1.0: P util {p:.3f}  M util {m:.3f}", flush=True)
    cents = lab.kmeans_centroids(vecs, 256, seed=0)
    for z in ZIPFS:
        p = first_refusal("p", 256, 32, z, vecs, topics, n_topics, cents)
        m = first_refusal("m", 256, 32, z, vecs, topics, n_topics, cents)
        out["grid"].append({"buckets": 256, "capacity": 32, "zipf": z, "p_util": p, "m_util": m})
        print(f"B= 256 C= 32 zipf={z}: P util {p:.3f}  M util {m:.3f}", flush=True)

    p_utils = [g["p_util"] for g in out["grid"]]
    m_utils = [g["m_util"] for g in out["grid"]]
    out["p_util_worst"] = float(min(p_utils))
    out["p_util_best"] = float(max(p_utils))
    out["m_util_worst"] = float(min(m_utils))
    out["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(
        f"P utilisation at first refusal: {out['p_util_worst']:.3f}-{out['p_util_best']:.3f}; "
        f"M worst {out['m_util_worst']:.3f}. wrote {OUT.relative_to(C.repo_root())}"
    )


if __name__ == "__main__":
    main()
