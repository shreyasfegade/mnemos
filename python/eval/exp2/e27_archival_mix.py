"""E27: how much historical knowledge the adversary needs.

E4 compares an adversary that induces entirely with archival content (cohort H's own era) against
one that induces entirely with current content. The question is how much alignment is required.
This mixes the two: per topic, a fraction ``alpha`` of the induction budget is drawn from the
archive and the rest from the current era, and reports the advantage on cohort H as alpha grows.
Everything else -- corpus, eras, partitioning, estimator, scoring -- is E4's.

    cd python && python -m eval.exp2.e27_archival_mix
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.cluster import KMeans

from eval.exp2 import e4_drift as E4
from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e27_archival_mix.json"
ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0)


def mixed_pool(topics, archive, current, n_topics, alpha, budget, rng) -> np.ndarray:
    """Exactly ``budget`` induction records per topic, ``alpha`` of them archival."""
    out = []
    for t in range(n_topics):
        a, c = archive[topics[archive] == t], current[topics[current] == t]
        k = min(round(alpha * budget), len(a))
        j = min(budget - k, len(c))
        if k:
            out.extend(rng.choice(a, size=k, replace=False).tolist())
        if j:
            out.extend(rng.choice(c, size=j, replace=False).tolist())
    return np.array(out, dtype=int)


def main() -> None:
    t0 = time.time()
    texts, topics, names, _ = E4.load("github")
    vecs = embed_cached(texts)
    n, n_topics = len(texts), len(names)
    edges = [round(i * n / E4.BINS) for i in range(E4.BINS + 1)]
    era = [np.arange(edges[i], edges[i + 1]) for i in range(E4.BINS)]
    oldest = era[0]
    H, archive = oldest[: len(oldest) // 2], oldest[len(oldest) // 2 :]
    current = organic = era[-1]

    parts = []
    for ks in E4.KMEANS_SEEDS:
        km = KMeans(n_clusters=E4.N_BUCKETS, random_state=ks, n_init=1).fit(vecs)
        cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        parts.append(np.argmax(vecs @ cents.T, axis=1))

    out = {"experiment": "E27", "corpus": "github", "chance": 1.0 / n_topics, "rows": []}
    for alpha in ALPHAS:
        per_seed = []
        for ks, part in zip(E4.KMEANS_SEEDS, parts, strict=True):
            rng = np.random.default_rng(2700 + ks)
            pool = mixed_pool(topics, archive, current, n_topics, alpha, E4.BUDGET_PER_TOPIC, rng)
            r = E4.attack(part, topics, H, pool, organic, n_topics, E4.N_BUCKETS, rng)
            per_seed.append(r["adv_H"])
        m, lo, hi = lab.boot_ci(per_seed)
        out["rows"].append({"alpha": alpha, "adv_H": m, "lo": lo, "hi": hi, "seeds": len(per_seed)})
        print(f"alpha {alpha:.2f}: adv_H {m:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    out["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())} in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
