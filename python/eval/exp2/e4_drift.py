"""E4: does drift protect early memories against an adversary that knows the past?

The earlier result -- advantage on cohort H falling from 0.31 under a shuffled
split to 0.11 under a chronological one -- was read as "being written early protects a record once
the agent's subject matter drifts". But the adversary chooses what to induce. For a public tracker
it can supply *old* issues; in general it can supply any document it believes resembles the past.
If inducing with archival content recovers the loss, the honest statement is narrower: drift
protects early records only against an adversary that lacks knowledge of the historical
distribution.

Design
======
The corpus is sorted by real creation date and cut into ``BINS`` equal-count eras. Cohort H is the
first half of the oldest era. The adversary induces from one era at a time, and the second half of
the oldest era is its **archive** -- records contemporaneous with cohort H that it could plausibly
obtain but that were never stored, so the measurement is never scored on the records it induced.

That gives a curve: advantage on cohort H against the *time gap* between cohort H and the content
the adversary induces. Gap 0 is the historically aware adversary; the largest gap is the naive one
that induces with current content, which is what the earlier experiment measured.

Coarse-taxonomy scoring is stated explicitly rather than left to be inferred: the coarse arm scores
coarse labels against coarse truth, so its chance rate is 1/4 and not 1/12.

    cd python && python -m eval.exp2.e4_drift
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.cluster import KMeans

from eval.attacks import estimators as X
from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e4_drift.json"
E3DIR = C.repo_root() / "data" / "e3"

N_BUCKETS = 128
BINS = 6
BUDGET_PER_TOPIC = 100
W = 16
KMEANS_SEEDS = range(5)
MAX_RECORDS = 7200

# The kubernetes sig/* areas, grouped as a coarser adversary would group them.
COARSE = {
    "node": "compute",
    "scheduling": "compute",
    "apps": "compute",
    "network": "infra",
    "storage": "infra",
    "cloud-provider": "infra",
    "api-machinery": "controlplane",
    "auth": "controlplane",
    "cli": "tooling",
    "testing": "tooling",
    "instrumentation": "tooling",
    "windows": "tooling",
}


def load(name: str):
    """Texts, topic indices and topic names, capped per topic and sorted by timestamp."""
    rows = [
        json.loads(line)
        for line in (E3DIR / f"{name}.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    names = sorted({r["topic"] for r in rows})
    idx = {t: i for i, t in enumerate(names)}
    by_topic: dict[str, list[dict]] = {t: [] for t in names}
    for r in rows:
        by_topic[r["topic"]].append(r)
    cap = max(50, MAX_RECORDS // len(names))
    kept: list[dict] = []
    for t in names:
        kept.extend(by_topic[t][:cap])
    kept.sort(key=lambda r: r["ts"])
    return [r["text"] for r in kept], np.array([idx[r["topic"]] for r in kept]), names, kept


def attack(
    part, topics, H, induce_pool, organic, n_topics, n_buckets, rng, budget=BUDGET_PER_TOPIC
):
    """Label partitions from induced writes drawn from ``induce_pool``; score on cohort H."""
    induced = []
    for t in range(n_topics):
        pool = induce_pool[topics[induce_pool] == t]
        if len(pool):
            induced.extend(rng.choice(pool, size=min(budget, len(pool)), replace=False).tolist())
    induced = np.array(induced, dtype=int)
    if induced.size == 0:
        return {"adv_H": float("nan"), "acc_H": float("nan"), "coverage_H": float("nan")}

    org_parts = part[organic]
    pi = (np.bincount(org_parts, minlength=n_buckets) + 1.0) / (len(org_parts) + n_buckets)
    flushes: X.Flushes = {t: [] for t in range(n_topics)}
    for r in induced:
        org = rng.choice(org_parts, size=W - 1, replace=True)
        s = np.concatenate([[part[r]], org]).astype(np.int64)
        rng.shuffle(s)
        flushes[int(topics[r])].append(s)
    lab_, _ = X.em(flushes, pi, n_buckets)
    chance = 1.0 / n_topics
    sH = X.score(lab_, part[H], topics[H], n_topics)
    # The advantage is taken from the adversary that guesses where it has no label; see
    # estimators.score. The abstaining accuracy is kept so the difference stays visible.
    return {
        "adv_H": sH["accuracy_guess"] - chance,
        "acc_H": sH["accuracy_guess"],
        "acc_H_abstaining": sH["accuracy"],
        "coverage_H": sH["coverage"],
        "chance": chance,
    }


def coarsen(topics, names):
    coarse_names = sorted(set(COARSE.values()))
    cidx = {c: i for i, c in enumerate(coarse_names)}
    return np.array([cidx[COARSE[names[t]]] for t in topics]), coarse_names


def cohort_h(n: int):
    """Cohort H: the first half of the oldest of ``BINS`` equal-count eras."""
    edges = [round(i * n / BINS) for i in range(BINS + 1)]
    oldest = np.arange(edges[0], edges[1])
    return oldest[: len(oldest) // 2]


def cohort_prior(topics, H, n_topics) -> dict:
    """Cohort H's topic prior: its empirical entropy, so the leakage ledger's Fano bound uses the
    true H(X) rather than log2 K, and its majority-class rate, so advantage
    can be reported over the skewed baseline and not only over 1/K."""
    import math

    counts = np.bincount(np.asarray(topics)[H], minlength=n_topics)
    tot = counts.sum()
    return {
        "majority_rate": float(counts.max() / tot) if tot else 1.0 / n_topics,
        "label_entropy_bits": -sum((c / tot) * math.log2(c / tot) for c in counts if c > 0)
        if tot
        else 0.0,
    }


def run(texts, topics, names, coarse: bool):
    n_topics = len(names)
    vecs = embed_cached(texts)
    n = len(texts)

    if coarse:
        topics, names = coarsen(topics, names)
        n_topics = len(names)

    # Equal-count eras over the (already chronological) corpus.
    edges = [round(i * n / BINS) for i in range(BINS + 1)]
    era = [np.arange(edges[i], edges[i + 1]) for i in range(BINS)]
    H = cohort_h(n)  # cohort H: stored, never induced
    archive = era[0][len(H) :]  # same era, available to the adversary, never stored
    organic = era[-1]  # what the agent is writing now, the null distribution

    curve = []
    for gap in range(BINS):
        pool = archive if gap == 0 else era[gap]
        per_seed = []
        for ks in KMEANS_SEEDS:
            km = KMeans(n_clusters=N_BUCKETS, random_state=ks, n_init=1).fit(vecs)
            cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
            part = np.argmax(vecs @ cents.T, axis=1)
            r = attack(
                part,
                topics,
                H,
                pool,
                organic,
                n_topics,
                N_BUCKETS,
                np.random.default_rng(1000 * gap + ks),
            )
            per_seed.append(r["adv_H"])
        m, lo, hi = lab.boot_ci(per_seed)
        curve.append(
            {
                "gap_eras": gap,
                "adv_H": m,
                "lo": lo,
                "hi": hi,
                "n_induce_pool": len(pool),
                "chance": 1.0 / n_topics,
                "adversary": "historically aware"
                if gap == 0
                else ("naive (current content)" if gap == BINS - 1 else "intermediate"),
            }
        )
        print(
            f"  gap {gap} eras ({curve[-1]['adversary']:>24}): adv_H {m:+.3f} [{lo:+.3f}, "
            f"{hi:+.3f}]",
            flush=True,
        )
    return {
        "n_topics": n_topics,
        "chance": 1.0 / n_topics,
        **cohort_prior(topics, H, n_topics),
        "n_records": n,
        "curve": curve,
    }


def shuffled_control(texts, topics, names):
    """The i.i.d. split, for comparison: cohorts drawn from the same distribution."""
    vecs = embed_cached(texts)
    n_topics = len(names)
    n = len(texts)
    per_seed = []
    for ks in KMEANS_SEEDS:
        km = KMeans(n_clusters=N_BUCKETS, random_state=ks, n_init=1).fit(vecs)
        cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        part = np.argmax(vecs @ cents.T, axis=1)
        perm = np.random.default_rng(100 + ks).permutation(n)
        H, P = perm[: n // 2], perm[n // 2 :]
        cut = len(P) // 2
        r = attack(
            part, topics, H, P[:cut], P[cut:], n_topics, N_BUCKETS, np.random.default_rng(ks)
        )
        per_seed.append(r["adv_H"])
    m, lo, hi = lab.boot_ci(per_seed)
    return {"adv_H": m, "lo": lo, "hi": hi, "chance": 1.0 / n_topics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="github")
    ap.add_argument(
        "--backfill-prior",
        action="store_true",
        help="add cohort H's majority rate and entropy to an existing result without rerunning: "
        "both depend only on the corpus and the era split, not on any attack",
    )
    args = ap.parse_args()
    t0 = time.time()
    if args.backfill_prior:
        texts, topics, names, _ = load(args.corpus)
        results = json.loads(OUT.read_text(encoding="utf-8"))
        H = cohort_h(len(texts))
        results["fine"].update(cohort_prior(topics, H, len(names)))
        ct, cn = coarsen(topics, names)
        results["coarse"].update(cohort_prior(ct, H, len(cn)))
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print("fine", results["fine"]["majority_rate"], "coarse", results["coarse"]["majority_rate"])
        return

    texts, topics, names, _ = load(args.corpus)
    print(f"{args.corpus}: {len(texts)} records, {len(names)} topics, {BINS} eras", flush=True)

    results = {
        "experiment": "E4",
        "corpus": args.corpus,
        "n_buckets": N_BUCKETS,
        "bins": BINS,
        "budget_per_topic": BUDGET_PER_TOPIC,
        "w": W,
    }
    print("shuffled control (i.i.d. split):", flush=True)
    results["shuffled_control"] = shuffled_control(texts, topics, names)
    print(f"  adv_H {results['shuffled_control']['adv_H']:+.3f}", flush=True)

    print("fine taxonomy (12 areas, chance 1/12):", flush=True)
    results["fine"] = run(texts, topics, names, coarse=False)
    print("coarse taxonomy (4 areas, chance 1/4, scored against coarse truth):", flush=True)
    results["coarse"] = run(texts, topics, names, coarse=True)

    fine = results["fine"]["curve"]
    aware, naive = fine[0]["adv_H"], fine[-1]["adv_H"]
    results["recovery_fraction"] = (
        (aware - naive) / (results["shuffled_control"]["adv_H"] - naive)
        if results["shuffled_control"]["adv_H"] != naive
        else float("nan")
    )
    print(
        f"\nhistorically aware {aware:+.3f} vs naive {naive:+.3f} vs i.i.d. "
        f"{results['shuffled_control']['adv_H']:+.3f}; the aware adversary recovers "
        f"{results['recovery_fraction']:.0%} of what drift took away",
        flush=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
