"""E3: structure transfer on realistic, timestamped corpora, chronological vs shuffled.

The earlier paper shuffled authoring order precisely so cohorts
H and P were an i.i.d. split, which makes the cohort ratio ~1 by construction and removes the only
interesting version of the question. Real agent memory drifts: old memories are about different
things than new ones. This experiment splits each corpus **chronologically** (H = oldest half,
P = newest half) as well as by shuffle (the control), and reports how much of cohort H's topics a
post-tau attack recovers in each case.

Corpora (built by ``data/e3/build_corpora.py``): kubernetes ``github`` issues (real sig/* labels
and creation dates), ``enron`` mail (folder-topic and send date), ``newsgroups`` (fine
overlapping taxonomy, shuffled only). The attack is the realistic frequency/EM estimator from E1,
run against an unpadded partitioned store with a public partition table trained on the whole
corpus.

Open-world taxonomy: for github, a variant where the adversary's topic set is a coarser grouping
than the user's, so labels learned post-tau do not line up one-to-one with cohort-H topics.

    cd python && python -m eval.experiments.e3_realistic
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.cluster import KMeans

from eval.attacks import estimators as X
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e3_realistic.json"
E3 = C.repo_root() / "data" / "e3"
N_BUCKETS = 128
BUDGET_PER_TOPIC = 100
W = 16
KMEANS_SEEDS = range(4)
MAX_RECORDS = 6000


def load(name: str) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """texts, topic indices, timestamp order (argsort), topic names."""
    rows = [json.loads(ln) for ln in (E3 / f"{name}.jsonl").open(encoding="utf-8") if ln.strip()]
    # balance: cap per topic so a few huge topics do not dominate
    names = sorted({r["topic"] for r in rows})
    idx = {t: i for i, t in enumerate(names)}
    by_topic: dict[str, list[dict]] = {t: [] for t in names}
    for r in rows:
        by_topic[r["topic"]].append(r)
    cap = max(50, MAX_RECORDS // len(names))
    kept = []
    for t in names:
        kept.extend(by_topic[t][:cap])
    kept.sort(key=lambda r: r["ts"])
    texts = [r["text"] for r in kept]
    topics = np.array([idx[r["topic"]] for r in kept])
    return texts, topics, np.arange(len(kept)), names


def attack(
    part: np.ndarray,
    topics: np.ndarray,
    H: np.ndarray,
    P: np.ndarray,
    n_topics: int,
    n_buckets: int,
    rng,
) -> dict:
    induced = []
    for t in range(n_topics):
        pool = P[topics[P] == t]
        if len(pool):
            induced.extend(
                rng.choice(pool, size=min(BUDGET_PER_TOPIC, len(pool)), replace=False).tolist()
            )
    induced = np.array(induced)
    P_rest = P[np.isin(P, induced, invert=True)]
    organic = part[P_rest]
    pi = (np.bincount(organic, minlength=n_buckets) + 1.0) / (len(organic) + n_buckets)
    flushes: X.Flushes = {t: [] for t in range(n_topics)}
    for r in induced:
        org = rng.choice(organic, size=W - 1, replace=True)
        s = np.concatenate([[part[r]], org]).astype(np.int64)
        rng.shuffle(s)
        flushes[int(topics[r])].append(s)
    lab, _ = X.em(flushes, pi, n_buckets)
    chance = 1.0 / n_topics
    sH = X.score(lab, part[H], topics[H])
    sP = X.score(lab, part[P_rest], topics[P_rest])
    return {
        "adv_H": sH["accuracy"] - chance,
        "adv_P": sP["accuracy"] - chance,
        "acc_H": sH["accuracy"],
        "coverage_H": sH["coverage"],
        "chance": chance,
        "H_minus_P": sH["accuracy"] - sP["accuracy"],
    }


def run_corpus(name: str, shuffled_only: bool = False) -> dict:
    texts, topics, _, names = load(name)
    n_topics = len(names)
    vecs = embed_cached(texts)
    print(f"{name}: {len(texts)} records, {n_topics} topics", flush=True)

    splits = ["shuffled"] if shuffled_only else ["chronological", "shuffled"]
    out = {"corpus": name, "n_records": len(texts), "n_topics": n_topics, "splits": {}}
    for split in splits:
        accH, hmp, adv = [], [], []
        for ks in KMEANS_SEEDS:
            km = KMeans(n_clusters=N_BUCKETS, random_state=ks, n_init=1).fit(vecs)
            cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
            part = np.argmax(vecs @ cents.T, axis=1)
            n = len(texts)
            if split == "chronological":
                H, P = np.arange(n // 2), np.arange(n // 2, n)
            else:
                perm = np.random.default_rng(100 + ks).permutation(n)
                H, P = perm[: n // 2], perm[n // 2 :]
            r = attack(part, topics, H, P, n_topics, N_BUCKETS, np.random.default_rng(ks))
            adv.append(r["adv_H"])
            accH.append(r["acc_H"])
            hmp.append(r["H_minus_P"])
        out["splits"][split] = {
            "adv_H_mean": float(np.mean(adv)),
            "adv_H_std": float(np.std(adv)),
            "acc_H_mean": float(np.mean(accH)),
            "H_minus_P_mean": float(np.mean(hmp)),
            "H_minus_P_std": float(np.std(hmp)),
            "chance": 1.0 / n_topics,
        }
    return out


def run_open_world() -> dict:
    """github with the adversary's taxonomy coarser than the user's (sig/* grouped into areas)."""
    texts, topics, _, names = load("github")
    groups = {
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
    coarse_names = sorted(set(groups.values()))
    cidx = {c: i for i, c in enumerate(coarse_names)}
    adv_topic = np.array([cidx[groups[names[t]]] for t in topics])  # what the adversary induces
    vecs = embed_cached(texts)
    km = KMeans(n_clusters=N_BUCKETS, random_state=0, n_init=1).fit(vecs)
    cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    part = np.argmax(vecs @ cents.T, axis=1)
    n = len(texts)
    H, P = np.arange(n // 2), np.arange(n // 2, n)
    # Adversary labels partitions with coarse topics; we score whether it recovers the coarse
    # topic of cohort-H records (the fine topic is not what a coarse adversary can ask).
    r = attack(part, adv_topic, H, P, len(coarse_names), N_BUCKETS, np.random.default_rng(0))
    return {
        "corpus": "github_open_world",
        "n_coarse_topics": len(coarse_names),
        "split": "chronological",
        **r,
    }


def main() -> None:
    t0 = time.time()
    results = {
        "experiment": "E3",
        "n_buckets": N_BUCKETS,
        "budget_per_topic": BUDGET_PER_TOPIC,
        "w": W,
        "corpora": [],
    }
    for name in ("github", "newsgroups"):
        results["corpora"].append(run_corpus(name, shuffled_only=(name == "newsgroups")))
    if (E3 / "enron.jsonl").exists():
        results["corpora"].append(run_corpus("enron"))
    results["open_world"] = run_open_world()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n{'corpus':>16} {'split':>14} {'chance':>7} {'acc_H':>7} {'adv_H':>7} {'H-P':>8}")
    for c in results["corpora"]:
        for split, s in c["splits"].items():
            print(
                f"{c['corpus']:>16} {split:>14} {s['chance']:7.3f} {s['acc_H_mean']:7.3f} "
                f"{s['adv_H_mean']:7.3f} {s['H_minus_P_mean']:8.3f}"
            )
    o = results["open_world"]
    print(
        f"{o['corpus']:>16} {'chronological':>14} {o['chance']:7.3f} {o['acc_H']:7.3f} "
        f"{o['adv_H']:7.3f} {o['H_minus_P']:8.3f}"
    )
    print(f"wrote {OUT.relative_to(C.repo_root())} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
