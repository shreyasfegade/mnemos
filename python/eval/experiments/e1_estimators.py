"""E1: structure transfer on D2 without the attribution oracle.

Setting
=======
An unpadded partitioned store (B = 256 content-assigned partitions in growing regions) holds the
D2 corpus split into cohorts H and P. The adversary induces ``per`` writes of each of the 8 topics.
Each induced record is written in a flush with ``w - 1`` organic records drawn from the rest of
cohort P, and the adversary observes the partitions each flush wrote. It also observes organic
flushes it did not cause, from which it estimates the organic partition distribution.

Estimators: the attribution oracle (upper bound), frequency analysis, and EM (see
:mod:`eval.attacks.estimators`). A fourth row is the zero-budget attack available when the
partition table is public: embed public BEIR queries per topic, assign them to their nearest
centroid, and label partitions by majority.

Scoring
=======
Cohort H and cohort P' (P without the induced records). Accuracy counts an unlabelled partition
as wrong; coverage and purity are reported separately. Every number is raw accuracy minus chance
(no truncation at zero), averaged over k-means seeds x cohort splits x attack trials, with a 95%
percentile bootstrap over the (seed, split) configurations, which are the independent units.

    cd python && python -m eval.experiments.e1_estimators
"""

from __future__ import annotations

import json
import time
import zipfile

import numpy as np
from sklearn.cluster import KMeans

from eval.attacks import estimators as X
from eval.lab import corpus as C
from eval.lab.embcache import device, embed_cached

N_BUCKETS = 256
WS = (1, 16)
BUDGETS = (1, 5, 20, 50, 100, 250)
KMEANS_SEEDS = range(5)
SPLIT_SEEDS = range(6)
TRIALS = 5
ORGANIC_OBS = 4000
N_BOOT = 10_000
AUX_PER_TOPIC = 200

D2 = C.repo_root() / "data" / "d2-beir"
OUT = C.repo_root() / "python" / "eval" / "results" / "e1_estimators.json"


def load_d2() -> tuple[list[str], np.ndarray, list[str]]:
    rows = [
        json.loads(line) for line in (D2 / "notes.jsonl").open(encoding="utf-8") if line.strip()
    ]
    rows.sort(key=lambda r: r["seq"])
    names = sorted({r["topic"] for r in rows})
    idx = {t: i for i, t in enumerate(names)}
    return [r["text"] for r in rows], np.array([idx[r["topic"]] for r in rows]), names


def load_public_queries(names: list[str]) -> tuple[list[str], np.ndarray]:
    """BEIR queries per domain: public text of each topic, disjoint from the corpus."""
    texts, labels = [], []
    rng = np.random.default_rng(20260915)
    for i, name in enumerate(names):
        with zipfile.ZipFile(D2 / "_zips" / f"{name}.zip") as z:
            member = next(m for m in z.namelist() if m.endswith("queries.jsonl"))
            qs = [
                json.loads(line)["text"]
                for line in z.read(member).decode("utf-8").splitlines()
                if line.strip()
            ]
        pick = rng.choice(len(qs), size=min(AUX_PER_TOPIC, len(qs)), replace=False)
        texts.extend(qs[int(j)] for j in pick)
        labels.extend([i] * len(pick))
    return texts, np.array(labels)


def boot(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean and 95% percentile interval over configurations."""
    v = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def main() -> None:
    t0 = time.time()
    texts, topics, names = load_d2()
    n_topics = len(names)
    chance = 1.0 / n_topics
    X_all = embed_cached(texts)
    aux_texts, aux_labels = load_public_queries(names)
    X_aux = embed_cached(aux_texts)
    print(
        f"D2 {len(texts)} records, {n_topics} topics; aux {len(aux_texts)} queries; device "
        f"{device()}",
        flush=True,
    )

    records: list[dict] = []
    for kseed in KMEANS_SEEDS:
        km = KMeans(n_clusters=N_BUCKETS, random_state=kseed, n_init=1).fit(X_all)
        cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        part = np.argmax(X_all @ cents.T, axis=1)
        aux_part = np.argmax(X_aux @ cents.T, axis=1)

        # Zero-budget attack with a public partition table.
        zb = np.full(N_BUCKETS, -1, dtype=np.int64)
        for b in range(N_BUCKETS):
            m = aux_labels[aux_part == b]
            if m.size:
                zb[b] = int(np.bincount(m, minlength=n_topics).argmax())

        for split in SPLIT_SEEDS:
            perm = np.random.default_rng(1000 + split).permutation(len(texts))
            H, P = perm[: len(perm) // 2], perm[len(perm) // 2 :]
            s_zb = X.score(zb, part[H], topics[H])
            records.append(
                {
                    "k": kseed,
                    "split": split,
                    "trial": 0,
                    "estimator": "zero_budget_public_T",
                    "w": 0,
                    "per": 0,
                    "H": s_zb,
                    "P": X.score(zb, part[P], topics[P]),
                }
            )

            for trial in range(TRIALS):
                rng = np.random.default_rng(kseed * 1_000_003 + split * 10_007 + trial)
                for per in BUDGETS:
                    induced = []
                    for t in range(n_topics):
                        pool = P[topics[P] == t]
                        induced.extend(
                            rng.choice(pool, size=min(per, len(pool)), replace=False).tolist()
                        )
                    induced = np.array(induced)
                    mask = np.isin(P, induced, invert=True)
                    P_rest = P[mask]
                    organic_parts = part[P_rest]
                    obs = rng.choice(organic_parts, size=ORGANIC_OBS, replace=True)
                    pi = (np.bincount(obs, minlength=N_BUCKETS) + 1.0) / (ORGANIC_OBS + N_BUCKETS)

                    for w in WS:
                        flushes: X.Flushes = {t: [] for t in range(n_topics)}
                        for r in induced:
                            org = rng.choice(organic_parts, size=w - 1, replace=True)
                            s = np.concatenate([[part[r]], org]).astype(np.int64)
                            rng.shuffle(s)
                            flushes[int(topics[r])].append(s)

                        lab_o = X.oracle_intersection(
                            X.group_by_true_partition(part[induced], topics[induced]),
                            N_BUCKETS,
                            w,
                            organic_parts,
                            rng,
                        )
                        lab_f = X.frequency(flushes, pi, w, N_BUCKETS)
                        lab_e, _ = X.em(flushes, pi, N_BUCKETS)
                        for name, lab in (
                            ("oracle_intersection", lab_o),
                            ("frequency", lab_f),
                            ("em", lab_e),
                        ):
                            records.append(
                                {
                                    "k": kseed,
                                    "split": split,
                                    "trial": trial,
                                    "estimator": name,
                                    "w": w,
                                    "per": per,
                                    "induced": len(induced),
                                    "H": X.score(lab, part[H], topics[H]),
                                    "P": X.score(lab, part[P_rest], topics[P_rest]),
                                }
                            )
        print(f"k-means seed {kseed} done ({time.time() - t0:.0f}s)", flush=True)

    # Aggregate: config-level means over trials, then bootstrap over configurations.
    rng = np.random.default_rng(20260915)
    rows = []
    keys = sorted({(r["estimator"], r["w"], r["per"]) for r in records})
    for est, w, per in keys:
        sel = [r for r in records if (r["estimator"], r["w"], r["per"]) == (est, w, per)]
        cfgs = sorted({(r["k"], r["split"]) for r in sel})

        def cfg_mean(f, sel=sel, cfgs=cfgs):
            return np.array(
                [np.nanmean([f(r) for r in sel if (r["k"], r["split"]) == c]) for c in cfgs]
            )

        accH = cfg_mean(lambda r: r["H"]["accuracy"])
        accP = cfg_mean(lambda r: r["P"]["accuracy"])
        row = {
            "estimator": est,
            "w": w,
            "per_topic": per,
            "induced_writes": per * n_topics,
            "n_configs": len(cfgs),
            "adv_H": boot(accH - chance, rng),
            "adv_P": boot(accP - chance, rng),
            "H_minus_P": boot(accH - accP, rng),
            "coverage_H": boot(cfg_mean(lambda r: r["H"]["coverage"]), rng),
            "purity_H": boot(cfg_mean(lambda r: r["H"]["purity"]), rng),
            "mapped": float(np.mean([r["H"]["mapped"] for r in sel])),
        }
        denom = accP - chance
        if np.all(denom > 0.02):
            row["ratio_H_over_P"] = boot((accH - chance) / denom, rng)
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "experiment": "E1",
                "corpus": "D2",
                "n_buckets": N_BUCKETS,
                "chance": chance,
                "kmeans_seeds": list(KMEANS_SEEDS),
                "split_seeds": list(SPLIT_SEEDS),
                "trials": TRIALS,
                "organic_observations": ORGANIC_OBS,
                "device": device(),
                "rows": rows,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    print(
        f"\n{'estimator':>22} {'w':>3} {'writes':>6} {'adv_H [95% CI]':>24} {'adv_P':>7} "
        f"{'H-P [95% CI]':>24} {'cov':>6} {'purity':>6}"
    )
    for r in rows:
        a, hp = r["adv_H"], r["H_minus_P"]
        print(
            f"{r['estimator']:>22} {r['w']:>3} {r['induced_writes']:>6} "
            f"{a[0]:7.3f} [{a[1]:.3f}, {a[2]:.3f}] {r['adv_P'][0]:7.3f} "
            f"{hp[0]:7.3f} [{hp[1]:.3f}, {hp[2]:.3f}] {r['coverage_H'][0]:6.3f} "
            f"{r['purity_H'][0]:6.3f}"
        )
    print(f"\nwrote {OUT.relative_to(C.repo_root())} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
