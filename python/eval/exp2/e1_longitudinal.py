"""E1: a longitudinal case study, and the paper's headline comparison.

One figure is needed that puts the whole argument in one place: a real stream of
timestamped work, processed month by month, compromised partway through, and the exposure of the
records written *before* that moment measured for each design and each adversary class.

The stream
==========
``kubernetes`` issues in their real creation order, bucketed into calendar months. The agent's
memories are the ones a local model actually wrote, if ``data/e3/agent_memories.jsonl`` exists
(produced by ``e2_agent_attack --generate``); otherwise the issue text stands in, and the run says
which it used. τ falls at a month boundary: cohort H is everything written before it.

The designs
===========
``growing``  a content-placed layout whose partition regions grow with occupancy -- the model of
             the deployed stores, whose real behaviour E6 measures under gocryptfs. Its snapshot
             reveals the partition of every stored record.
``P``        Mnemos-P, the sealed partitioned file.
``M``        Mnemos-M, memory-resident with content-oblivious placement.

The adversary classes
=====================
``W``   the write observer: a sync account, a backup, repeated imaging. Sees file versions, so it
        sees which regions a flush wrote. It does **not** see reads, which is why the search-side
        results in the paper do not apply to it.
``RW``  the read-write observer: a network share or a storage component that sees I/O but not
        process memory. Sees reads as well, and for Mnemos-P that is enough to recover the keyed
        permutation from probe co-occurrence alone (E7), which hands it region labels without
        inducing a single write.

    cd python && python -m eval.exp2.e1_longitudinal
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.attacks import estimators as X
from eval.exp2 import lab
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e1_longitudinal.json"
MEMS = C.repo_root() / "data" / "e3" / "agent_memories.jsonl"
GITHUB = C.repo_root() / "data" / "e3" / "github.jsonl"

B = 64
CAP = 32
DIM = 768
BLOCK = 256
W = 4
BUDGETS = (4, 16, 64)
TRIALS = 20

# Mnemos-P's probe spread in this experiment. The RW adversary's permutation recovery is granted
# at the rate E7 measured for this spread, so the two must agree.
SPREAD = 2


class _E7Recovery(dict):
    """E7's measured exact-recovery rate per probe spread, read from its result file on first use.

    Read rather than typed in, so E1 cannot drift from the experiment it borrows a number from.
    """

    def __missing__(self, spread: int) -> float:
        path = C.repo_root() / "python" / "eval" / "results" / "e7_permutation.json"
        if not path.exists():
            raise SystemExit(
                "E1's RW arm needs E7's measured recovery rate; run e7_permutation first"
            )
        modes = {
            m["mode"]: m["exact_mean"]
            for m in json.loads(path.read_text(encoding="utf-8"))["modes"]
        }
        key = f"spread_s{spread}"
        if key not in modes:
            raise SystemExit(f"E7 has no measurement for {key}")
        self[spread] = float(modes[key])
        return self[spread]


E7_RECOVERY = _E7Recovery()
MAX_RECORDS = 4000


def load_stream() -> tuple[list[str], np.ndarray, list[str], np.ndarray, str]:
    """Texts, topic ids, topic names, month index, and the provenance of the stream."""
    if MEMS.exists():
        rows = [json.loads(line) for line in MEMS.open(encoding="utf-8") if line.strip()]
        rows = [r for r in rows if "ts" in r]
        if rows:
            rows.sort(key=lambda r: r["ts"])
            source = "agent-authored memories"
        else:
            rows = []
    else:
        rows = []
    if not rows:
        rows = [json.loads(line) for line in GITHUB.open(encoding="utf-8") if line.strip()]
        rows.sort(key=lambda r: r["ts"])
        source = "issue text (agent memories not generated)"

    # Cap per topic so a few huge areas do not dominate, then restore chronological order.
    names = sorted({r["topic"] for r in rows})
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_topic[r["topic"]].append(r)
    cap = max(40, MAX_RECORDS // len(names))
    kept: list[dict] = []
    for t in names:
        kept.extend(by_topic[t][:cap])
    kept.sort(key=lambda r: r["ts"])

    idx = {t: i for i, t in enumerate(names)}
    texts = [r["text"][:1500] for r in kept]
    topics = np.array([idx[r["topic"]] for r in kept])
    months = np.array([str(r["ts"])[:7] for r in kept])
    uniq = sorted(set(months.tolist()))
    month_idx = np.array([uniq.index(m) for m in months])
    return texts, topics, names, month_idx, source


def frequency_labels(by_topic: dict[int, list[list[int]]], n_regions: int) -> dict[int, int]:
    topics = [t for t, f in by_topic.items() if f]
    if not topics:
        return {}
    rate = np.zeros((len(topics), n_regions))
    for i, t in enumerate(topics):
        for f in by_topic[t]:
            for r in set(f):
                if 0 <= r < n_regions:
                    rate[i, r] += 1
        rate[i] /= max(len(by_topic[t]), 1)
    mean = rate.mean(0)
    n_min = min(len(by_topic[t]) for t in topics)
    q = float(mean.mean())
    margin = 2.0 * float(np.sqrt(max(q * (1 - q), 1e-9) / max(n_min, 1)))
    out: dict[int, int] = {}
    for r in range(n_regions):
        best = int(np.argmax(rate[:, r]))
        if rate[best, r] - mean[r] > margin:
            out[r] = topics[best]
    return out


def read_flushes(trace: Path, offset: int) -> tuple[list[list[int]], int]:
    if not trace.exists():
        return [], offset
    with trace.open("r", encoding="utf-8") as fh:
        fh.seek(offset)
        lines = [line for line in fh if line.strip()]
    new_offset = trace.stat().st_size
    flushes, pending = [], []
    for line in lines:
        e = json.loads(line)
        if e["kind"] == "region_write":
            pending.append(int(e["region"]))
        elif e["kind"] == "index_write":
            # Mnemos-RS: the index bucket a flush wrote, dummies included, which is its padding.
            pending.append(int(e["bucket"]))
        elif e["kind"] == "epoch_flush":
            flushes.append(pending)
            pending = []
    if pending:
        flushes.append(pending)
    return flushes, new_offset


def run_growing(vecs, topics, n_topics, cents, H, P, budget, rng) -> dict:
    """The deployed-store model: placement is content, and the snapshot shows it."""
    part = np.argmax(vecs @ cents.T, axis=1)
    by_topic: dict[int, list[list[int]]] = {t: [] for t in range(n_topics)}
    pool_by_topic: dict[int, list[int]] = defaultdict(list)
    for i in P:
        pool_by_topic[int(topics[i])].append(int(i))
    for t in range(n_topics):
        pool = pool_by_topic.get(t, [])
        picks = rng.choice(pool, size=min(budget, len(pool)), replace=False) if pool else []
        for i in picks:
            # An unpadded growing layout writes only the record's own region, so the flush names
            # exactly its partition.
            by_topic[t].append([int(part[i])])
    labels = frequency_labels(by_topic, B)
    hits = sum(1 for i in H if labels.get(int(part[i])) == int(topics[i]))
    unlabelled = sum(1 for i in H if int(part[i]) not in labels)
    acc = X.guess_accuracy(hits, unlabelled, len(H), n_topics)
    pred = [labels.get(int(part[i]), -1) for i in H]
    return {
        "acc_H": acc,
        "adv_H": acc - 1.0 / n_topics,
        "adv_H_bal": X.balanced_guess_accuracy(pred, topics[H], n_topics) - 1.0 / n_topics,
        "regions_labelled": len(labels),
        "coverage_H": 1 - unlabelled / max(len(H), 1),
        "membership": "disclosed by the snapshot",
    }


def run_engine(kind, vecs, topics, n_topics, cents, H, P, budget, adversary, rng) -> dict:
    tmp = Path(tempfile.mkdtemp())
    try:
        over = lab.base_overrides(
            dim=DIM, buckets=B, capacity=CAP, block=BLOCK, epoch_writes=W, probe_spread=SPREAD
        )
        if kind == "rs":
            # RS caps live entries globally (with a load-factor margin), not per bucket, so size
            # it for cohort H plus every induced write. Bucket choice is still content, which is
            # the channel under test; only the refusal point moves.
            need = len(H) + budget * n_topics
            n_max = 1 << int(np.ceil(np.log2(2 * need)))
            over = lab.merge(over, {"index": {"n_max": n_max, "overflow_depth": 2}})
        eng, _, _ = lab.create(kind, tmp / "s", cents, over, {})

        # Keep the id of every admitted cohort-H record beside its topic. Mnemos-M places at
        # random, so where a record ended up can only be *asked for*, never recomputed from its
        # content -- which is the whole point of the design and the reason `actual_region` exists.
        hist: list[tuple[str, int]] = []
        for i in H:
            try:
                mid = eng.put(f"h{i}", vecs[int(i)].tolist(), "{}")
                hist.append((mid, int(topics[int(i)])))
            except Exception:
                pass
        lab.drain(eng)
        eng.flush()

        trace = tmp / "t.jsonl"
        trace.write_text("", encoding="utf-8")
        eng.set_trace_sink(str(trace))
        offset = 0

        pool_by_topic: dict[int, list[int]] = defaultdict(list)
        for i in P:
            pool_by_topic[int(topics[i])].append(int(i))

        by_topic: dict[int, list[list[int]]] = {t: [] for t in range(n_topics)}
        for t in range(n_topics):
            pool = pool_by_topic.get(t, [])
            picks = rng.choice(pool, size=min(budget, len(pool)), replace=False) if pool else []
            for i in picks:
                with contextlib.suppress(Exception):
                    eng.put(f"p{int(i)}", vecs[int(i)].tolist(), "{}")
                eng.flush()
                fl, offset = read_flushes(trace, offset)
                for f in fl:
                    by_topic[t].append(f)

        labels = frequency_labels(by_topic, B)

        # The RW adversary additionally recovers the permutation from probe co-occurrence, which
        # for Mnemos-P gives it region labels from a public table with no induced writes at all.
        #
        # Two things the adversary does NOT get, both of which an earlier form of this code handed
        # it. (1) The labels of the public table come from public text only -- here cohort P, the
        # half the adversary can read or induce -- never from cohort H's own records and true
        # topics, which would score the adversary against labels derived from the answers.
        # (2) The permutation is granted at the strength E7 *measured* for this probe spread, not
        # exactly: each logical partition's region is recovered with that probability and is
        # otherwise a wrong region. Together they turned a deterministic +0.374 into a measurement.
        if adversary == "RW" and kind == "p":
            perm = np.array(eng.ground_truth_permutation())
            recovered = perm.copy()
            wrong = rng.random(B) >= E7_RECOVERY[SPREAD]
            for logical in np.where(wrong)[0]:
                recovered[logical] = int(rng.choice([r for r in range(B) if r != perm[logical]]))
            content = np.argmax(vecs @ cents.T, axis=1)
            public = np.asarray(P, dtype=int)
            for logical in range(B):
                members = public[content[public] == logical]
                if len(members):
                    t = int(np.bincount(topics[members], minlength=n_topics).argmax())
                    labels[int(recovered[logical])] = t

        # Ground truth: the region each cohort-H record actually occupies. For Mnemos-P that is a
        # function of content, so it can be computed; for Mnemos-M it is a draw the engine made,
        # so it is read back per record. Either way the adversary never sees it -- it is used only
        # to score the labels the adversary produced.
        if kind == "m":
            # Random placement: the region is a draw the engine made, so read it back per record.
            placed = [(eng.actual_region(mid), topic) for mid, topic in hist]
            pairs = [(int(r), t) for r, t in placed if r is not None]
        elif kind == "rs":
            # Mnemos-RS: found by the record's tag in the index, so quantised assignment and
            # spills out of full buckets are both where they really are.
            placed = [(eng.actual_bucket(mid), topic) for mid, topic in hist]
            pairs = [(int(r), t) for r, t in placed if r is not None]
        else:
            # Content placement: the region is a function of the record, so compute it.
            pairs = [
                (int(eng.ground_truth_region(vecs[int(i)].tolist())), int(topics[int(i)]))
                for i in H
            ]
        hits = sum(1 for region, topic in pairs if labels.get(region, -1) == topic)
        scored = len(pairs)
        unlabelled = sum(1 for region, _ in pairs if region not in labels)
        # The adversary guesses where it has no label (estimators.score explains why).
        acc = X.guess_accuracy(hits, unlabelled, scored, n_topics)
        with contextlib.suppress(Exception):
            eng.close()
        # For the sealed designs the harness supplies each cohort-H record's region to score the
        # labels. The adversary does not have it -- a sealed layout does not disclose it, which is
        # the third hypothesis of the transfer lemma -- so these rows state what the labels would
        # be worth to an adversary that obtained membership elsewhere (E8: version diffs).
        bal = X.balanced_guess_accuracy(
            [labels.get(region, -1) for region, _ in pairs], [t for _, t in pairs], n_topics
        )
        return {
            "acc_H": acc,
            "adv_H": acc - 1.0 / n_topics,
            "adv_H_bal": bal - 1.0 / n_topics,
            "regions_labelled": len(labels),
            "scored": scored,
            "membership": "supplied by the harness",
            "permutation_recovery": E7_RECOVERY[SPREAD]
            if (adversary == "RW" and kind == "p")
            else None,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--tau-month", type=float, default=0.5, help="fraction of months before tau")
    ap.add_argument(
        "--designs",
        nargs="*",
        default=None,
        help="run only these designs and merge them into the existing result file",
    )
    args = ap.parse_args()
    t0 = time.time()

    import mnemos_native as native

    # The RS arm scores against Engine.actual_bucket, which an extension built before it existed
    # does not have. Skip it by name rather than crash twenty trials into the run; it can be added
    # afterwards with --designs rs, which merges into this file.
    rs_supported = hasattr(native.Engine, "actual_bucket")
    if not rs_supported:
        print(
            "RS arm skipped: this mnemos_native predates Engine.actual_bucket. Rebuild the "
            "extension, then run with --designs rs to add it.",
            flush=True,
        )

    texts, topics, names, month_idx, source = load_stream()
    n_topics = len(names)
    vecs = embed_cached(texts)
    cents = lab.kmeans_centroids(vecs, B, seed=0)
    n_months = int(month_idx.max()) + 1
    tau_month = round(n_months * args.tau_month)
    H = np.where(month_idx < tau_month)[0]
    P = np.where(month_idx >= tau_month)[0]
    chance = 1.0 / n_topics
    print(
        f"stream: {source}; {len(texts)} records over {n_months} months, {n_topics} topics.\n"
        f"tau at month {tau_month}: cohort H = {len(H)} records, cohort P = {len(P)}. "
        f"chance {chance:.3f}",
        flush=True,
    )

    results = {
        "experiment": "E1",
        "stream_source": source,
        "n_records": len(texts),
        "n_months": n_months,
        "tau_month": tau_month,
        "n_hist": len(H),
        "n_post": len(P),
        "n_topics": n_topics,
        "chance": chance,
        "buckets": B,
        "rows": [],
    }
    # A subset run keeps every other design's rows rather than overwriting them, but only rows
    # measured at the same number of trials: rows at different power are not mixed in one table.
    if args.designs is not None and OUT.exists():
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        kept = [r for r in prev.get("rows", []) if r["design"] not in args.designs]
        if any(r.get("trials") != args.trials for r in kept):
            raise SystemExit(f"{OUT.name} holds rows at a different trial count; refusing to merge")
        results["rows"] = kept

    for budget in BUDGETS:
        for design, adversary in (
            ("growing", "W"),
            ("growing", "RW"),
            ("p", "W"),
            ("p", "RW"),
            ("m", "W"),
            ("m", "RW"),
            # Mnemos-RS for the write observer: it is the third design. Its read-side
            # channel is result-dependent reads (sec:eval-search), measured there, not here.
            ("rs", "W"),
        ):
            if args.designs is not None and design not in args.designs:
                continue
            if design == "rs" and not rs_supported:
                continue
            advs = []
            bals = []
            r: dict = {}
            for tr in range(args.trials):
                rng = np.random.default_rng(5000 + tr)
                if design == "growing":
                    r = run_growing(vecs, topics, n_topics, cents, H, P, budget, rng)
                else:
                    r = run_engine(
                        design, vecs, topics, n_topics, cents, H, P, budget, adversary, rng
                    )
                advs.append(r["adv_H"])
                bals.append(r["adv_H_bal"])
            m, lo, hi = lab.boot_ci(advs)
            bm, blo, bhi = lab.boot_ci(bals)
            row = {
                "design": design,
                "adversary": adversary,
                "budget": budget,
                "adv_H": m,
                "lo": lo,
                "hi": hi,
                # advantage in balanced accuracy (mean per-topic recall), which the topic prior
                # alone cannot raise; see estimators.balanced_guess_accuracy
                "adv_H_bal": bm,
                "lo_bal": blo,
                "hi_bal": bhi,
                "trials": args.trials,
                "membership": r.get("membership"),
                "permutation_recovery": r.get("permutation_recovery"),
            }
            results["rows"].append(row)
            print(
                f"budget={budget:>3} {design:>7} / {adversary:<2}: adv_H {m:+.3f} [{lo:+.3f}, "
                f"{hi:+.3f}]",
                flush=True,
            )
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"wrote {OUT.relative_to(C.repo_root())} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
