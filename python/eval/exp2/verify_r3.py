"""Registered checks for the third review round (the two reader reports of 2026-09-28).

Every check here is committed to git **before** the experiment it binds is run, so the commit
that adds a check is a timestamp a reader can verify against the result file's. A check whose
prediction fails is never deleted: its range is widened only to the claim that survives, and the
failure is recorded in its ``why``, exactly as in ``verify.py``.

Experiment numbering continues from E15: E17 composed attack on Mnemos-P, E20 reactive game,
E21 flush cost, E22 probe frontier, E23 sensitivity, E24 reindex lifecycle, E25 scale,
E26 retrieval breadth, E27 archival alignment, E19 concrete version retention, E18 real
frameworks.
"""

from __future__ import annotations

import json

from eval.lab import corpus as C

R = C.repo_root() / "python" / "eval" / "results"


def _load(name: str):
    f = R / name
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def _check(name: str, claim: str, lo: float, hi: float, why: str, src: str):
    from eval.exp2.verify import Check

    return (Check(name, claim, lo, hi, why), src)


# ----------------------------------------------------------------------------- extractors


def e21() -> dict[str, float]:
    d = _load("e21_flush_cost.json")
    if not d:
        return {}
    out: dict[str, float] = {}
    g = d["geometries"]["shipped"]
    out["E21 sweep KiB per flush, shipped geometry"] = g["sweep"]["bytes_per_flush"] / 1024
    out["E21 region MiB per flush, shipped geometry"] = g["regions"]["bytes_per_flush"] / 2**20
    out["E21 sweep bytes, staged minus empty flush"] = float(
        g["sweep"]["bytes_per_flush_one_staged"] - g["sweep"]["bytes_per_flush"]
    )
    out["E21 sweep GB per day at Delta 10 s, 24 h unlocked"] = g["sweep"]["gb_per_day"]["10.0"]
    out["E21 daily volume ratio, regions at 0.5 s over sweep at 10 s"] = (
        g["regions"]["gb_per_day"]["0.5"] / g["sweep"]["gb_per_day"]["10.0"]
    )
    lk = {(r["rate"], r["load"]): r for r in d["lock_residual"]}
    if (0.1, 0.9) in lk:
        out["E21 lock-residual rate, 0.1 writes per flush, load 0.9"] = lk[(0.1, 0.9)][
            "residual_rate"
        ]
    return out


def e20() -> dict[str, float]:
    d = _load("e20_reactive.json")
    if not d:
        return {}
    out: dict[str, float] = {}
    for r in d["rows"]:
        best = max((r["learned"], r["hand_built"]), key=lambda a: abs(a["auc"] - 0.5))
        out[f"E20 {r['arm']}, stronger attacker AUC"] = best["auc"]
    return out


def e23() -> dict[str, float]:
    d = _load("e23_sensitivity.json")
    if not d:
        return {}
    return {
        "E23 Mnemos-P worst utilisation at first refusal": d["p_util_worst"],
        "E23 Mnemos-M worst utilisation at first refusal": d["m_util_worst"],
    }


def e22() -> dict[str, float]:
    d = _load("e22_probe_frontier.json")
    if not d:
        return {}
    sp = {r["spread"]: r for r in d.get("spread", [])}
    gr = {r["group_size"]: r for r in d.get("grouped", [])}
    out: dict[str, float] = {}
    if 2 in sp and 8 in sp:
        out["E22 spread J drop, s2 minus s8"] = sp[2]["threshold_j"] - sp[8]["threshold_j"]
    if 2 in gr and 32 in gr:
        out["E22 grouped J drop, g2 minus g32"] = gr[2]["threshold_j"] - gr[32]["threshold_j"]
    return out


def e18() -> dict[str, float]:
    d = _load("e18_real_mem0.json")
    if not d:
        return {}
    return {
        "E18 real mem0 dedup oracle AUC": d["auc_write_vs_nonmember"],
        "E18 real mem0 dedup oracle AUC lower": d["lo"],
        # Non-member minus member: a member is deduplicated and writes less, so the gap is positive.
        "E18 nonmember minus member write rate": d["nonmember_write_rate"] - d["member_write_rate"],
    }


def e16() -> dict[str, float]:
    d = _load("e16_real_layouts.json")
    if not d:
        return {}
    out: dict[str, float] = {}
    for s in d.get("stores", []):
        out[f"E16 {s['store']} adv_hist"] = s["adv_hist"]
        out[f"E16 {s['store']} adv_hist lower"] = s["adv_hist_lo"]
    return out


def e17() -> dict[str, float]:
    d = _load("e17_composed.json")
    if not d:
        return {}
    out: dict[str, float] = {"E17 oracle adv_H": d["oracle"]["adv_H"]}
    by = {r["granularity_flushes"]: r for r in d["composed"]}
    for g in (1, 16):
        if g in by:
            out[f"E17 composed adv_H, g={g}"] = by[g]["adv_H"]
            out[f"E17 composed adv_H lower, g={g}"] = by[g]["adv_H_lo"]
    if 1 in by:
        out["E17 oracle minus composed, g=1"] = d["oracle"]["adv_H"] - by[1]["adv_H"]
    return out


def e19() -> dict[str, float]:
    d = _load("e19_restic.json")
    if not d:
        return {}
    rows = {(r["kind"], r["granularity_flushes"]): r for r in d["rows"]}
    out = {}
    if ("p", 1) in rows:
        out["E19 Mnemos-P precision through restic-restored versions, g=1"] = rows[("p", 1)][
            "precision"
        ]
        out["E19 Mnemos-P restic upload over bytes rewritten, g=1"] = rows[("p", 1)][
            "upload_amplification"
        ]
    return out


def e25() -> dict[str, float]:
    d = _load("e25_scale.json")
    if not d:
        return {}
    rows = {(r["engine"], r["n"]): r for r in d["rows"] if "error" not in r}
    out = {}
    disk, mem = rows.get(("m-disk", 100_000)), rows.get(("m-sweep-ivf", 100_000))
    if disk and mem:
        out["E25 M/disk resident over M resident, n=100k"] = (
            disk["resident_bytes"] / mem["resident_bytes"]
        )
    if disk:
        out["E25 M/disk search p50 ms, n=100k"] = disk["search_ms"]["p50"]
    return out


def e27() -> dict[str, float]:
    d = _load("e27_archival_mix.json")
    if not d:
        return {}
    rows = {r["alpha"]: r["adv_H"] for r in d["rows"]}
    if 0.25 in rows and 0.0 in rows:
        return {"E27 advantage at alpha=0.25 minus alpha=0": rows[0.25] - rows[0.0]}
    return {}


def e26() -> dict[str, float]:
    d = _load("e26_second_model.json")
    if not d:
        return {}
    rows = {r["domain"]: r for r in d["domains"]}
    sf = rows.get("scifact")
    if not sf:
        return {}
    return {
        "E26 nomic SciFact: quantisation cost": sf["plaintext_exhaustive"]["ndcg"]
        - sf["m_int8_exhaustive"]["ndcg"],
        "E26 nomic SciFact: M ivf k8 minus P s=2": sf["m_ivf_k8"]["ndcg"] - sf["p_s2"]["ndcg"],
    }


EXTRACTORS = {
    "e26": e26,
    "e27": e27,
    "e25": e25,
    "e21": e21,
    "e20": e20,
    "e17": e17,
    "e16": e16,
    "e18": e18,
    "e22": e22,
    "e23": e23,
    "e19": e19,
}


# ----------------------------------------------------------------------------- checks

CHECKS = [
    # --- E21: what the clocked flush costs (registered 2026-09-28, before the run) -------------
    _check(
        "E21 sweep KiB per flush, shipped geometry",
        "sec:eval-perf -- the sweep flush writes tens of kilobytes, not megabytes",
        55.0,
        85.0,
        "Predicted from the geometry: 8 slots x 4,964 B = 39.7 KB, a 4-entry journal of "
        "~20 KB, and a manifest of 2 KiB bitmap + 512 x 16 B tombstones ~ 10 KB. About 68 KiB.",
        "e21",
    ),
    _check(
        "E21 region MiB per flush, shipped geometry",
        "sec:eval-perf -- the earlier region flush wrote megabytes every Delta",
        4.5,
        6.5,
        "17 whole regions x 64 slots x 4,964 B = 5.4 MB plus a 4,096-tombstone manifest of "
        "66 KB: about 5.2 MiB. This is the number a reader computed from Table 5.",
        "e21",
    ),
    _check(
        "E21 sweep bytes, staged minus empty flush",
        "sec:mnemos-m -- a flush writes the same whether or not the agent wrote",
        0.0,
        0.0,
        "Exactly zero: the slab range, the journal and the manifest are fixed-length.",
        "e21",
    ),
    _check(
        "E21 sweep GB per day at Delta 10 s, 24 h unlocked",
        "sec:eval-perf -- the shipped clock costs well under a gigabyte a day",
        0.4,
        0.9,
        "8,640 flushes a day at ~70 KB each is ~0.6 GB, and a store is rarely unlocked all day.",
        "e21",
    ),
    _check(
        "E21 daily volume ratio, regions at 0.5 s over sweep at 10 s",
        "sec:eval-perf -- the revision cuts the flush volume by three orders of magnitude",
        800.0,
        3000.0,
        "~5.4 MB x 172,800 flushes = ~930 GB/day against ~0.6 GB/day: a ratio near 1,500.",
        "e21",
    ),
    _check(
        "E21 lock-residual rate, 0.1 writes per flush, load 0.9",
        "sec:mnemos-m -- the sweep's one residual (backlog above the journal at lock) is rare at "
        "agent write rates",
        0.0,
        0.01,
        "At 0.1 writes per flush the sweep places ~0.8 records per flush at 90% load (8 slots x "
        "10% free), so a backlog above 4 at lock should essentially never happen.",
        "e21",
    ),
    # --- E20: the reactive (feedback-agent) game (registered 2026-09-28, before the run) --------
    _check(
        "E20 M sweep, stronger attacker AUC",
        "sec:eval-reactive -- Theorem 2: clock-only store at chance vs a feedback agent",
        0.44,
        0.56,
        "Every flush writes the same slots, journal and manifest whatever the agent decided, so "
        "no statistic of the trace depends on whether the dedup agent wrote.",
        "e20",
    ),
    _check(
        "E20 M regions, stronger attacker AUC",
        "sec:eval-reactive -- the earlier clocked region flush is also occurrence-hiding",
        0.44,
        0.56,
        "Clocked, w+1 uniform regions per flush: a placed probe lands in a uniform region.",
        "e20",
    ),
    _check(
        "E20 M flush on write, stronger attacker AUC",
        "sec:eval-reactive -- removing the clock re-opens the derived-write channel",
        0.90,
        1.00,
        "With flushes triggered by writes, a flush occurs iff the agent found the probe absent.",
        "e20",
    ),
    _check(
        "E20 M content placement, stronger attacker AUC",
        "sec:eval-reactive -- content placement re-opens it even under a clock",
        0.75,
        1.00,
        "A written probe lands in its content region, which the adversary computes from the "
        "public table; padding is uniform.",
        "e20",
    ),
    _check(
        "E20 P shipped, stronger attacker AUC",
        "sec:eval-reactive -- Mnemos-P is dual-epoch secure (Theorem 1) and not reactively secure",
        0.75,
        1.00,
        "P flushes on demand, so whether a flush happens depends on the agent's decision, and a "
        "written probe goes to its (permuted) content region.",
        "e20",
    ),
    # --- E17: the composed attack on Mnemos-P (registered 2026-09-30, before the run) -----------
    _check(
        "E17 oracle adv_H",
        "sec:eval-multisnapshot -- Mnemos-P keeps cohort H at chance even with oracle membership",
        -0.05,
        0.05,
        "This is the check that survived the run. On agent memories the induced-label map does not "
        "localize topics on the sealed, padded Mnemos-P layout (B=64, 10 topics, w+1 regions per "
        "flush), so even an adversary handed each record's exact region recovers its topic only "
        "at chance, consistent with E2's Mnemos-P rows. Measured -0.003.",
        "e17",
    ),
    _check(
        "E17 oracle minus composed, g=1",
        "sec:eval-multisnapshot -- diff-derived membership is no stronger than the oracle",
        -0.02,
        1.00,
        "The oracle knows the exact region; the composed attack chooses among ~5 diff candidates, "
        "so it cannot beat the oracle. Measured +0.032 (both near chance). A clearly negative "
        "value would signal a bug; a small negative tolerance absorbs noise at chance.",
        "e17",
    ),
    _check(
        "E17 composed adv_H, g=1",
        "sec:eval-multisnapshot -- the composed attack does not reach cohort H on Mnemos-P",
        -0.15,
        0.05,
        "A FAILED PREDICTION IS RECORDED HERE. Registered as composed-lower in [0.005, 1.00], "
        "expecting the diff-derived composed attack to reach cohort H; it measured adv -0.035 at "
        "g=1 and -0.096 at g=16. Mnemos-P defeats the composition: cohort H stays at chance, and "
        "under coarse membership the vote over a weak committed label map drifts below it "
        "(the anti-learning of E4's far-era arm). The surviving claim: the multi-version plus "
        "induced-label composition does not reach cohort H on the sealed design.",
        "e17",
    ),
    # --- E16: real content-placed layouts under gocryptfs (registered 2026-10-01, before the run) -
    _check(
        "E16 pgvector adv_hist lower",
        "sec:eval-real -- pgvector ivfflat places inserts by content into list pages, so the "
        "structure-transfer attack reaches cohort H on a real, deployed on-disk store",
        0.0,
        1.00,
        "pgvector's ivfflat index assigns an insert to its nearest list and writes that list's "
        "heap and index pages; induced writes should localize to the list pages their topic "
        "selects, and the lemma transfers the labels to cohort H. The prediction is a detectable "
        "channel: the held-out advantage interval lower end clears zero. If it does not, if WAL "
        "and heap churn swamp the list-page signal at 4 KiB -- that null is recorded, and the "
        "growing layout remains the clearest demonstration, which Table 6 and the abstract allow.",
        "e16",
    ),
    # --- E18: dedup oracle vs real mem0 2.2.1 (registered 2026-10-01, before the run) ---
    _check(
        "E18 nonmember minus member write rate",
        "sec:eval-dedup -- against the real mem0 library a member fact is written less than a "
        "non-member, so the occurrence of a vector-store write is a membership signal",
        0.05,
        1.00,
        "mem0's LLM add/skip stage deduplicates a fact already in memory, so a member probe (an "
        "exact repeat of a seeded fact) should produce a vector-store write less often than a "
        "non-member. The write-rate gap is the oracle's signal at the file level; predict it is "
        "positive. mem0's LLM dedup is lenient on paraphrases, so the gap may be modest (like "
        "E3's LLM arm, AUC ~0.62), but its sign is the claim. A non-positive gap is recorded.",
        "e18",
    ),
    # --- E23: capacity/refusal sensitivity (registered 2026-10-01, before the run) ---
    _check(
        "E23 Mnemos-P worst utilisation at first refusal",
        "sec:eval-refusal -- under skew Mnemos-P leaves the admissible regime well before full",
        0.0,
        0.60,
        "The capacity trade-off is geometry- and skew-dependent; across B in {64,128,256}, C in "
        "{16,32,64} and Zipf a in {0.8,1.0,1.2} the worst-case utilisation at the first refusal "
        "should sit well below 1.0 -- the point of M9. Predict the worst case is under 0.60.",
        "e23",
    ),
    _check(
        "E23 Mnemos-M worst utilisation at first refusal",
        "sec:eval-refusal -- Mnemos-M's global-only bound admits essentially every slot",
        0.95,
        1.00,
        "Mnemos-M has no per-partition capacity, so it refuses only when the whole store is full; "
        "across the same grid its utilisation at first refusal should be ~1.0 regardless of skew.",
        "e23",
    ),
    # --- E22: probe-design frontier (registered 2026-10-01, before the run) ---
    _check(
        "E22 spread J drop, s2 minus s8",
        "sec:eval-query -- more random cover (larger s) weakens the overlap-threshold test",
        0.0,
        1.00,
        "Each query reads s*k regions, (s-1)*k fresh random cover, so the expected overlap "
        "between two equal queries falls as s grows (the gap formula k(1-(2s-1)k/B)); the held-out "
        "query-equality J at s=8 should fall below s=2. The frontier shape is the deliverable; "
        "check fixes its direction.",
        "e22",
    ),
    _check(
        "E22 grouped J drop, g2 minus g32",
        "sec:eval-query -- a larger grouped anonymity set weakens query-equality recognition",
        0.0,
        1.00,
        "Grouped probing reads one whole group, so two distinct queries collide when they share a "
        "group, with probability ~group_size/B; a larger group means fewer groups, more "
        "a higher false-positive rate, and a lower J. Predict J at g=32 below g=2.",
        "e22",
    ),
    # --- E19: the multi-version channel through restic (registered 2026-10-01, before the run) ---
    _check(
        "E19 Mnemos-P precision through restic-restored versions, g=1",
        "sec:eval-multisnapshot -- a real backup tool keeps versions the region diff can use",
        0.05,
        1.00,
        "restic restores byte-exact files, so a key holder diffing consecutive restored snapshots "
        "should reproduce E8's per-flush precision (0.21 at B=64, null 1/64 = 0.016). Predict at "
        "least 0.05: well above the null. A lower value would mean the tool's versioning blunts "
        "the channel, and is recorded.",
        "e19",
    ),
    _check(
        "E19 Mnemos-P restic upload over bytes rewritten, g=1",
        "sec:eval-perf -- content-defined chunking uploads whole chunks around each rewrite",
        1.0,
        1000.0,
        "restic chunks at ~1 MiB (min 512 KiB) and the store's ciphertext is incompressible, so a "
        "snapshot cannot add fewer bytes than the flush rewrote; predict amplification >= 1.",
        "e19",
    ),
    # --- E25: Mnemos-M/disk at scale (registered 2026-10-01, before the run) ---
    _check(
        "E25 M/disk resident over M resident, n=100k",
        "sec:mnemos-m -- M/disk keeps only a position map and partition ids resident",
        0.0,
        0.05,
        "Mnemos-M keeps every header and payload (~1.8 KB per slot at E10's geometry) in the "
        "arena; M/disk keeps only a few bytes of partition id per slot. Predict the ratio of "
        "resident bytes under 5%.",
        "e25",
    ),
    _check(
        "E25 M/disk search p50 ms, n=100k",
        "sec:eval-perf -- M/disk search stays interactive at 100k records",
        0.0,
        50.0,
        "A search reads and decrypts the headers of its 8 probed logical partitions, about 1,600 "
        "at n=100k and B=512, mostly from the page cache. Predict a median under 50 ms; a slower "
        "median is recorded as the price of leaving the memory limit.",
        "e25",
    ),
    # --- E27: archival alignment (registered 2026-10-01, before the run) ---
    _check(
        "E27 advantage at alpha=0.25 minus alpha=0",
        "sec:eval-drift -- partial archival knowledge already recovers part of drift's protection",
        0.0,
        1.0,
        "E4 shows full archival induction recovers the advantage drift removes. If the recovery "
        "came only from an adversary with complete historical alignment it would be a weak threat; "
        "predict that a quarter of the budget drawn from the archive already raises the advantage "
        "on cohort H above the purely current adversary's.",
        "e27",
    ),
    # --- E26: a second embedding model (registered 2026-10-01, before the run) ---
    _check(
        "E26 nomic SciFact: quantisation cost",
        "sec:eval-recall -- int8 quantisation is nearly free with a second embedding model too",
        -0.02,
        0.02,
        "With bge, int8 costs at most a few thousandths of nDCG@10. Quantisation error is a "
        "property of the vector norms, not of one model, so predict |cost| under 0.02 with "
        "nomic-embed-text on SciFact.",
        "e26",
    ),
    _check(
        "E26 nomic SciFact: M ivf k8 minus P s=2",
        "sec:eval-recall -- the in-memory index beats Mnemos-P at matched work, model-independent",
        0.0,
        1.0,
        "Mnemos-M's eight in-memory partitions are all the query's nearest, while half of "
        "Mnemos-P's eight at s=2 are random cover; predict M ivf k8 >= P s=2 on nDCG@10.",
        "e26",
    ),
]
