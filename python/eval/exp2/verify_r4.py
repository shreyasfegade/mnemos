"""Registered checks for the fourth round (pre-submission, 2026-10-13).

Committed to git **before** any of these experiments ran; the commit that adds a check is its
timestamp. As in ``verify_r3``, a failed prediction is never deleted: its range is widened only to
the claim that survives and the failure is recorded in its ``why``.

E16 extension -- pgvector at 20 seeds instead of 3 (the abstract's deployed-store claim).
E29 -- mem0 2.2.1 unmodified, on a Mnemos backend; the oracle with and without mem0's history file.
E30 -- the oracle in a second framework, LangMem, end to end.
E31 -- a second model deciding (llama3.1:8b), and amplification by sequential re-probing.
E32 -- pgvector at capture granularities g = 1, 4, 16.
E33 -- Syncthing's staggered versioning as a real version-retention policy.
"""

from __future__ import annotations

import json

from eval.lab import corpus as C

R = C.repo_root() / "python" / "eval" / "results"


def _load(name: str):
    f = R / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _check(name: str, claim: str, lo: float, hi: float, why: str, src: str):
    from eval.exp2.verify import Check

    return (Check(name, claim, lo, hi, why), src)


# ----------------------------------------------------------------------------- extractors


def e16x() -> dict[str, float]:
    d = _load("e16_real_layouts.json")
    pg = next((s for s in (d or {}).get("stores", []) if s["store"] == "pgvector"), None)
    if not pg or pg["seeds"] < 20:
        return {}
    return {
        "E16x pgvector adv_hist, 20 seeds": pg["adv_hist"],
        "E16x pgvector adv_hist lower, 20 seeds": pg["adv_hist_lo"],
    }


def e29() -> dict[str, float]:
    d = _load("e29_mem0_mnemos.json")
    arms = (d or {}).get("arms", {})
    out: dict[str, float] = {}
    names = {
        "qdrant": "E29 mem0 on Qdrant: oracle AUC",
        "m_store": "E29 mem0 on Mnemos-M, store observed: oracle AUC",
        "m_all": "E29 mem0 on Mnemos-M, store and history observed: oracle AUC",
        "p_store": "E29 mem0 on Mnemos-P, store observed: oracle AUC",
    }
    for arm, name in names.items():
        if arm in arms:
            out[name] = arms[arm]["auc_blocks"]
    if "m_store" in arms:
        out["E29 mem0 on Mnemos-M, store observed: AUC upper"] = arms["m_store"]["hi_blocks"]
    return out


def e30() -> dict[str, float]:
    d = _load("e30_langmem.json")
    if not d:
        return {}
    return {
        "E30 LangMem oracle AUC": d["auc_any"],
        "E30 LangMem oracle AUC, bytes appended": d["auc_bytes"],
        "E30 LangMem non-member minus member write rate": d["nonmember_write_rate"]
        - d["member_write_rate"],
    }


def e31() -> dict[str, float]:
    d = _load("e31_amplify.json")
    models = (d or {}).get("models", {})
    out: dict[str, float] = {}
    if "llama3.1:8b" in models:
        out["E31 llama3.1:8b single-probe oracle AUC"] = models["llama3.1:8b"]["auc_k1"]
    if "qwen2.5:7b-instruct" in models:
        m = models["qwen2.5:7b-instruct"]
        out["E31 qwen2.5 AUC gain, sequential k=3 over one probe"] = m["auc_gain"]
    return out


def e32() -> dict[str, float]:
    d = _load("e32_pg_granularity.json")
    rows = {r["checkpoint_every"]: r for r in (d or {}).get("rows", [])}
    out: dict[str, float] = {}
    if 1 in rows:
        out["E32 pgvector adv_hist lower, g=1"] = rows[1]["adv_hist_lo"]
    if 4 in rows:
        out["E32 pgvector adv_hist lower, g=4"] = rows[4]["adv_hist_lo"]
    if 1 in rows and 16 in rows:
        out["E32 pgvector adv_hist, g=16 minus g=1"] = rows[16]["adv_hist"] - rows[1]["adv_hist"]
    return out


def e33() -> dict[str, float]:
    runs = (_load("e33_syncthing.json") or {}).get("runs", {})
    out: dict[str, float] = {}
    for pace, tag in (("agent-paced", "agent-paced"), ("fast", "fast")):
        d = runs.get(pace)
        if not d or d.get("mean_flushes_per_version") is None:
            continue
        out[f"E33 {tag}: flushes per retained Syncthing version"] = d["mean_flushes_per_version"]
        out[f"E33 {tag}: version-diff precision over the 1/B null"] = (
            d["precision"] / d["null_precision"]
        )
        out[f"E33 {tag}: exposure of a record's region"] = d["exposure"]
    return out


EXTRACTORS = {"e16x": e16x, "e29": e29, "e30": e30, "e31": e31, "e32": e32, "e33": e33}


CHECKS = [
    # --- E16 extension -------------------------------------------------------------------------
    _check(
        "E16x pgvector adv_hist lower, 20 seeds",
        "sec:eval-real -- the attack reaches cohort H in pgvector; three seeds were too few",
        0.0,
        1.0,
        "The three-seed interval was [0.042, 0.260] around +0.151. Seventeen further seeds of the "
        "identical workload should narrow it; predict its lower end still clears zero.",
        "e16x",
    ),
    _check(
        "E16x pgvector adv_hist, 20 seeds",
        "sec:eval-real -- the size of the pgvector effect",
        0.05,
        0.30,
        "Predict the 20-seed mean lands near the three-seed +0.151: above +0.05 (a real effect, "
        "weaker than the growing layout's) and below +0.30.",
        "e16x",
    ),
    # --- E32 -----------------------------------------------------------------------------------
    _check(
        "E32 pgvector adv_hist lower, g=1",
        "sec:eval-real -- with twice E16's induction budget the per-write channel is detectable",
        0.0,
        1.0,
        "E32 doubles the induced writes per topic to 40 so that g = 16 still yields 20 versions; "
        "at g = 1 that can only strengthen E16's detectable channel.",
        "e32",
    ),
    _check(
        "E32 pgvector adv_hist lower, g=4",
        "sec:granularity -- a client that coalesces four writes per version still leaks",
        0.0,
        1.0,
        "Frequency analysis tolerates mixed-topic flushes (sec:transfer: padding and batching "
        "raise the cost without removing the difference between topics); four writes per version "
        "is the "
        "padded design's own flush width, so predict the channel survives at g = 4.",
        "e32",
    ),
    _check(
        "E32 pgvector adv_hist, g=16 minus g=1",
        "sec:granularity -- coarser capture weakens every write-channel attack against W",
        -1.0,
        0.0,
        "Each version at g = 16 mixes sixteen writes, so the per-block rate difference between "
        "topics shrinks; predict the advantage at g = 16 is no higher than at g = 1.",
        "e32",
    ),
    # --- E29 -----------------------------------------------------------------------------------
    _check(
        "E29 mem0 on Qdrant: oracle AUC",
        "sec:eval-dedup -- mem0 as shipped leaks membership through write occurrence",
        0.65,
        1.0,
        "E18 measured 0.80 with an immediate observation. A fixed window observed by block counts "
        "carries at least that information; predict at least 0.65.",
        "e29",
    ),
    _check(
        "E29 mem0 on Mnemos-M, store observed: oracle AUC",
        "thm:reactive -- a clock-only store closes the oracle under the unmodified library",
        0.40,
        0.60,
        "The sweep writes the same byte ranges every Delta whether or not mem0 inserted anything "
        "(E21: staged minus empty flush is exactly 0 bytes), so over a fixed window the changed "
        "blocks are independent of membership; predict chance.",
        "e29",
    ),
    _check(
        "E29 mem0 on Mnemos-M, store observed: AUC upper",
        "thm:reactive -- the closed oracle's interval contains chance",
        0.5,
        1.0,
        "If the store is clock-only the interval must contain 0.5; an upper end below 0.5 would "
        "mean a reversed signal, which is also a leak.",
        "e29",
    ),
    _check(
        "E29 mem0 on Mnemos-M, store and history observed: oracle AUC",
        "thm:reactive scope -- other state the framework persists reopens the channel",
        0.65,
        1.0,
        "mem0 appends a history row for every ADD (memory/main.py, batch_add_history) and none for "
        "a skipped fact, beside the message log it writes either way, so its SQLite file grows "
        "more for a non-member. Predict the oracle reopens when the history is observed.",
        "e29",
    ),
    _check(
        "E29 mem0 on Mnemos-P, store observed: oracle AUC",
        "prop:derived -- a dual-epoch-secure store that flushes on writes is not reactively secure",
        0.65,
        1.0,
        "Mnemos-P flushes when mem0 inserts and not otherwise, which is the occurrence leakage of "
        "prop:derived; predict the oracle is as open as on Qdrant.",
        "e29",
    ),
    # --- E30 -----------------------------------------------------------------------------------
    _check(
        "E30 LangMem oracle AUC",
        "sec:attack-derived -- the oracle is present in a second framework as shipped",
        0.60,
        1.0,
        "LangMem's store manager writes only memories that are new or changed "
        "(extraction.py: final_puts), so an exact repeat of a stored fact should usually produce "
        "no put; predict an AUC of at least 0.60.",
        "e30",
    ),
    _check(
        "E30 LangMem oracle AUC, bytes appended",
        "sec:attack-derived -- LangMem's writes differ in size for stored and novel facts",
        0.60,
        1.0,
        "Added 2026-10-13 after a two-probe smoke test of the harness in which a member probe also "
        "wrote (LangMem re-puts retrieved memories it patches), so 'any write' may not separate "
        "the classes. A patch rewrites an existing memory while a novel fact adds one, so the "
        "bytes appended to the store should still be larger for a non-member; gocryptfs preserves "
        "the file size to within a block. Predict an AUC of at least 0.60 on that statistic.",
        "e30",
    ),
    _check(
        "E30 LangMem non-member minus member write rate",
        "sec:attack-derived -- a stored fact is written less often than a novel one",
        0.05,
        1.0,
        "The write-rate gap is the oracle's raw signal; predict it is positive.",
        "e30",
    ),
    # --- E31 -----------------------------------------------------------------------------------
    _check(
        "E31 llama3.1:8b single-probe oracle AUC",
        "sec:eval-dedup -- the oracle does not depend on one model making the decision",
        0.60,
        1.0,
        "mem0 drops an extracted fact whose hash is already present whatever model extracted it, "
        "and a second model should restate an exact repeat the same way often enough; predict "
        "at least 0.60.",
        "e31",
    ),
    _check(
        "E31 qwen2.5 AUC gain, sequential k=3 over one probe",
        "sec:eval-dedup -- re-probing a fact that wrote nothing can only add evidence",
        0.0,
        1.0,
        "A probe that writes nothing leaves the store unchanged, so later paraphrase probes on a "
        "member add evidence while a non-member usually stops after its first, writing, probe; "
        "predict the sequential statistic's AUC is at least the single probe's.",
        "e31",
    ),
    # --- E33 -----------------------------------------------------------------------------------
    # Ranges set 2026-10-13 after a three-minute infrastructure smoke test of the harness (one
    # write every 5 s): 11.5 flushes per retained version, precision 1.36 x the null, exposure
    # 1.00. That test is why the experiment runs two paces, and these ranges are informed by it;
    # they are registered before the 30-minute runs that test them.
    _check(
        "E33 agent-paced: flushes per retained Syncthing version",
        "sec:granularity -- at an agent's pace a sync client keeps about one version per write",
        0.9,
        3.0,
        "With one write every 30 s, Syncthing's 10 s watcher syncs each change before the next "
        "one and staggered versioning keeps one version per 30 s; predict at most three flushes "
        "per retained version.",
        "e33",
    ),
    _check(
        "E33 agent-paced: version-diff precision over the 1/B null",
        "sec:eval-multisnapshot -- a real client's history names an earlier record's region",
        3.0,
        64.0,
        "A version spanning one or two flushes names 5 to 10 of 64 regions; predict precision at "
        "least three times the null.",
        "e33",
    ),
    _check(
        "E33 agent-paced: exposure of a record's region",
        "sec:eval-multisnapshot -- the region holding a record is inside the diff",
        0.9,
        1.0,
        "A record's region is rewritten in the flush that places it, so it is inside the diff of "
        "the retained versions around that flush; predict exposure near 1.",
        "e33",
    ),
    _check(
        "E33 fast: flushes per retained Syncthing version",
        "sec:granularity -- a continuously written file is captured coarsely",
        5.0,
        30.0,
        "Syncthing delays rescanning a file that keeps changing, so at one write every 5 s each "
        "retained version spans many flushes (11.5 in the smoke test).",
        "e33",
    ),
    _check(
        "E33 fast: version-diff precision over the 1/B null",
        "sec:granularity -- coarse capture drives the diff toward the null",
        1.0,
        2.0,
        "A version spanning about a dozen flushes of five whole regions names most of the file; "
        "predict precision between the null and twice it.",
        "e33",
    ),
]
