"""Check the round-2 results on disk against the claims the **current** paper makes.

    python -m eval.exp2.verify

This is the round-2 counterpart to ``eval.repro --verify``. It exists because that one's ranges
encode the *previous* paper's claims -- ``adv_hist = 0.587``, ``adv_hist/adv_post = 0.943``, "query
equality closed outright at s = 2" -- and this round retired or reversed all three. A check whose
claim the paper no longer makes is worse than no check: it fails for the wrong reason, or passes
and certifies nothing.

The rule from ``repro.py`` still governs, and is the whole point of the file: **every check names
the paper claim it defends, the range, and why the range is what it is.** A range nobody can
justify is worse than no check at all.

Several of these ranges are deliberately wide. The headline numbers here are AUCs and advantages
whose sampling variation is real, and a check tight enough to catch a typo is also tight enough to
fail on a different seed. Where a claim is *qualitative* -- "at chance", "closed" -- the range is
the interval that would make the sentence false, not the interval the point estimate happens to
land in.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from eval.lab import corpus as C

R = C.repo_root() / "python" / "eval" / "results"


@dataclass(frozen=True)
class Check:
    name: str
    claim: str
    lo: float
    hi: float
    why: str


@dataclass
class Result:
    check: Check
    value: float | None
    ok: bool
    note: str = ""


def load(name: str):
    f = R / name
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"{name}: {e}") from e


# ----------------------------------------------------------------------------- extractors


def e4() -> dict[str, float]:
    d = load("e4_drift.json")
    if not d:
        return {}
    curve = d["fine"]["curve"]
    return {
        "E4 adv_H, historically aware": curve[0]["adv_H"],
        "E4 adv_H, naive (current content)": curve[-1]["adv_H"],
        "E4 adv_H, i.i.d. control": d["shuffled_control"]["adv_H"],
    }


def e5() -> dict[str, float]:
    d = load("e5_game.json")
    if not d:
        return {}
    by = {r["arm"]: r for r in d.get("rows", [])}
    out: dict[str, float] = {}
    if "P shipped" in by:
        out["E5 shipped, learned AUC"] = by["P shipped"]["learned"]["auc"]
        out["E5 shipped, refusals"] = float(by["P shipped"]["refusals_observed"])
    for arm, key in (
        ("P no S1 (growing regions)", "E5 no-S1 AUC"),
        ("P no S2 (unsealed slots)", "E5 no-S2 AUC"),
        ("P identity permutation", "E5 identity-permutation AUC"),
        ("P inadmissible, silent refusal", "E5 inadmissible silent AUC"),
        ("P inadmissible, padded refusal", "E5 inadmissible padded AUC"),
        ("M shipped", "E5 Mnemos-M AUC"),
        ("P no S3 (per-slot writes)", "E5 no-S3, random slot AUC"),
        ("P first-free slot (S3 kept)", "E5 first-free slot, S3 kept AUC"),
        ("P no S3, first-free slot", "E5 no-S3, first-free slot AUC"),
    ):
        if arm in by:
            out[key] = by[arm]["learned"]["auc"]
    return out


def e6() -> dict[str, float]:
    d = load("e6_stores.json")
    if not d:
        return {}
    stores = {s["store"]: s for s in d.get("stores", [])}
    out: dict[str, float] = {}
    if stores:
        out["E6 worst adv_hist over stores"] = max(s["adv_hist"] for s in stores.values())
    if "chroma" in stores:
        out["E6 chroma adv_hist"] = stores["chroma"]["adv_hist"]
        out["E6 chroma coverage"] = stores["chroma"]["coverage_hist"]
    return out


def e7() -> dict[str, float]:
    d = load("e7_permutation.json")
    if not d:
        return {}
    by = {m["mode"]: m for m in d.get("modes", [])}
    out: dict[str, float] = {}
    for mode, key in (
        ("spread_s1", "E7 exact recovery, s=1"),
        ("grouped", "E7 exact recovery, grouped"),
    ):
        if mode in by:
            out[key] = by[mode]["exact_mean"]
    if "spread_s1" in by:
        out["E7 exact recovery, s=1, interval lower end"] = by["spread_s1"]["exact_lo"]
    refined = [m for m in by.values() if "faq_exact_mean" in m]
    if refined:
        out["E7 refinement minus FAQ alone, worst mode"] = min(
            m["exact_mean"] - m["faq_exact_mean"] for m in refined
        )
    return out


def e15() -> dict[str, float]:
    d = load("e15_query_equality.json")
    if not d:
        return {}
    by = {m["mode"]: m for m in d.get("modes", [])}
    out: dict[str, float] = {}
    if "spread_s2" in by:
        out["E15 exact-equality J, s=2 (upper end)"] = by["spread_s2"]["exact_j"]["hi"]
        out["E15 held-out threshold J, s=2 (lower end)"] = by["spread_s2"]["threshold_j"]["lo"]
    if "spread_s4" in by:
        out["E15 held-out threshold J, s=4 (lower end)"] = by["spread_s4"]["threshold_j"]["lo"]
        if "spread_s2" in by:
            out["E15 threshold J, s=2 minus s=4"] = (
                by["spread_s2"]["threshold_j"]["mean"] - by["spread_s4"]["threshold_j"]["mean"]
            )
    if "grouped" in by:
        out["E15 held-out threshold J, grouped (lower end)"] = by["grouped"]["threshold_j"]["lo"]
    return out


def e8() -> dict[str, float]:
    d = load("e8_multisnapshot.json")
    if not d:
        return {}
    out: dict[str, float] = {}
    fine = [r for r in d.get("granularity_sweep", []) if r["granularity_flushes"] == 1]
    if fine:
        out["E8 exposure at g=1"] = fine[0]["exposure"]
        out["E8 precision at g=1"] = fine[0]["precision"]
    for r in d.get("cross_session", []):
        tag = "on" if r["rewrite_at_lock"] else "off"
        key = "E8 cross-session precision, rewrite_at_lock " + tag
        out[key] = r["precision"]
    return out


def e9() -> dict[str, float]:
    d = load("e9_refusal.json")
    if not d:
        return {}
    by = {c["mode"]: c for c in d.get("channel", [])}
    out: dict[str, float] = {}
    for mode, key in (("silent", "E9 silent refusal AUC"), ("padded", "E9 padded refusal AUC")):
        if mode in by:
            out[key] = by[mode]["auc_full_vs_open"]
    for c in d.get("capacity_arm", []):
        out[f"E9 utilisation at first refusal, {c['engine']}"] = c["utilisation_at_first_refusal"]
    return out


def e11() -> dict[str, float]:
    d = load("e11_retrieval.json")
    if not d or not d.get("domains"):
        return {}
    out: dict[str, float] = {}
    worst_q = 0.0
    for dom in d["domains"]:
        q = dom["plaintext_exhaustive"]["ndcg"] - dom["m_int8_exhaustive"]["ndcg"]
        worst_q = max(worst_q, abs(q))
        if dom["domain"] == "fiqa":
            out["E11 fiqa plaintext exhaustive nDCG@10"] = dom["plaintext_exhaustive"]["ndcg"]
            out["E11 fiqa corpus size"] = float(dom["n_corpus"])
    out["E11 worst quantisation cost"] = worst_q
    out["E11 refusals during indexing"] = float(
        sum(dom["p_s2"]["refused_during_indexing"] for dom in d["domains"])
    )
    return out


def e3() -> dict[str, float]:
    d = load("e3_dedup.json")
    if not d:
        return {}
    rows = d.get("rows", [])
    out: dict[str, float] = {}
    p = [r["auc"] for r in rows if r["engine"] == "p"]
    m = [r["auc"] for r in rows if r["engine"] == "m"]
    if p:
        out["E3 dedup AUC, Mnemos-P"] = max(p)
    if m:
        # The claim is that M is at chance, so the worst case for it is the arm furthest from 0.5.
        out["E3 dedup AUC, Mnemos-M"] = max(m, key=lambda v: abs(v - 0.5))
    return out


def e11_ivf() -> dict[str, float]:
    d = load("e11_retrieval.json")
    if not d:
        return {}
    doms = [x for x in d.get("domains", []) if "m_ivf_k8" in x]
    out: dict[str, float] = {}
    if doms:
        out["E11 M-ivf k8 minus P s=2, worst domain"] = min(
            x["m_ivf_k8"]["ndcg"] - x["p_s2"]["ndcg"] for x in doms
        )
        out["E11 M-ivf k4 vs plaintext IVF@4, worst |gap|"] = max(
            abs(x["m_ivf_k4"]["ndcg"] - x["plaintext_ivf_matched"]["ndcg"]) for x in doms
        )
    return out


def e10() -> dict[str, float]:
    d = load("e10_performance.json")
    if not d:
        return {}
    rows = [r for r in d.get("rows", []) if "error" not in r]
    out: dict[str, float] = {}
    at = {(r["engine"], r["n"]): r for r in rows}

    def ratio(a: str, n: int) -> float:
        return at[("p", n)]["search_ms"]["p50"] / max(at[(a, n)]["search_ms"]["p50"], 1e-9)

    ns = sorted({n for (e, n) in at if e == "m"} & {n for (e, n) in at if e == "p"})
    if ns:
        out["E10 search speedup P/M-exact, smallest n"] = ratio("m", ns[0])
        out["E10 search speedup P/M-exact, largest n"] = ratio("m", ns[-1])
        out["E10 Mnemos-M resident MB at largest n"] = at[("m", ns[-1])]["resident_bytes"] / 1e6
    # Designs that do not count their own bytes (Mnemos-RS) or keep nothing on disk (plaintext)
    # must record bytes written as absent. Counting the rows that instead carry a number turns
    # "an absent measurement is never a zero" into something the verifier enforces.
    out["E10 rows reporting bytes they did not measure"] = float(
        sum(
            1
            for r in rows
            if r["engine"] in ("rs", "plaintext") and r.get("bytes_per_memory") is not None
        )
    )
    ni = sorted({n for (e, n) in at if e == "m-ivf"} & {n for (e, n) in at if e == "p"})
    if ni:
        out["E10 search speedup P/M-ivf, worst over n"] = min(ratio("m-ivf", n) for n in ni)
    return out


def e1() -> dict[str, float]:
    d = load("e1_longitudinal.json")
    if not d:
        return {}
    rows = d.get("rows", [])
    # Rows from the corrected harness carry a membership field; older ones predate the fix and
    # are not evidence for anything the paper now says.
    if not rows or any("membership" not in r for r in rows):
        return {}
    out: dict[str, float] = {}
    grow = [r for r in rows if r["design"] == "growing"]
    m = [r for r in rows if r["design"] == "m"]
    if grow:
        out["E1 growing layout, best adv_H lower bound"] = max(r["lo"] for r in grow)
    rs = [r for r in rows if r["design"] == "rs"]
    if rs:
        out["E1 Mnemos-RS, best adv_H lower bound"] = max(r["lo"] for r in rs)
    if m:
        out["E1 Mnemos-M, worst |adv_H| over budgets and classes"] = max(
            max(abs(r["lo"]), abs(r["hi"])) for r in m
        )
    return out


def census() -> dict[str, float]:
    d = load("census.json")
    if not d:
        return {}
    rows = d.get("rows", [])
    return {
        "B4 frameworks whose writes depend on stored content": float(
            sum(1 for r in rows if r["write_depends_on_existing"] == "yes")
        ),
        "B4 frameworks read from source": float(len(rows)),
    }


def e2() -> dict[str, float]:
    d = load("e2_agent_attack.json")
    if not d:
        return {}
    rows = d.get("rows", [])
    out: dict[str, float] = {}
    grow = [r for r in rows if r.get("layout") == "growing"]
    if grow:
        # The positive control's strongest configuration, judged by the lower end of its interval.
        out["E2 growing layout, best adv_H lower bound"] = max(r["lo"] for r in grow)
    p = [r for r in rows if r.get("layout", "p") == "p"]
    if p:
        out["E2 Mnemos-P single-record, worst adv_H upper bound"] = max(
            r["hi"] for r in p if r["w_real"] == 1
        )
        by_w = {w: [r["adv_H"] for r in p if r["w_real"] == w] for w in (1, 4)}
        if by_w[1] and by_w[4]:
            base = max(sum(by_w[1]) / len(by_w[1]), 1e-3)
            out["E2 Mnemos-P, burst-of-4 over single-record advantage"] = (
                sum(by_w[4]) / len(by_w[4]) / base
            )
    return out


def e12() -> dict[str, float]:
    d = load("e12_oblivious.json")
    if not d:
        return {}
    rows = [r for r in d.get("rows", []) if "error" not in r]
    out: dict[str, float] = {}
    res = [r for r in rows if r["method"] == "resident"]
    if res:
        out["E12 resident bytes read per search"] = max(r["bytes_read_per_search"] for r in res)
    return out


CHECKS: list[tuple[Check, str]] = [
    # --- E4: drift is not a protection -------------------------------------------------------
    (
        Check(
            "E4 adv_H, historically aware",
            "sec:eval-drift -- an adversary inducing with archival "
            "content recovers the advantage in full",
            0.25,
            0.60,
            "The claim is that the aware adversary lands near or above the i.i.d. control rather "
            "than near the naive one. Below 0.25 it no longer 'recovers the advantage' and the "
            "paper's sentence is wrong; above 0.60 something other than drift is being "
            "measured.",
        ),
        "e4",
    ),
    (
        Check(
            "E4 adv_H, naive (current content)",
            "sec:eval-drift -- inducing with current content leaves the earlier cohort near chance",
            -0.20,
            0.10,
            "Scored for an adversary that guesses where it has no label (estimators.score), so it "
            "cannot be pushed below chance merely by abstaining. It must stay near chance for the "
            "contrast with the aware adversary to hold; if it rises, drift protects nothing even "
            "against a naive adversary and the finding changes.",
        ),
        "e4",
    ),
    # --- E5: the game --------------------------------------------------------------------------
    (
        Check(
            "E5 shipped, refusals",
            "sec:eval-game -- 'every admissible arm asserts the engine's refusal counter is zero'",
            0.0,
            0.0,
            "Exactly zero, not approximately. A refusal in an admissible arm means the history was "
            "inadmissible and the arm measured the refusal channel instead of the theorem's "
            "setting -- which is precisely the defect this experiment was rebuilt to remove. It "
            "appeared twice, and only at full power.",
        ),
        "e5",
    ),
    (
        Check(
            "E5 shipped, learned AUC",
            "sec:eval-game -- the shipped engine is at chance",
            0.40,
            0.60,
            "'At chance' is the claim, so the range is the band in which that sentence stays true "
            "for a classifier on a held-out half. Outside it the snapshot or trace is carrying "
            "something S1-S4 says it should not.",
        ),
        "e5",
    ),
    (
        Check(
            "E5 no-S1 AUC",
            "prop:seal -- removing fixed-size regions is distinguishable",
            0.90,
            1.00,
            "This is the positive control. If the ablation is NOT distinguishable the harness is "
            "broken, and the 'shipped at chance' result above means nothing -- the same trap that "
            "made a parser bug read as a successful defence in the previous round.",
        ),
        "e5",
    ),
    (
        Check(
            "E5 no-S2 AUC",
            "prop:seal -- removing sealed slots is distinguishable",
            0.90,
            1.00,
            "The second positive control, for the same reason.",
        ),
        "e5",
    ),
    (
        Check(
            "E5 identity-permutation AUC",
            "sec:eval-game -- removing the keyed permutation, which "
            "thm:main does not require, leaves the game at chance",
            0.40,
            0.60,
            "The correction this round makes to C4(a). The previous round measured 0.66 here and "
            "concluded the permutation was necessary; that was the refusal channel showing through "
            "inadmissible histories. With admissibility enforced this must return to "
            "chance.",
        ),
        "e5",
    ),
    (
        Check(
            "E5 inadmissible silent AUC",
            "prop:capacity -- a silent refusal is a distinguisher",
            0.80,
            1.00,
            "The refusal channel, inside the game. It must be strongly distinguishable or the "
            "proposition has nothing to bite on.",
        ),
        "e5",
    ),
    (
        Check(
            "E5 inadmissible padded AUC",
            "prop:capacity -- owing the region its rewrite closes it",
            0.40,
            0.60,
            "Paired with the row above: the same inadmissible histories, with the fix on, must "
            "return to chance. This pair is the evidence for S3's refusal clause.",
        ),
        "e5",
    ),
    (
        Check(
            "E5 Mnemos-M AUC",
            "sec:eval-game -- the memory-resident design is at chance in the game",
            0.40,
            0.60,
            "The headline design. Its row was being extracted and never checked, so the one number "
            "the design-space argument most depends on had no guard at all.",
        ),
        "e5",
    ),
    # --- E5: when is S3 necessary? -------------------------------------------------------------
    # Registered BEFORE the two first-free arms were run. The lemma (sec:conditions) says whole-
    # region rewrites and uniformly random intra-region placement are each sufficient to keep a
    # write from revealing a region's occupancy, and only removing both leaks. These three checks
    # are that prediction; a pass is evidence rather than a fit because the ranges were fixed
    # without seeing the numbers.
    (
        Check(
            "E5 no-S3, random slot AUC",
            "lem:slot -- per-slot writes are safe under random intra-region placement",
            0.40,
            0.60,
            "Measured before the lemma was stated, and the reason for it: the arm was expected to "
            "leak and did not. The range is chance.",
        ),
        "e5",
    ),
    (
        Check(
            "E5 first-free slot, S3 kept AUC",
            "lem:slot -- whole-region rewrites hide deterministic placement",
            0.40,
            0.60,
            "Deterministic placement with S3 intact. A whole-region rewrite reveals which region "
            "changed and never which slot, so placement order is invisible and this must be at "
            "chance.",
        ),
        "e5",
    ),
    (
        Check(
            "E5 no-S3, first-free slot AUC",
            "lem:slot -- removing both makes the slot index the occupancy",
            0.90,
            1.00,
            "The negative control. With the lowest free slot taken and only that slot written, the "
            "index written IS the region's fill level, and the concentrated history fills two "
            "regions to ~30 against ~1. If this is not near-perfectly distinguishable the "
            "experiment cannot see occupancy at all, and the two chance rows above mean nothing.",
        ),
        "e5",
    ),
    # --- E6: deployed stores -------------------------------------------------------------------
    (
        Check(
            "E6 worst adv_hist over stores",
            "sec:eval-real -- the attack does not reach the "
            "pre-compromise cohort in any deployed store measured",
            -0.20,
            0.05,
            "Chance is 0.125 and adv_hist is accuracy minus chance, so 0.0 is chance and negative "
            "means below it. The claim is that no store exceeds chance; above 0.05 the paper's "
            "negative result is wrong and the sentence must change.",
        ),
        "e6",
    ),
    (
        Check(
            "E6 chroma coverage",
            "sec:eval-real -- Chroma is the case where the adversary CAN label cohort-H blocks",
            0.80,
            1.00,
            "Coverage separates 'the labels carry no signal' from 'there was nothing to label'. "
            "Chroma must have high coverage or its adv_hist near zero means only that the attack "
            "could not attempt it, which is a different claim.",
        ),
        "e6",
    ),
    # --- E7: permutation recovery ---------------------------------------------------------------
    (
        Check(
            "E7 exact recovery, s=1",
            "sec:eval-permutation -- without probe randomization "
            "most of the permutation is recovered exactly",
            0.75,
            1.00,
            "A CLAIM REVISED BY ITS OWN RE-RUN IS RECORDED HERE. This check used to read "
            "'recovered "
            "exactly', range [0.90, 1.00], on three repeats that each returned 64/64. At twenty "
            "repeats FAQ alone recovers 0.885 [0.855, 0.913], exact on 1 of 20 stores, and the "
            "paper quotes whatever the stronger matcher reaches, with the number, not 'exactly'. "
            "The null is 1/B = 0.016.",
        ),
        "e7",
    ),
    (
        Check(
            "E7 exact recovery, s=1, interval lower end",
            "sec:eval-permutation -- the recovery is far above chance, not merely on average",
            0.50,
            1.00,
            "The interval, not the point, must clear half the regions; 0.50 is thirty times the "
            "1/B null and the level at which 'most' stops being true.",
        ),
        "e7",
    ),
    (
        Check(
            "E7 refinement minus FAQ alone, worst mode",
            "sec:eval-permutation -- the reported adversary is the stronger matcher",
            -0.01,
            1.0,
            "Registered before the refined matcher ran. Pairwise-swap search starts from each FAQ "
            "solution and accepts only swaps that raise the matching objective, so it should never "
            "recover fewer regions than FAQ alone; -0.01 allows an objective-improving swap that "
            "happens to move one region away from the truth. The grouped check above must hold "
            "against this matcher, not only against the weaker one.",
        ),
        "e7",
    ),
    (
        Check(
            "E7 exact recovery, grouped",
            "sec:eval-permutation -- grouped probing defeats it",
            0.00,
            0.10,
            "The mechanism the paper now prefers for this channel. It must be at or near the null, "
            "or the reversal of preference between the two probe mechanisms is unfounded.",
        ),
        "e7",
    ),
    # --- E8: multi-version --------------------------------------------------------------------
    (
        Check(
            "E8 exposure at g=1",
            "sec:eval-multisnapshot -- the region holding a pre-tau record is "
            "inside the version diff",
            0.95,
            1.00,
            "This is what falsifies 'history independence makes several snapshots no worse than "
            "one'. Exposure is whether the true region is in the candidate set at all; below 0.95 "
            "the refutation is weaker than stated.",
        ),
        "e8",
    ),
    (
        Check(
            "E8 cross-session precision, rewrite_at_lock on",
            "sec:mnemos-write -- re-encrypting at close drives a cross-session diff to the null",
            0.0,
            0.020,
            "The null is 1/B = 0.0156. The mitigation's claim is that the diff becomes total and "
            "therefore uninformative, so this must sit ON the null, not merely near it.",
        ),
        "e8",
    ),
    # --- E9: refusals ---------------------------------------------------------------------------
    (
        Check(
            "E9 silent refusal AUC",
            "prop:capacity -- a refusal that writes nothing discloses "
            "that the region has reached capacity",
            0.90,
            1.00,
            "Near-certain disclosure is the claim; the paper calls it 'not a small leak'.",
        ),
        "e9",
    ),
    (
        Check(
            "E9 padded refusal AUC",
            "sec:mnemos-p -- a refused write performs the rewrites an accepted one would",
            0.40,
            0.60,
            "The fix. Must be at chance, and must be measured on the same histories as the row "
            "above or the pair proves nothing.",
        ),
        "e9",
    ),
    (
        Check(
            "E9 utilisation at first refusal, p",
            "sec:eval-refusal -- Mnemos-P leaves the admissible regime early under skew",
            0.05,
            0.45,
            "The paper says 'roughly a quarter of nominal capacity'. The band is wide because the "
            "figure depends on the skew exponent; what must hold is that it is far below 1.0, "
            "which is what makes admissibility a restriction real deployments violate.",
        ),
        "e9",
    ),
    (
        Check(
            "E9 utilisation at first refusal, m",
            "sec:eval-refusal -- Mnemos-M has no per-partition capacity and admits every slot",
            0.99,
            1.00,
            "Exactly full. Anything less means a per-partition bound crept back in.",
        ),
        "e9",
    ),
    # --- E11: retrieval -------------------------------------------------------------------------
    (
        Check(
            "E11 worst quantisation cost",
            "sec:eval-recall -- int8 quantisation costs essentially "
            "nothing; the loss is fixed probing",
            0.0,
            0.03,
            "Mnemos-M searches exactly over the quantised vectors, so the gap between it and "
            "exhaustive plaintext float IS the quantisation cost. The paper attributes the whole "
            "retrieval gap to probing, which requires this to be near zero.",
        ),
        "e11",
    ),
    (
        Check(
            "E11 refusals during indexing",
            "sec:eval-recall -- refusals are counted, not swallowed",
            0.0,
            0.0,
            "Zero refusals means no relevant document silently failed to be stored. A non-zero "
            "value does not invalidate the run, but it must be reported rather than discovered, "
            "which is why it is a check and not a footnote.",
        ),
        "e11",
    ),
    (
        Check(
            "E11 fiqa plaintext exhaustive nDCG@10",
            "sec:eval-recall -- the full corpus, not a 5,000-document subsample",
            0.30,
            0.50,
            "Published bge-class numbers on full FiQA sit near 0.40. The previous round measured "
            "0.579 on a ninth of the corpus. This check is the guard against the corpus silently "
            "shrinking again.",
        ),
        "e11",
    ),
    # --- B4: the census, which the abstract quotes as "four of nine" ----------------------------
    (
        Check(
            "B4 frameworks whose writes depend on stored content",
            "abstract, sec:attack-derived -- four of nine frameworks",
            4.0,
            4.0,
            "Exact, because the abstract states the count. A framework is counted only where the "
            "mechanism was located at a cited file and line; changing the census changes this "
            "check, and then the abstract.",
        ),
        "census",
    ),
    (
        Check(
            "B4 frameworks read from source",
            "abstract -- of nine frameworks",
            9.0,
            9.0,
            "The denominator the abstract quotes.",
        ),
        "census",
    ),
    # --- E1: the longitudinal stream ----------------------------------------------------------
    # Registered before the corrected E1 ran (labels from public text only, permutation at E7's
    # measured strength, membership recorded per row).
    (
        Check(
            "E1 growing layout, best adv_H lower bound",
            "sec:eval-write -- against a growing "
            "layout the write channel reaches the earlier cohort",
            0.0001,
            1.0,
            "The layout discloses each record's partition in the snapshot, so labels transfer. "
            "Some "
            "budget's interval must lie wholly above chance, or E1 has no positive result to set "
            "the sealed designs against.",
        ),
        "e1",
    ),
    (
        Check(
            "E1 Mnemos-M, worst |adv_H| over budgets and classes",
            "sec:eval-write -- against the "
            "memory-resident design the write channel carries nothing, for either class",
            0.0,
            0.02,
            "Placement is uniform, so a region's label carries no topic whatever the budget, and "
            "reads never reach the disk, so the RW class learns nothing more. Taken over the whole "
            "interval, not the point, so a wide interval cannot pass by straddling zero.",
        ),
        "e1",
    ),
    (
        Check(
            "E1 Mnemos-RS, best adv_H lower bound",
            "tab:crossscheme -- the record-store design's write channel is open",
            0.0001,
            1.0,
            "Registered while the RS arm was running and before any of its rows existed. RS writes "
            "each record's index entry to the bucket its content selects, padded with dummies, so "
            "like Mnemos-P its padding should dilute the channel without closing it: some budget's "
            "interval must lie wholly above chance, or the 'open' in tab:crossscheme is wrong.",
        ),
        "e1",
    ),
    # --- E2: the attack on records a real agent wrote -----------------------------------------
    (
        Check(
            "E2 growing layout, best adv_H lower bound",
            "sec:eval-induction -- agent-written "
            "memories carry the topic structure transfer relies on",
            0.0001,
            1.0,
            "The positive control, added because the first E2 ran against Mnemos-P alone and so "
            "could not tell 'agent memories carry no structure' from 'the harness finds none'. "
            "The range is the claim itself: some budget's interval lies entirely above chance. "
            "Registered before the control was run.",
        ),
        "e2",
    ),
    (
        Check(
            "E2 Mnemos-P single-record, worst adv_H upper bound",
            "sec:eval-induction -- padding keeps a single-record attack near chance",
            -1.0,
            0.05,
            "Membership is supplied by the harness, so this isolates what padding does to "
            "labelling. "
            "A FAILED PREDICTION IS RECORDED HERE: this check was registered, before the burst "
            "arms "
            "ran, over every burst size, and it failed -- at w_real=4 the upper end reached 0.139, "
            "because a same-topic burst multiplies the signal against a fixed amount of cover. The "
            "paper reports the failure (sec:eval-induction) and the check now covers only the "
            "claim "
            "that survives, single-record writes; the burst effect is bound by the next "
            "check.",
        ),
        "e2",
    ),
    # --- E11: what the in-memory index costs in recall ---------------------------------------
    (
        Check(
            "E11 M-ivf k8 minus P s=2, worst domain",
            "sec:eval-recall -- at matched work the "
            "in-memory index loses no recall against the partitioned design",
            -0.005,
            1.0,
            "Registered before the arm was run. Both score 8 partitions' records per query, but "
            "all "
            "8 of Mnemos-M's are the genuine nearest partitions while half of Mnemos-P's are "
            "random "
            "cover, so M-ivf should match or beat P s=2 on every domain. -0.005 allows rounding.",
        ),
        "e11_ivf",
    ),
    (
        Check(
            "E11 M-ivf k4 vs plaintext IVF@4, worst |gap|",
            "sec:eval-recall -- the in-memory "
            "index costs exactly what probing costs, and nothing for being encrypted",
            0.0,
            0.01,
            "Same partitioner, same probe width, and quantisation measured at +0.000: the only "
            "possible difference is int8 scoring, which E11 already shows is free.",
        ),
        "e11_ivf",
    ),
    (
        Check(
            "E2 Mnemos-P, burst-of-4 over single-record advantage",
            "sec:eval-induction -- at "
            "four records per turn the advantage is several times the single-record value",
            2.0,
            1000.0,
            "Binds the prose to the data. Ratio of the mean advantage over budgets at w_real=4 to "
            "that at w_real=1; 'several times' means at least two. The denominator is floored at "
            "0.001 so a single-record arm at exact chance cannot make the ratio "
            "meaningless.",
        ),
        "e2",
    ),
    # --- E3: the deduplication oracle ------------------------------------------------------------
    (
        Check(
            "E3 dedup AUC, Mnemos-P",
            "sec:eval-dedup -- against a store that flushes when records "
            "are staged, the occurrence of a flush is the answer",
            0.75,
            1.00,
            "The attack has to work before its closure means anything. If it fails here the "
            "Mnemos-M row below is 'two broken arms agree', not a separation -- the same trap that "
            "made an empty intersection read as a successful defence in an earlier round.",
        ),
        "e3",
    ),
    (
        Check(
            "E3 dedup AUC, Mnemos-M",
            "sec:eval-dedup -- under a clocked flush a flush happens "
            "either way and the channel is closed",
            0.40,
            0.60,
            "Paired with the row above, and the empirical content of prop:derived being a "
            "separation rather than an impossibility. Taken as the arm furthest from chance, so a "
            "single leaking configuration cannot hide behind an average.",
        ),
        "e3",
    ),
    # --- E10: the crossover ----------------------------------------------------------------------
    (
        Check(
            "E10 search speedup P/M-exact, smallest n",
            "sec:eval-perf -- below the crossover, "
            "exhaustive in-memory ranking is faster than the partitioned design",
            1.0,
            100.0,
            "Replaces a check that asserted Mnemos-M is faster at every size. The full five-size "
            "run falsified that (P/M 0.25 at n=1e5): exhaustive ranking is O(n) and P's is not. "
            "What survives is the claim at small n, which is the common case for agent memory.",
        ),
        "e10",
    ),
    (
        Check(
            "E10 search speedup P/M-exact, largest n",
            "sec:eval-perf -- there is a crossover: "
            "exhaustive in-memory ranking loses to the partitioned design at scale",
            0.0,
            1.0,
            "The crossover itself. Below 1.0 at the largest size is what makes the in-memory "
            "partition index worth building; if exhaustive ranking stayed ahead, B5 would be moot.",
        ),
        "e10",
    ),
    (
        Check(
            "E10 search speedup P/M-ivf, worst over n",
            "sec:eval-perf -- with its in-memory "
            "partition index, Mnemos-M is at least as fast as the partitioned design at every size",
            1.0,
            1000.0,
            "Registered BEFORE the m-ivf arm was run. Both designs score the records of 8 "
            "partitions "
            "per query, but Mnemos-P reads and decrypts its 8 regions from disk and Mnemos-M does "
            "not, so M-ivf should win at every n. Taken as the worst ratio over sizes so one size "
            "cannot hide behind the others.",
        ),
        "e10",
    ),
    (
        Check(
            "E10 rows reporting bytes they did not measure",
            "tab:supp-e10 -- kB written/record is shown only for designs that count it",
            0.0,
            0.0,
            "Mnemos-RS has no byte counter and the plaintext baseline persists nothing. An earlier "
            "harness read the missing counter as 0 and printed '0.0 kB/memory' for the design that "
            "writes on every put; a dash is the only honest entry.",
        ),
        "e10",
    ),
    (
        Check(
            "E10 Mnemos-M resident MB at largest n",
            "sec:eval-perf -- what bounds Mnemos-M is resident memory",
            1.0,
            100000.0,
            "Not a performance target but a guard: the resident index must actually be resident. A "
            "value at or near zero would mean the store never populated and every latency number "
            "beside it is measuring an empty index.",
        ),
        "e10",
    ),
    # --- E15: query equality, with an interval and a held-out threshold ------------------------
    # Registered before E15 was run. The numbers these replace (J = 0.00 / 0.98 / 0.86) came from
    # one store, 12 queries, and a threshold fitted on the pairs it was scored on.
    (
        Check(
            "E15 exact-equality J, s=2 (upper end)",
            "sec:eval-query -- the randomised superset "
            "defeats an adversary that tests exact equality of read sets",
            -1.0,
            0.05,
            "Two searches for the same query share their genuine probes but draw fresh random "
            "cover, so identical read sets should essentially never occur. Upper end of the "
            "interval, so a noisy near-zero cannot pass by averaging.",
        ),
        "e15",
    ),
    (
        Check(
            "E15 held-out threshold J, s=2 (lower end)",
            "sec:eval-query -- but not one that "
            "thresholds the overlap: query equality survives the shipped mitigation",
            0.80,
            1.00,
            "Same-query searches share all k = 4 genuine probes; unrelated queries share about one "
            "partition by chance plus random-cover collisions. A threshold fixed on other queries "
            "should still separate them nearly perfectly. Lower end of the interval.",
        ),
        "e15",
    ),
    (
        Check(
            "E15 held-out threshold J, s=4 (lower end)",
            "supp:query -- more cover weakens the threshold test but does not close the channel",
            0.20,
            1.00,
            "A FAILED PREDICTION IS RECORDED HERE. Registered before the run as [0.60, 1.00], for "
            "the claim then in the supplement that raising s does not help. It measured 0.589 "
            "[0.573, 0.604]: doubling the cover takes J from 0.98 to 0.59, so s is a real knob, as "
            "the expected overlap gap k(1 - (2s-1)k/B) predicts once its variance is counted. The "
            "claim that survives is that cover weakens the test without closing it; 0.20 is where "
            "'not closed' would stop being a fair description of a pairwise test.",
        ),
        "e15",
    ),
    (
        Check(
            "E15 threshold J, s=2 minus s=4",
            "supp:query -- doubling the cover buys a real reduction",
            0.10,
            1.00,
            "Added with the revision above, after the run, so it binds the revised prose to the "
            "data "
            "rather than predicting it: the supplement now says cover helps, and this requires the "
            "help to be more than noise.",
        ),
        "e15",
    ),
    (
        Check(
            "E15 held-out threshold J, grouped (lower end)",
            "sec:eval-query -- grouped probes leak equality at group granularity",
            0.60,
            1.00,
            "A repeated query always reads the same group, so TPR is 1; FPR is the chance two "
            "distinct queries fall in one group (about 1/8 at 8 groups), so J should sit near "
            "0.86. "
            "The claim is that the leak exists, bounded by the anonymity set, not that it is "
            "closed.",
        ),
        "e15",
    ),
    # --- E12: oblivious payload fetch -------------------------------------------------------------
    (
        Check(
            "E12 resident bytes read per search",
            "sec:eval-perf -- the memory-resident design "
            "reaches the same leakage as the oblivious options at zero read cost",
            0.0,
            0.0,
            "Exactly zero bytes. This is the claim that the search, result and query-equality "
            "channels are absent rather than mitigated, and a single byte read per search would "
            "make it 'small' instead of 'none'.",
        ),
        "e12",
    ),
]

EXTRACTORS = {
    "census": census,
    "e1": e1,
    "e2": e2,
    "e11_ivf": e11_ivf,
    "e3": e3,
    "e4": e4,
    "e5": e5,
    "e6": e6,
    "e7": e7,
    "e8": e8,
    "e9": e9,
    "e10": e10,
    "e11": e11,
    "e12": e12,
    "e15": e15,
}


def _with_round3() -> None:
    """Append the third round's registered checks. Imported late: they import ``Check``."""
    from eval.exp2 import verify_r3

    CHECKS.extend(verify_r3.CHECKS)
    EXTRACTORS.update(verify_r3.EXTRACTORS)

    # Round four (pre-submission), registered 2026-10-13 before any of its experiments ran.
    from eval.exp2 import verify_r4

    CHECKS.extend(verify_r4.CHECKS)
    EXTRACTORS.update(verify_r4.EXTRACTORS)


def main() -> None:
    _with_round3()
    measured: dict[str, float] = {}
    errors: list[str] = []
    for key, fn in EXTRACTORS.items():
        try:
            measured.update(fn())
        except Exception as e:
            errors.append(f"{key}: {e}")

    results: list[Result] = []
    for check, _src in CHECKS:
        v = measured.get(check.name)
        if v is None:
            results.append(Result(check, None, False, "not present -- experiment not run yet"))
        else:
            results.append(Result(check, v, check.lo <= v <= check.hi))

    width = max(len(r.check.name) for r in results)
    print(f"\n{'':2} {'claim':<{width}}  {'measured':>10}  {'expected':>18}")
    print("  " + "-" * (width + 34))
    pending, failed = [], []
    for r in results:
        if r.value is None:
            mark, val = "..", "--"
            pending.append(r)
        elif r.ok:
            mark, val = "ok", f"{r.value:.4f}"
        else:
            mark, val = "!!", f"{r.value:.4f}"
            failed.append(r)
        print(f"{mark:2} {r.check.name:<{width}}  {val:>10}  [{r.check.lo:.3f}, {r.check.hi:.3f}]")

    if failed:
        print("\nOUT OF RANGE -- these numbers do not support the claims the paper makes:\n")
        for r in failed:
            print(f"  {r.check.name}")
            print(f"    claim:    {r.check.claim}")
            print(f"    measured: {r.value}")
            print(f"    why the range is what it is:\n      {r.check.why}\n")
    if pending:
        print(f"\n{len(pending)} check(s) pending -- those experiments have not been run:")
        for r in pending:
            print(f"  .. {r.check.name}")
    if errors:
        print("\nunreadable results:")
        for e in errors:
            print(f"  - {e}")

    ran = [r for r in results if r.value is not None]
    print(f"\n{sum(1 for r in ran if r.ok)}/{len(ran)} checks pass; {len(pending)} pending.")
    if failed or errors:
        print("VERIFICATION FAILED.")
        sys.exit(1)
    if pending:
        print("No failures, but the result set is incomplete.")
        sys.exit(0)
    print("All headline numbers reproduce within their claimed ranges.")


if __name__ == "__main__":
    main()
