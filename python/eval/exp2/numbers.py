"""Every number the prose quotes, as a LaTeX macro generated from the result files.

``make_tables`` already guarantees that no number in a *table* is typed by hand. The prose was the
gap: a sentence saying "the recovery is exact" or "J = 0.98" is a number too, and it stayed true
only as long as someone remembered to re-read it after a re-run. Two did not survive (E7's
"exact" became 0.885 at twenty repeats; E10's "faster on search" held only below ~10^4 records).

This module writes ``paper/tables/numbers.tex``: one ``\\newcommand`` per quoted value, computed
from the same result files, and where a verifier check exists, from the verifier's own extractor
-- so a number in the prose is by construction the number the verifier checked. A value whose
experiment has not run is emitted as a visible ``[not run]`` and listed on stdout, never as a
stale literal.

    cd python && python -m eval.exp2.numbers
"""

from __future__ import annotations

import json
import sys

from eval.exp2 import verify as V
from eval.lab import corpus as C

R = C.repo_root() / "python" / "eval" / "results"
OUT = C.repo_root() / "paper" / "tables" / "numbers.tex"

MISSING = r"\textbf{[not run]}"


def load(name: str):
    f = R / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def f(v: float | None, d: int = 3) -> str | None:
    return None if v is None else f"{v:.{d}f}"


def signed(v: float | None, d: int = 3) -> str | None:
    return None if v is None else f"{v:+.{d}f}".replace("-", r"\ensuremath{-}")


def ci(m, lo, hi, d=3) -> str | None:
    if m is None or lo is None or hi is None:
        return None
    return f"{m:.{d}f}~[{lo:.{d}f}, {hi:.{d}f}]"


def ci_of(r: dict | None, keys: tuple[str, str, str], d: int = 3) -> str | None:
    """``ci`` over three fields of a result row, or None when the row is absent."""
    return None if not r else ci(r[keys[0]], r[keys[1]], r[keys[2]], d)


def big(n: int | None) -> str | None:
    return None if n is None else f"{n:,}".replace(",", "{,}")


def collect() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    verified: dict[str, float] = {}
    for fn in V.EXTRACTORS.values():
        verified.update(fn())

    # --- E7: permutation recovery -----------------------------------------------------------
    e7 = load("e7_permutation.json")
    by7 = {m["mode"]: m for m in (e7 or {}).get("modes", [])}
    for mode, tag in (("spread_s1", "Sone"), ("spread_s2", "Stwo"), ("grouped", "Grouped")):
        m = by7.get(mode)
        out[f"REsevenExact{tag}"] = f(m and m["exact_mean"])
        out[f"REsevenExactCI{tag}"] = ci_of(m, ("exact_mean", "exact_lo", "exact_hi"))
        out[f"REsevenNeigh{tag}"] = f(m and m["neighbourhood_mean"])
        out[f"REsevenRegions{tag}"] = (
            None if not m or not e7 else str(round(m["exact_mean"] * e7["buckets"]))
        )
    for mode, tag in (("spread_s1", "Sone"), ("spread_s2", "Stwo")):
        m = by7.get(mode)
        out[f"REsevenFaq{tag}"] = f(m.get("faq_exact_mean")) if m else None
    out["REsevenRepeats"] = str(len(by7["spread_s1"]["repeats"])) if "spread_s1" in by7 else None
    out["REsevenAllExact"] = (
        str(sum(1 for x in by7["spread_s1"]["repeats"] if x == 1.0)) if "spread_s1" in by7 else None
    )

    ledger = {r["channel"]: r for r in (load("ledger.json") or {}).get("rows", [])}
    perm = ledger.get("permutation recovery (spread_s1)")
    # The conservative bound (Fano at the interval's lower end), as the ledger table prints it.
    out["RLedgerPermBits"] = f(perm and perm["bits_conservative"], 1)
    out["RLedgerPermBitsMax"] = f(perm and perm["bits_max"], 0)

    # --- E15: query equality ---------------------------------------------------------------------
    e15 = load("e15_query_equality.json")
    by15 = {m["mode"]: m for m in (e15 or {}).get("modes", [])}
    for mode, tag in (
        ("spread_s1", "Sone"),
        ("spread_s2", "Stwo"),
        ("spread_s4", "Sfour"),
        ("grouped", "Grouped"),
    ):
        m = by15.get(mode)
        tj = m and m["threshold_j"]
        ej = m and m["exact_j"]
        out[f"REfifteenThresh{tag}"] = ci_of(tj, ("mean", "lo", "hi"), 2)
        out[f"REfifteenExact{tag}"] = f(ej and ej["mean"], 2)
        out[f"REfifteenARI{tag}"] = f(m and m["cluster_ari"]["mean"], 2)
    out["REfifteenQueries"] = str(e15["n_queries_per_repeat"]) if e15 else None
    out["REfifteenSearches"] = str(e15["searches_per_query"]) if e15 else None
    out["REfifteenRepeats"] = str(len(next(iter(by15.values()))["repeats"])) if by15 else None

    # --- E4: drift (the abstract's range) --------------------------------------------------------
    out["REfourNaive"] = signed(verified.get("E4 adv_H, naive (current content)"), 2)
    out["REfourAware"] = signed(verified.get("E4 adv_H, historically aware"), 2)
    out["REfourShuffled"] = signed(verified.get("E4 adv_H, i.i.d. control"), 2)

    # --- E6: deployed stores -------------------------------------------------------------------
    e6 = load("e6_stores.json")
    st6 = {s["store"]: s for s in (e6 or {}).get("stores", [])}
    ch = st6.get("chroma")
    out["REsixChroma"] = (
        None
        if not ch
        else f"{signed(ch['adv_hist'])}~[{signed(ch['adv_hist_lo'])}, {signed(ch['adv_hist_hi'])}]"
    )
    out["REsixWorstHi"] = signed(max(s["adv_hist_hi"] for s in st6.values())) if st6 else None
    out["REsixSeeds"] = str(next(iter(st6.values()))["seeds"]) if st6 else None

    # --- search and deletion channels (earlier harness, eval/experiments/e5_search_transfer) ------
    sd = load("e5_e6_transfer.json")
    srch = {r["engine"]: r for r in (sd or {}).get("e5", [])}
    dele = {r["engine"]: r for r in (sd or {}).get("e6", [])}
    out["RSearchReadsRS"] = big(srch["v1"]["result_dependent_reads"]) if "v1" in srch else None
    out["RSearchPositionsRS"] = (
        big(srch["v1"]["distinct_positions_labelled"]) if "v1" in srch else None
    )
    out["RSearchReadsP"] = big(srch["v2"]["result_dependent_reads"]) if "v2" in srch else None
    out["RDeleteN"] = big(dele["v1"]["deletions"]) if "v1" in dele else None
    out["RDeleteNamedRS"] = (
        big(dele["v1"]["deletions_that_named_a_region_immediately"]) if "v1" in dele else None
    )
    out["RDeleteNamedP"] = (
        big(dele["v2"]["deletions_that_named_a_region_immediately"]) if "v2" in dele else None
    )

    # --- E9: refusals ----------------------------------------------------------------------------
    e9 = load("e9_refusal.json")
    ch9 = {c["mode"]: c for c in (e9 or {}).get("channel", [])}
    for mode, tag in (("silent", "Silent"), ("padded", "Padded")):
        c = ch9.get(mode)
        out[f"REnine{tag}AUC"] = ci_of(c, ("auc_full_vs_open", "lo", "hi"))
        out[f"REnine{tag}Full"] = f(c and c["mean_write_rate_full_regions"])
        out[f"REnine{tag}Open"] = f(c and c["mean_write_rate_open_regions"])
    cap9 = {c["engine"]: c for c in (e9 or {}).get("capacity_arm", [])}
    out["REnineUtilP"] = (
        f(cap9["p"]["utilisation_at_first_refusal"] * 100, 1) if "p" in cap9 else None
    )
    out["REnineUtilM"] = (
        f(cap9["m"]["utilisation_at_first_refusal"] * 100, 1) if "m" in cap9 else None
    )

    # --- E3: the deduplication oracle --------------------------------------------------------
    e3 = load("e3_dedup.json")
    rows3 = {(r["engine"], r["dedup"]): r for r in (e3 or {}).get("rows", [])}
    pl = rows3.get(("p", "llm"))
    ml = rows3.get(("m", "llm"))
    out["REthreePLlmAUC"] = ci_of(pl, ("auc", "lo", "hi"), 2)
    out["REthreeMLlmAUC"] = f(ml and ml["auc"], 2)
    out["REthreeLlmWriteMember"] = f(pl and pl["write_rate_member"], 2)
    out["REthreeLlmWriteNonmember"] = f(pl and pl["write_rate_nonmember"], 2)
    out["REthreeModel"] = (
        (pl["model"] or "").removeprefix("ollama:").replace("_", r"\_") if pl else None
    )

    # --- E5: the 2x2 that tests lem:slot ------------------------------------------------------
    e5 = load("e5_game.json")
    by5 = {r["arm"]: r for r in (e5 or {}).get("rows", [])}
    for arm, tag in (
        ("P no S3 (per-slot writes)", "SlotRandom"),
        ("P first-free slot (S3 kept)", "RegionFirstFree"),
        ("P no S3, first-free slot", "SlotFirstFree"),
        ("P shipped", "Shipped"),
    ):
        r = by5.get(arm)
        lr = r and r["learned"]
        out[f"REfive{tag}"] = ci_of(lr, ("auc", "lo", "hi"), 2)
    out["REfiveInstances"] = big(e5["instances_per_bit"]) if e5 else None

    # --- E10: the crossover and the in-memory index -----------------------------------------
    e10 = load("e10_performance.json")
    rows10 = [r for r in (e10 or {}).get("rows", []) if "error" not in r]
    at = {(r["engine"], r["n"]): r for r in rows10}
    ns = sorted({n for (e, n) in at if e == "m"} & {n for (e, n) in at if e == "p"})

    def speed(a: str, n: int) -> float:
        return at[("p", n)]["search_ms"]["p50"] / at[(a, n)]["search_ms"]["p50"]

    out["REtenSpeedupSmall"] = f(verified.get("E10 search speedup P/M-exact, smallest n"), 1)
    out["REtenSpeedupLarge"] = f(verified.get("E10 search speedup P/M-exact, largest n"), 2)
    out["REtenNSmall"] = big(ns[0]) if ns else None
    out["REtenNLarge"] = big(ns[-1]) if ns else None
    faster = [n for n in ns if speed("m", n) >= 1.0]
    slower = [n for n in ns if speed("m", n) < 1.0]
    out["REtenCrossLo"] = big(max(faster)) if faster and slower else None
    out["REtenCrossHi"] = big(min(slower)) if faster and slower else None
    ni = sorted({n for (e, n) in at if e == "m-ivf"} & set(ns))
    out["REtenIvfSpeedupMin"] = f(verified.get("E10 search speedup P/M-ivf, worst over n"), 1)
    out["REtenIvfSpeedupMax"] = f(max(speed("m-ivf", n) for n in ni), 1) if ni else None
    out["REtenIvfSearchLarge"] = (
        f(at[("m-ivf", ns[-1])]["search_ms"]["p50"], 2) if ni and ns else None
    )
    out["REtenMSearchLarge"] = f(at[("m", ns[-1])]["search_ms"]["p50"], 2) if ns else None
    out["REtenPSearchLarge"] = f(at[("p", ns[-1])]["search_ms"]["p50"], 2) if ns else None
    out["REtenUnlockMLarge"] = f(at[("m", ns[-1])]["unlock_s"], 1) if ns else None
    out["REtenUnlockIvfLarge"] = f(at[("m-ivf", ns[-1])]["unlock_s"], 1) if ni and ns else None
    out["REtenResidentLarge"] = f(verified.get("E10 Mnemos-M resident MB at largest n"), 0)
    out["REtenPutMLarge"] = f(at[("m", ns[-1])]["put_sparse_ms"]["p50"], 1) if ns else None
    out["REtenPutPLarge"] = f(at[("p", ns[-1])]["put_sparse_ms"]["p50"], 1) if ns else None

    # --- E11: recall of the in-memory index --------------------------------------------------
    e11 = load("e11_retrieval.json")
    doms = [x for x in (e11 or {}).get("domains", []) if "m_ivf_k8" in x]
    gains = [x["m_ivf_k8"]["ndcg"] - x["p_s2"]["ndcg"] for x in doms]
    out["REelevenWorstQuant"] = f(verified.get("E11 worst quantisation cost"))
    out["REelevenRefusals"] = (
        str(int(verified["E11 refusals during indexing"]))
        if "E11 refusals during indexing" in verified
        else None
    )
    out["REelevenFiqa"] = f(verified.get("E11 fiqa plaintext exhaustive nDCG@10"))
    out["REelevenFiqaSize"] = (
        big(int(verified["E11 fiqa corpus size"])) if "E11 fiqa corpus size" in verified else None
    )
    probe_cost = [
        x["plaintext_exhaustive"]["ndcg"] - x["plaintext_ivf_matched"]["ndcg"]
        for x in (e11 or {}).get("domains", [])
    ]
    out["REelevenProbeCostMax"] = f(max(probe_cost)) if probe_cost else None
    out["REelevenIvfGainMin"] = signed(min(gains)) if gains else None
    out["REelevenIvfGainMax"] = signed(max(gains)) if gains else None

    def dom(name: str) -> dict | None:
        return next((x for x in (e11 or {}).get("domains", []) if x["domain"] == name), None)

    def nd(x: dict | None, key: str) -> float | None:
        return x[key]["ndcg"] if x and key in x else None

    for name in ("scifact", "fiqa"):
        x = dom(name)
        tag = name.capitalize()
        out[f"REelevenGroupedExh{tag}"] = f(nd(x, "plaintext_exhaustive"))
        out[f"REelevenGroupedS{tag}"] = f(nd(x, "p_s2"))
        out[f"REelevenGrouped{tag}"] = f(nd(x, "p_grouped"))

    # --- E2: bursts ---------------------------------------------------------------------------
    e2 = load("e2_agent_attack.json")
    p2 = [r for r in (e2 or {}).get("rows", []) if r.get("layout", "p") == "p"]
    one = [r for r in p2 if r["w_real"] == 1]
    four = [r for r in p2 if r["w_real"] == 4]
    out["REtwoSingleMaxHi"] = f(max(r["hi"] for r in one)) if one else None
    top4 = max(four, key=lambda r: r["adv_H"]) if four else None
    out["REtwoBurstMax"] = ci_of(top4, ("adv_H", "lo", "hi"))
    out["REtwoBurstRatio"] = f(
        verified.get("E2 Mnemos-P, burst-of-4 over single-record advantage"), 1
    )
    grow = [r for r in (e2 or {}).get("rows", []) if r.get("layout") == "growing"]
    gtop = max(grow, key=lambda r: r["adv_H"]) if grow else None
    out["REtwoGrowingMax"] = ci_of(gtop, ("adv_H", "lo", "hi"))

    # --- E1: the longitudinal stream ---------------------------------------------------------
    e1 = load("e1_longitudinal.json")
    rows1 = (e1 or {}).get("rows", [])
    for design, tag in (("growing", "Growing"), ("rs", "RS"), ("p", "P"), ("m", "M")):
        rs = [r for r in rows1 if r["design"] == design]
        top = max(rs, key=lambda r: r["adv_H"]) if rs else None
        out[f"REone{tag}Max"] = ci_of(top, ("adv_H", "lo", "hi"))

    # --- E20: the reactive (feedback-agent) game ---------------------------------------------
    e20 = {r["arm"]: r for r in (load("e20_reactive.json") or {}).get("rows", [])}

    def r20(arm: str) -> dict | None:
        r = e20.get(arm)
        if not r:
            return None
        return max((r["learned"], r["hand_built"]), key=lambda a: abs(a["auc"] - 0.5))

    for arm, tag in (
        ("M sweep", "MSweep"),
        ("M regions", "MRegions"),
        ("M flush on write", "MOnDemand"),
        ("M content placement", "MContent"),
        ("P shipped", "PShipped"),
    ):
        out[f"REtwenty{tag}"] = ci_of(r20(arm), ("auc", "lo", "hi"), 3)

    # --- E21: what the clocked flush costs ----------------------------------------------------
    e21 = load("e21_flush_cost.json")
    g21 = (e21 or {}).get("geometries", {}).get("shipped")
    if g21:
        out["REtwentyoneSweepKiB"] = f(g21["sweep"]["bytes_per_flush"] / 1024, 1)
        out["REtwentyoneRegionMiB"] = f(g21["regions"]["bytes_per_flush"] / 2**20, 1)
        out["REtwentyoneSweepGBDay"] = f(g21["sweep"]["gb_per_day"]["10.0"], 2)
        out["REtwentyoneRegionGBDay"] = f(g21["regions"]["gb_per_day"]["0.5"], 0)
        out["REtwentyoneRatio"] = big(
            round(g21["regions"]["gb_per_day"]["0.5"] / g21["sweep"]["gb_per_day"]["10.0"])
        )
        out["REtwentyonePLockMiB"] = f(g21["p_lock"]["store_bytes"] / 2**20, 0)
        out["REtwentyoneSweepMs"] = f(g21["sweep"]["flush_ms_p50"], 1)
        out["REtwentyoneDriveYears"] = big(round(g21["sweep"]["drive_years_at_600tbw"]["10.0"]))
    else:
        for k in (
            "SweepKiB",
            "RegionMiB",
            "SweepGBDay",
            "RegionGBDay",
            "Ratio",
            "PLockMiB",
            "SweepMs",
            "DriveYears",
        ):
            out[f"REtwentyone{k}"] = None

    # --- E17: the composed attack on Mnemos-P ------------------------------------------------
    e17 = load("e17_composed.json")
    by17 = {r["granularity_flushes"]: r for r in (e17 or {}).get("composed", [])}
    orc = e17.get("oracle") if e17 else None
    out["REseventeenOracle"] = signed(orc["adv_H"]) if orc else None

    # --- E16: real content-placed layouts under gocryptfs ---------------------------------------
    e16 = load("e16_real_layouts.json")
    pg = next((s for s in (e16 or {}).get("stores", []) if s["store"] == "pgvector"), None)
    out["REsixteenPgAdv"] = ci_of(pg, ("adv_hist", "adv_hist_lo", "adv_hist_hi")) if pg else None
    out["REsixteenPgChance"] = f(pg["chance"], 3) if pg else None
    out["REsixteenPgSeeds"] = str(pg["seeds"]) if pg else None

    # --- E18: the dedup oracle against the real mem0 library ----------------------------------
    e18 = load("e18_real_mem0.json")
    if e18:
        out["REeighteenAUC"] = ci(e18["auc_write_vs_nonmember"], e18["lo"], e18["hi"], 2)
        out["REeighteenMemberWrite"] = f(e18["member_write_rate"], 2)
        out["REeighteenNonmemberWrite"] = f(e18["nonmember_write_rate"], 2)
        out["REeighteenModel"] = e18["model"].replace("_", r"\_")
    else:
        for k in ("AUC", "MemberWrite", "NonmemberWrite", "Model"):
            out[f"REeighteen{k}"] = None
    for g, tag in ((1, "One"), (2, "Two"), (4, "Four"), (16, "Sixteen")):
        r = by17.get(g)
        out[f"REseventeenComposed{tag}"] = signed(r and r["adv_H"])
        out[f"REseventeenCand{tag}"] = f(r and r["mean_candidates"], 1)

    # --- E22: the probe-design frontier (read amplification vs query equality) ----------------
    e22 = load("e22_probe_frontier.json") or {}
    words = {
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        6: "Six",
        8: "Eight",
        16: "Sixteen",
        32: "ThirtyTwo",
    }
    sp = {r["spread"]: r for r in e22.get("spread", [])}
    gr = {r["group_size"]: r for r in e22.get("grouped", [])}
    for s in (1, 2, 3, 4, 6, 8):
        out[f"REtwentytwoSpread{words[s]}"] = ci_of(
            sp.get(s), ("threshold_j", "threshold_j_lo", "threshold_j_hi")
        )
    for g in (2, 4, 8, 16, 32):
        r = gr.get(g)
        out[f"REtwentytwoGroup{words[g]}"] = ci_of(
            r, ("threshold_j", "threshold_j_lo", "threshold_j_hi")
        )
        out[f"REtwentytwoGroupAmp{words[g]}"] = f(r and r["read_amplification_x"], 1)

    # --- E23: refusal onset vs geometry and skew ----------------------------------------------
    e23 = load("e23_sensitivity.json")

    # Percentages of nominal capacity, as \REnineUtilP is.
    def pct(v: float | None) -> str | None:
        return f(None if v is None else 100 * v, 0)

    out["REtwentythreePWorst"] = pct(e23 and e23["p_util_worst"])
    out["REtwentythreePBest"] = pct(e23 and e23["p_util_best"])
    out["REtwentythreeMWorst"] = pct(e23 and e23["m_util_worst"])
    skew = {
        g["zipf"]: g
        for g in (e23 or {}).get("grid", [])
        if g["buckets"] == 256 and g["capacity"] == 32
    }
    for z, tag in ((0.8, "Low"), (1.2, "High")):
        out[f"REtwentythreePSkew{tag}"] = pct(skew[z]["p_util"] if z in skew else None)

    # --- E19: the multi-version channel through restic -----------------------------------------
    e19 = {
        (r["kind"], r["granularity_flushes"]): r
        for r in (load("e19_restic.json") or {}).get("rows", [])
    }
    for (kind, g), tag in ((("p", 1), "POne"), (("p", 4), "PFour")):
        r = e19.get((kind, g))
        out[f"REnineteenPrec{tag}"] = ci_of(r, ("precision", "lo", "hi"))
    for (kind, g), tag in ((("p", 1), "POne"), (("m", 1), "MOne")):
        r = e19.get((kind, g))
        out[f"REnineteenAdded{tag}"] = f(r and r["mean_data_added"] / 1024, 0)
        out[f"REnineteenRewritten{tag}"] = f(r and r["mean_rewritten_bytes"] / 1024, 0)
        out[f"REnineteenAmp{tag}"] = f(r and r["upload_amplification"], 0)
    m1 = e19.get(("m", 1))
    out["REnineteenMBlobs"] = f(m1 and m1["mean_new_blobs"], 1)

    # --- E25: Mnemos-M/disk at scale ---------------------------------------------------------
    e25 = {
        (r["engine"], r["n"]): r
        for r in (load("e25_scale.json") or {}).get("rows", [])
        if "error" not in r
    }
    for eng, tag in (("m-disk", "Disk"), ("m-sweep-ivf", "Mem"), ("p", "P")):
        r = e25.get((eng, 100_000))
        out[f"REtwentyfive{tag}Search"] = f(r and r["search_ms"]["p50"], 1)
        out[f"REtwentyfive{tag}Put"] = f(r and r["put_sparse_ms"]["p50"], 1)
        out[f"REtwentyfive{tag}Unlock"] = f(r and r["unlock_s"], 1)
        out[f"REtwentyfive{tag}RAM"] = f(r and r["resident_bytes"] / 1e6, 1)

    # --- E27: how much archival alignment the adversary needs ---------------------------------
    e27 = {r["alpha"]: r for r in (load("e27_archival_mix.json") or {}).get("rows", [])}
    for a, tag in ((0.0, "Zero"), (0.1, "Tenth"), (0.25, "Quarter"), (1.0, "Full")):
        out[f"REtwentyseven{tag}"] = signed(e27[a]["adv_H"]) if a in e27 else None

    # --- E24: the retrieval price of S4 (public vs corpus-trained table) -----------------------
    e24 = (load("e24_table_source.json") or {}).get("tables", {})
    for src, tag in (("public", "Pub"), ("corpus_trained", "Own")):
        t = e24.get(src)
        out[f"REtwentyfourP{tag}"] = f(t and t["p_s2"]["ndcg"], 3)
        out[f"REtwentyfourM{tag}"] = f(t and t["m_ivf_k8"]["ndcg"], 3)
        out[f"REtwentyfourIvf{tag}"] = f(t and t["plaintext_ivf"]["ndcg"], 3)
        out[f"REtwentyfourRefused{tag}"] = big(t and t["p_s2"]["refused_during_indexing"])
    pub = e24.get("public")
    out["REtwentyfourShare"] = f(pub and 100 * pub["max_partition_share"], 1)

    # --- E26: a second embedding model -------------------------------------------------------
    e26 = {r["domain"]: r for r in (load("e26_second_model.json") or {}).get("domains", [])}
    for dom, tag in (("scifact", "Sci"), ("nfcorpus", "Nf")):
        r = e26.get(dom)
        out[f"REtwentysix{tag}Quant"] = signed(
            r and r["plaintext_exhaustive"]["ndcg"] - r["m_int8_exhaustive"]["ndcg"]
        )
        out[f"REtwentysix{tag}MIvf"] = f(r and r["m_ivf_k8"]["ndcg"], 3)
        out[f"REtwentysix{tag}P"] = f(r and r["p_s2"]["ndcg"], 3)

    # --- E28: attack cost, descriptive, from E2's logs ----------------------------------------
    e28 = load("e28_attack_cost.json") or {}
    sec = [m["seconds_per_turn"] for m in e28.get("models", [])]
    out["REtwentyeightSecLo"] = f(min(sec), 1) if sec else None
    out["REtwentyeightSecHi"] = f(max(sec), 1) if sec else None
    mins = {(r["layout"], r["w_real"]): r for r in e28.get("minimum_detectable_budget", [])}
    for key, tag in ((("growing", 1), "GrowOne"), (("p", 1), "POne"), (("p", 2), "PTwo")):
        r = mins.get(key)
        out[f"REtwentyeight{tag}Msgs"] = big(r and r["adversary_messages"])
        out[f"REtwentyeight{tag}Min"] = f(r and r["agent_minutes_slowest_model"], 0)
    return out


def main() -> None:
    vals = collect()
    missing = sorted(k for k, v in vals.items() if v is None)
    lines = [
        "% Generated by `python -m eval.exp2.numbers` from python/eval/results/*.json.",
        "% Do not edit: re-run the generator. Every macro is a number the prose quotes.",
    ]
    for k in sorted(vals):
        v = vals[k] if vals[k] is not None else MISSING
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(vals)} macros to {OUT.relative_to(C.repo_root())}")
    if missing:
        print(f"{len(missing)} not yet measured (rendered as [not run]):")
        for k in missing:
            print(f"  \\{k}")
    if "--strict" in sys.argv and missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
