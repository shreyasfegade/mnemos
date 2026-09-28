"""Generate every results table in the paper from the result JSONs.

No number in the paper is typed by hand. Each table below reads the experiment's output and emits
a ``.tex`` fragment that ``paper/sections/07-evaluation.tex`` inputs. An experiment that has not
run yet emits a clearly marked placeholder, so the paper still builds and it is obvious which
numbers are missing rather than stale.

    cd python && python -m eval.exp2.make_tables
"""

from __future__ import annotations

import json

from eval.lab import corpus as C

R = C.repo_root() / "python" / "eval" / "results"
T = C.repo_root() / "paper" / "tables"


def load(name: str):
    f = R / name
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def pending(label: str, caption: str, extra_labels: tuple[str, ...] = ()) -> str:
    """A visible placeholder that still defines every label the prose references.

    ``extra_labels`` matters: a fragment that normally emits two floats must define both labels
    even when it falls back, or the build reports an undefined reference that looks like a prose
    bug rather than a missing experiment.

    The placeholder text must not date the manuscript against its own revisions. This string ships
    into the PDF for every experiment that has not yet run, so a phrase like "in the current round"
    puts revision history on the page in as many tables as there are pending experiments -- four,
    when ``lint_paper``'s REVISION rule caught it. Note where it was caught: in generated output,
    which no amount of reading the ``.tex`` sources would ever have surfaced, because the words
    live in this f-string rather than in the paper.
    """
    extra = "".join(f"\\label{{{x}}}\n" for x in extra_labels)
    return (
        f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{extra}"
        "\\small\n\\begin{tabular}{l}\n\\toprule\n"
        "\\emph{This experiment has not been run.} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def ci(m, lo, hi, d=3) -> str:
    return f"{m:.{d}f}\\,[{lo:.{d}f}, {hi:.{d}f}]"


# Tables that appear only in the supplement, where they carry interval columns too wide for one
# column of the two-column layout; they are set across the page there instead of overflowing.
SUPP_ONLY_WIDE = {
    "e1_longitudinal.tex",
    "e3_dedup.tex",
    "e7_permutation.tex",
    "e8_multisnapshot.tex",
    "e9_refusal.tex",
    "e12_oblivious.tex",
    # Narrow enough for one column, but alone in a column they strand a page of white space
    # among the supplement's full-width floats.
    "supp_e2_full.tex",
    "e4_drift.tex",
    "e9_capacity.tex",
}


def w(name: str, body: str) -> None:
    T.mkdir(parents=True, exist_ok=True)
    if name in SUPP_ONLY_WIDE and "\\begin{table*}" not in body:
        body = body.replace("\\begin{table}", "\\begin{table*}").replace(
            "\\end{table}", "\\end{table*}"
        )
    # The ledger's channel names are long by design (they say what the secret is); a size smaller
    # keeps its six columns inside the text width.
    if name in ("ledger.tex", "supp_ledger_full.tex", "supp_e10_full.tex"):
        body = body.replace("\\small\n\\begin{tabular}", "\\footnotesize\n\\begin{tabular}", 1)
    # Every generated table is set with slightly tighter column padding: the interval columns are
    # the widest cells, and default padding pushed several of them past the text width.
    for size in ("\\small", "\\footnotesize"):
        body = body.replace(
            f"{size}\n\\begin{{tabular}}",
            f"{size}\n\\setlength{{\\tabcolsep}}{{4pt}}\n\\begin{{tabular}}",
            1,
        )
    (T / name).write_text(body, encoding="utf-8")


# --------------------------------------------------------------------------- E5
def t_e5() -> None:
    d = load("e5_game.json")
    if not d:
        w("e5_game.tex", pending("tab:e5", "The dual-epoch game."))
        return
    rows = []
    for r in d["rows"]:
        lr = r["learned"]
        hb = r["hand_built"]
        verdict = "distinguishable" if r["distinguishable"] else "at chance"
        adm = "yes" if r["admissible"] else "\\textbf{no}"
        rows.append(
            f"{r['arm']} & {adm} & {r['refusals_observed']} & "
            f"{ci(lr['auc'], lr['lo'], lr['hi'], 2)} & {ci(hb['auc'], hb['lo'], hb['hi'], 2)} & "
            f"{verdict} \\\\"
        )
    body = "\n".join(rows)
    w(
        "e5_game.tex",
        f"""\\begin{{table*}}[t]
\\centering
\\caption{{The dual-epoch game against the compiled engines, {d["instances_per_bit"]} instances per
bit, evaluated on a held-out half. \\emph{{Learned}} is a gradient-boosted classifier over snapshot
and trace features; \\emph{{hand}} is one named statistic per arm, oriented on the training half.
A test bounds leakage from below, so the upper end of the interval is the ceiling for these
attackers and not a certificate.}}
\\label{{tab:e5}}
\\small
\\begin{{tabular}}{{lrrrrl}}
\\toprule
Configuration & Admissible & Refusals & Learned AUC [95\\% CI] & Hand AUC [95\\% CI] & Verdict \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""",
    )


# --------------------------------------------------------------------------- E6
def t_e6() -> None:
    d = load("e6_stores.json")
    # The aggregated form carries per-store intervals; an older single-run file does not, and a
    # table built from it would silently report a point estimate as if it had one.
    if not d or not d.get("stores") or "time_placed" not in d["stores"][0]:
        w("e6_stores.tex", pending("tab:e6", "Deployed vector stores under \\texttt{gocryptfs}."))
        return
    rows = []
    for s in d["stores"]:
        kind = "time" if s["time_placed"] else "content"
        rows.append(
            f"\\texttt{{{s['store'].replace('_', '-')}}} & {kind} & "
            f"{s['mean_blocks_per_write']:.1f} & "
            f"{ci(s['localisation_heldout'], s['localisation_lo'], s['localisation_hi'], 2)} & "
            f"{ci(s['adv_hist'], s['adv_hist_lo'], s['adv_hist_hi'], 2)} \\\\"
        )
    chance = d["stores"][0]["chance"]
    body = "\n".join(rows)
    w(
        "e6_stores.tex",
        f"""\\begin{{table*}}[t]
\\centering
\\caption{{Deployed vector stores under a \\texttt{{gocryptfs}} mount, {d["stores"][0]["seeds"]}
repeats each, 8 topics (chance {chance:.3f}). Induced writes are \\emph{{interleaved across
topics}}, so consecutive same-topic writes cannot make recency look like content placement, and
block labels are fitted on one half of them and scored on the other. \\advhist{{}} is the quantity
that matters: whether those labels identify the topics of records written \\emph{{before}} the
compromise.}}
\\label{{tab:e6}}
\\small
\\begin{{tabular}}{{llrrr}}
\\toprule
Store & Placed by & Blocks/write & Localisation, held out & \\advhist{{}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""",
    )


# --------------------------------------------------------------------------- E7
def t_e7() -> None:
    d = load("e7_permutation.json")
    if not d:
        w("e7_permutation.tex", pending("tab:e7", "Permutation recovery from probe co-occurrence."))
        return
    names = {
        "spread_s1": "Genuine probes only ($s=1$)",
        "spread_s2": "Randomised superset ($s=2$, shipped)",
        "grouped": "Grouped probes (group size 8)",
    }

    def faq(m: dict) -> str:
        if "faq_exact_mean" not in m:
            return "--"
        return ci(m["faq_exact_mean"], m["faq_exact_lo"], m["faq_exact_hi"], 3)

    rows = [
        f"{names.get(m['mode'], m['mode'])} & {faq(m)} & "
        f"{ci(m['exact_mean'], m['exact_lo'], m['exact_hi'], 3)} & "
        f"{m['neighbourhood_mean']:.3f} \\\\"
        for m in d["modes"]
    ]
    body = "\n".join(rows)
    w(
        "e7_permutation.tex",
        f"""\\begin{{table}}[t]
\\centering
\\caption{{Recovering the keyed partition permutation from search traffic alone, with no induced
writes, by matching the observed region co-occurrence graph against the one the public table
predicts ($B={d["buckets"]}$, $k_{{\\mathrm{{probe}}}}={d["kprobe"]}$, {d["n_queries"]} queries,
null $1/B = {1.0 / d["buckets"]:.3f}$), over {len(d["modes"][0]["repeats"])} independent stores.
\\emph{{FAQ}} is the fast approximate QAP solver alone; \\emph{{refined}} adds pairwise-swap local
search from each of its restarts and is the adversary reported, since a leakage measurement is only
as strong as its attacker. Applies to the RW class only.}}
\\label{{tab:e7}}
\\small
\\begin{{tabular}}{{lrrr}}
\\toprule
Probe mechanism & Exact, FAQ & Exact, refined & Neighbourhood \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""",
    )


# --------------------------------------------------------------------------- E8
def t_e8() -> None:
    d = load("e8_multisnapshot.json")
    if not d:
        w("e8_multisnapshot.tex", pending("tab:e8", "What a sequence of file versions reveals."))
        return
    rows = [
        f"every {r['granularity_flushes']} flushes & {r['exposure']:.3f} & "
        f"{ci(r['precision'], r['lo'], r['hi'], 3)} & {r['mean_candidate_regions']:.1f} \\\\"
        for r in d["granularity_sweep"]
    ]
    for r in d["cross_session"]:
        tag = "with" if r["rewrite_at_lock"] else "without"
        rows.append(
            f"across a session, {tag} re-encryption & {r['exposure']:.3f} & "
            f"{ci(r['precision'], r['lo'], r['hi'], 4)} & {r['mean_candidate_regions']:.1f} \\\\"
        )
    body = "\n".join(rows)
    null = d["granularity_sweep"][0]["null_precision"]
    w(
        "e8_multisnapshot.tex",
        f"""\\begin{{table}}[t]
\\centering
\\caption{{What a sequence of file versions reveals about writes made \\emph{{before}} the
compromise ($B={d["buckets"]}$, $w={d["epoch_writes"]}$, null precision $1/B={null:.4f}$).
\\emph{{Exposure}} is how often the record's true region is inside the diff; \\emph{{precision}} is
the chance of naming it by guessing within the diff.}}
\\label{{tab:e8}}
\\small
\\begin{{tabular}}{{lrrr}}
\\toprule
Capture & Exposure & Precision [95\\% CI] & Candidate regions \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""",
    )


# --------------------------------------------------------------------------- E9
def t_e9() -> None:
    d = load("e9_refusal.json")
    if not d or "lo" not in (d.get("channel") or [{}])[0]:
        w("e9_refusal.tex", pending("tab:e9", "The refusal channel."))
        w("e9_capacity.tex", pending("tab:e9cap", "Capacity under a skewed write trace."))
        return
    rows = []
    for r in d["channel"]:
        name = (
            "Silent (writes nothing)"
            if r["mode"] == "silent"
            else "Padded (rewrites the region anyway)"
        )
        rows.append(
            f"{name} & {ci(r['auc_full_vs_open'], r['lo'], r['hi'], 3)} & "
            f"{r['mean_write_rate_full_regions']:.3f} & {r['mean_write_rate_open_regions']:.3f} & "
            f"{r['gap']:+.3f} \\\\"
        )
    cap = "\n".join(
        f"\\texttt{{{c['engine']}}} & {c['admitted_total']} & {c['first_refusal_after']} & "
        f"{c['utilisation_at_first_refusal'] * 100:.1f}\\% \\\\"
        for c in d.get("capacity_arm", [])
    )
    body = "\n".join(rows)
    # Two floats in two files: the refusal channel is argued from in the main text, the capacity
    # arm is detail the supplement carries. (Two captions in one float would also share its
    # counter, so the second \label would resolve to the wrong number.)
    w(
        "e9_refusal.tex",
        f"""\\begin{{table}}[t]
\\centering
\\caption{{The refusal channel ($B={d["buckets"]}$, $C={d["capacity"]}$, $w={d["epoch_writes"]}$;
padding alone rewrites a given region on {d["padding_base_rate"]:.3f} of flushes). The adversary
induces writes destined for a region and watches whether that region is rewritten.}}
\\label{{tab:e9}}
\\small
\\begin{{tabular}}{{lrrrr}}
\\toprule
Refusal behaviour & AUC [95\\% CI] & Rate, full & Rate, open & Gap \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""",
    )
    w(
        "e9_capacity.tex",
        f"""\\begin{{table}}[t]
\\centering
\\caption{{Capacity under a skewed write trace: when each design first refuses. The partitioned
design leaves the admissible regime of \\cref{{thm:main}} long before the store is full.}}
\\label{{tab:e9cap}}
\\small
\\begin{{tabular}}{{lrrr}}
\\toprule
Design & Admitted & First refusal after & Utilisation \\\\
\\midrule
{cap}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""",
    )


# --------------------------------------------------------------------------- E10
TEMPLATE_WIDE = r"""\begin{table*}[t]
\centering
\caption{@CAPTION@}
\label{@LABEL@}
\small
\begin{tabular}{@COLS@}
\toprule
@HEAD@ \\
\midrule
@BODY@
\bottomrule
\end{tabular}
\end{table*}
"""

TEMPLATE_NARROW = TEMPLATE_WIDE.replace("table*", "table")


def fill(template: str, **parts: str) -> str:
    """Fill a LaTeX template's @NAME@ placeholders, refusing to leave any unfilled."""
    out = template
    for k, v in parts.items():
        out = out.replace(f"@{k.upper()}@", v)
    if "@" in out.replace(r"\@", ""):
        left = [x for x in ("@CAPTION@", "@LABEL@", "@COLS@", "@HEAD@", "@BODY@") if x in out]
        if left:
            raise ValueError(f"unfilled placeholders {left}")
    return out


E10_NAMES = {
    "plaintext": "Plaintext IVF (Rust)",
    "m": "Mnemos-M, exact",
    "m-ivf": "Mnemos-M, in-memory IVF",
    "p": "Mnemos-P",
    "rs": "Mnemos-RS",
}


def _e10_row(r: dict) -> str:
    name = E10_NAMES.get(r["engine"], r["engine"])
    if "error" in r:
        return name + " & " + f"{r['n']:,}" + r" & \multicolumn{6}{l}{did not run} \\"
    bpm = r.get("bytes_per_memory")
    # An engine that does not count its own bytes, and a store that keeps nothing on disk, both
    # print a dash: an absent measurement is never a zero.
    written = "--" if bpm is None else f"{bpm / 1e3:,.0f}"
    cells = [
        name,
        f"{r['n']:,}",
        f"{r['search_ms']['p50']:.3f}",
        f"{r['put_sparse_ms']['p50']:.2f}",
        written,
        f"{r['resident_bytes'] / 1e6:.1f}",
        f"{r['disk_bytes'] / 1e6:.1f}",
        f"{r['unlock_s']:.2f}",
    ]
    return " & ".join(cells) + r" \\"


E10_COMMON = (
    r"Against the in-tree Rust plaintext IVF, which shares the quantisation and the partitioner, "
    r"so a ratio is a statement about the security machinery. \emph{Put} is the sparse regime: one "
    r"record, one flush, which is what an agent writing seconds apart pays. \emph{kB "
    r"written/record} "
    r"is what a whole-region rewrite costs per record; a dash marks a design that does not count "
    r"its "
    r"bytes or keeps nothing on disk."
)
E10_HEAD = (
    r"Design & $n$ & Search $p50$ (ms) & Put $p50$ (ms) & kB written/record & RAM (MB) & "
    r"Disk (MB) & Unlock (s)"
)


def t_e10() -> None:
    d = load("e10_performance.json")
    if not d:
        w("supp_e10_full.tex", pending("tab:supp-e10", "Cost of the three designs."))
        return
    order = list(E10_NAMES)
    rows = sorted(
        d["rows"], key=lambda r: (r["n"], order.index(r["engine"]) if r["engine"] in order else 99)
    )
    # The main text argues from fig:crossover and quotes its numbers through numbers.tex; the
    # table lives in the supplement only, where it has room for every size.
    w(
        "supp_e10_full.tex",
        fill(
            TEMPLATE_WIDE,
            label="tab:supp-e10",
            cols="lrrrrrrr",
            head=E10_HEAD,
            body="\n".join(_e10_row(r) for r in rows),
            caption="Cost at every size measured. " + E10_COMMON,
        ),
    )


# --------------------------------------------------------------------------- E11
def t_e11() -> None:
    d = load("e11_retrieval.json")
    if not d:
        w("e11_retrieval.tex", pending("tab:e11", "Retrieval quality on the full BEIR corpora."))
        return
    rows = []
    for dom in d["domains"]:
        pe, pi = dom["plaintext_exhaustive"], dom["plaintext_ivf_matched"]
        me, p2 = dom["m_int8_exhaustive"], dom["p_s2"]
        g = dom["p_grouped"]
        mi = dom.get("m_ivf_k8")
        mi_s = f"{mi['ndcg']:.3f}" if mi else "--"
        rows.append(
            f"{dom['domain']} & {dom['n_corpus']:,} & {pe['ndcg']:.3f} & {pi['ndcg']:.3f} & "
            f"{me['ndcg']:.3f} & {mi_s} & {p2['ndcg']:.3f} & {g['ndcg']:.3f} & "
            f"{p2['refused_during_indexing']} \\\\"
        )
    body = "\n".join(rows)
    w(
        "e11_retrieval.tex",
        f"""\\begin{{table*}}[t]
\\centering
\\caption{{nDCG@10 against BEIR relevance judgments on the \\emph{{full}} corpora. The columns
separate the two sources of loss: the gap from exhaustive to IVF at the same
$k_{{\\mathrm{{probe}}}}={d["kprobe"]}$ is the cost of fixed probing, and the gap from exhaustive to
Mnemos-M, which searches exactly, is the cost of int8 quantisation. \\emph{{M (IVF)}} is Mnemos-M
with its in-memory partition index probing 8 partitions, the number of regions Mnemos-P reads at
$s{{=}}2$. Refusals during indexing are counted, not swallowed.}}
\\label{{tab:e11}}
\\small
\\begin{{tabular}}{{lrrrrrrrr}}
\\toprule
 & & \\multicolumn{{2}}{{c}}{{Plaintext float}} & \\multicolumn{{4}}{{c}}{{int8 engines}} & \\\\
\\cmidrule(lr){{3-4}}\\cmidrule(lr){{5-8}}
Domain & $|D|$ & exhaustive & IVF@$k$ & M (exact) & M (IVF) & P ($s{{=}}2$) & P grouped
 & Refused \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""",
    )


# --------------------------------------------------------------------------- simple ones
def t_e4() -> None:
    d = load("e4_drift.json")
    if not d:
        w("e4_drift.tex", pending("tab:e4", "Drift, and an adversary that knows the past."))
        return
    rows = []
    for c in d["fine"]["curve"]:
        rows.append(f"{c['gap_eras']} & {c['adversary']} & {ci(c['adv_H'], c['lo'], c['hi'])} \\\\")
    sc = d["shuffled_control"]
    body = "\n".join(rows)
    w(
        "e4_drift.tex",
        f"""\\begin{{table}}[t]
\\centering
\\caption{{Advantage on cohort H against the time gap between cohort H and the content the
adversary induces ({d["corpus"]}, {d["fine"]["n_topics"]} topics, chance
{d["fine"]["chance"]:.3f}). Gap 0 is an adversary inducing with archival content from cohort H's own
era; the largest gap is one inducing with current content. The i.i.d.\\ control is
{ci(sc["adv_H"], sc["lo"], sc["hi"])}.}}
\\label{{tab:e4}}
\\small
\\begin{{tabular}}{{llr}}
\\toprule
Gap (eras) & Adversary & \\advhist{{}} [95\\% CI] \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""",
    )


def t_e3() -> None:
    d = load("e3_dedup.json")
    if not d:
        w("e3_dedup.tex", pending("tab:e3", "Deduplication membership inference."))
        return

    def decider(r: dict) -> str:
        if r["dedup"] != "llm":
            return "similarity"
        return "model: " + tex_text((r.get("model") or "?").removeprefix("ollama:").split(":")[0])

    rows = [
        f"{decider(r)} & {'Mnemos-P' if r['engine'] == 'p' else 'Mnemos-M'} & "
        f"{ci(r['auc'], r['lo'], r['hi'], 2)} & {r['write_rate_member']:.2f} & "
        f"{r['write_rate_nonmember']:.2f} \\\\"
        for r in d["rows"]
    ]
    body = "\n".join(rows)
    w(
        "e3_dedup.tex",
        f"""\\begin{{table}}[t]
\\centering
\\caption{{Deduplication membership inference: whether an adversary learns that a fact was already
in cohort H, from the occurrence of a write alone ({d["n_probes_per_class"]} probes per class,
bootstrap intervals). The write rates are the agent's decisions and are the same for both designs;
Mnemos-M's clocked flush makes what the adversary observes identical either way.}}
\\label{{tab:e3}}
\\small
\\begin{{tabular}}{{llrrr}}
\\toprule
Decided by & Design & AUC [95\\% CI] & Writes, member & non-member \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""",
    )


def t_e1() -> None:
    d = load("e1_longitudinal.json")
    if not d:
        w("e1_longitudinal.tex", pending("tab:e1", "Longitudinal exposure of the earlier cohort."))
        return
    names = {"growing": "Growing layout", "p": "Mnemos-P", "m": "Mnemos-M", "rs": "Mnemos-RS"}
    how = {"disclosed by the snapshot": "snapshot", "supplied by the harness": "supplied"}
    body = []
    for r in d["rows"]:
        body.append(
            " & ".join(
                [
                    names.get(r["design"], r["design"]),
                    r["adversary"],
                    str(r["budget"]),
                    how.get(r.get("membership") or "", "--"),
                    ci(r["adv_H"], r["lo"], r["hi"]),
                ]
            )
            + r" \\"
        )
    rec = next(
        (r["permutation_recovery"] for r in d["rows"] if r.get("permutation_recovery")), None
    )
    rec_s = f"{rec:.3f}" if rec is not None else "--"
    caption = (
        f"Exposure of the pre-compromise cohort on a timestamped stream ({d['n_records']:,} "
        f"records "
        f"over {d['n_months']} months, $\\tau$ at month {d['tau_month']}, {d['n_topics']} topics, "
        f"chance {d['chance']:.3f}). Source: {tex_text(str(d['stream_source']))}. "
        r"\emph{Membership} is how the attack learns which region holds a given earlier record: "
        r"the growing layout's snapshot discloses it, while for the sealed designs the harness "
        r"supplies it, so those rows state what the labels would be worth to an adversary that "
        r"obtained membership elsewhere (\cref{sec:eval-multisnapshot}). The RW adversary against "
        r"Mnemos-P recovers the keyed permutation at the rate \cref{sec:eval-permutation} "
        r"measured, " + rec_s + r", and labels partitions from public text only."
    )
    w(
        "e1_longitudinal.tex",
        fill(
            TEMPLATE_NARROW,
            label="tab:e1",
            cols="llrlr",
            head=r"Design & Class & Budget & Membership & \advhist{} [95\% CI]",
            body="\n".join(body),
            caption=caption,
        ),
    )


def _e2_rows(rows: list[dict]) -> str:
    names = {"growing": "growing (control)", "p": "Mnemos-P"}
    out = []
    for r in rows:
        lay = r.get("layout", "p")
        out.append(
            " & ".join(
                [
                    names.get(lay, lay),
                    str(r["w_real"]),
                    str(r["budget_turns_per_topic"]),
                    ci(r["adv_H"], r["lo"], r["hi"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(out)


def _e2_caption(d: dict, lead: str) -> str:
    return (
        lead + f" The attack on {d['n_memories']:,} memories a local model actually wrote "
        f"({d['n_topics']} topics, chance {d['chance']:.3f}). "
        r"$w_{\mathrm{real}}$ is the burst size: how many same-topic memories one writing turn "
        r"contributes to a flush. The growing layout discloses each record's partition in the "
        r"snapshot and is the positive control; the advantage is scored for an adversary that "
        r"guesses where it has no label."
    )


E2_HEAD = r"Layout & $w_{\mathrm{real}}$ & Turns/topic & \advhist{} [95\% CI]"


def t_e2() -> None:
    d = load("e2_agent_attack.json")
    if not d:
        w("e2_agent.tex", pending("tab:e2", "The attack on records a real agent wrote."))
        return
    rows = d["rows"]
    top = max(r["budget_turns_per_topic"] for r in rows)
    w(
        "e2_agent.tex",
        fill(
            TEMPLATE_NARROW,
            label="tab:e2",
            cols="lrrr",
            head=E2_HEAD,
            body=_e2_rows([r for r in rows if r["budget_turns_per_topic"] == top]),
            caption=_e2_caption(
                d,
                f"At the largest budget measured ({top} turns per topic); the "
                "supplement gives every budget.",
            ),
        ),
    )
    w(
        "supp_e2_full.tex",
        fill(
            TEMPLATE_NARROW,
            label="tab:supp-e2",
            cols="lrrr",
            head=E2_HEAD,
            body=_e2_rows(rows),
            caption=_e2_caption(d, "Every budget measured."),
        ),
    )


def t_e15() -> None:
    d = load("e15_query_equality.json")
    if not d:
        w("supp_e15.tex", pending("tab:supp-e15", "Query equality under each probe mechanism."))
        return
    names = {
        "spread_s1": "Genuine probes only ($s=1$)",
        "spread_s2": "Randomised superset ($s=2$, shipped)",
        "spread_s4": "Randomised superset ($s=4$)",
        "grouped": "Grouped probes (group size 8)",
    }
    body = "\n".join(
        " & ".join(
            [
                names.get(m["mode"], tex_text(m["mode"])),
                f"{m['regions_per_search']:.0f}",
                ci(m["exact_j"]["mean"], m["exact_j"]["lo"], m["exact_j"]["hi"], 2),
                ci(m["threshold_j"]["mean"], m["threshold_j"]["lo"], m["threshold_j"]["hi"], 2),
                str(m["threshold_tau_mode"]),
                ci(m["cluster_ari"]["mean"], m["cluster_ari"]["lo"], m["cluster_ari"]["hi"], 2),
            ]
        )
        + r" \\"
        for m in d["modes"]
    )
    reps = len(d["modes"][0]["repeats"])
    w(
        "supp_e15.tex",
        fill(
            TEMPLATE_WIDE,
            label="tab:supp-e15",
            cols="lrrrrr",
            head=(
                r"Probe mechanism & Regions/search & Exact-equality $J$ & Threshold $J$ & "
                r"$\tau$ (mode) & Clustering ARI"
            ),
            body=body,
            caption=(
                rf"Query equality against three adversaries ($B={d['buckets']}$, "
                rf"$k_{{\mathrm{{probe}}}}={d['probe_width']}$), over {reps} independent "
                rf"stores with {d['n_queries_per_repeat']} queries each, every query searched "
                rf"{d['searches_per_query']} times in random order. The overlap threshold "
                r"$\tau$ and the clustering threshold are fixed on pairs among half of each "
                r"store's queries and scored on the other half, so no adversary is fitted to "
                r"the pairs it is scored on. "
                r"$J$ is Youden's (TPR $-$ FPR); ARI is the adjusted Rand index of the recovered "
                r"query groups. Applies to the RW class only."
            ),
        ),
    )


def t_e12() -> None:
    d = load("e12_oblivious.json")
    if not d:
        w("e12_oblivious.tex", pending("tab:e12", "Oblivious payload fetch."))
        return
    rows = []
    for r in d["rows"]:
        if "error" in r:
            continue
        lat = f"{r['latency_ms']['p50']:.3f}" if "latency_ms" in r else "---"
        src = "measured" if r.get("measured") else "computed"
        rows.append(
            f"{r['method'].replace('_', ' ')} & {r['n']} & {src} & {lat} & "
            f"{r['bytes_read_per_search'] / 1e3:.1f} & "
            f"{'yes' if r['leaks_result_positions'] else 'no'} \\\\"
        )
    body = "\n".join(rows)
    w(
        "e12_oblivious.tex",
        f"""\\begin{{table}}[t]
\\centering
\\caption{{Fetching the $top\\_k$ payloads, for the RW class. Path ORAM is a cost model with its
parameters stated ($Z={d["oram_bucket_size"]}$), not a measurement.}}
\\label{{tab:e12}}
\\small
\\begin{{tabular}}{{lrlrrl}}
\\toprule
Method & $n$ & Source & $p50$ (ms) & kB read/search & Leaks positions \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""",
    )


# --------------------------------------------------------------------------- ledger and census
LEDGER_HEAD = r"Channel & Design & Class & Status & Bits $\geq$ & Detectable $\geq$"

# The rows the main text argues from. Everything is in the supplement's full ledger; this subset
# is chosen so that every channel family appears once for each design it applies to.
LEDGER_MAIN = [
    "structure transfer, historically aware adversary",
    "snapshot + post-tau writes (the game)",
    "refusal, silent",
    "refusal, padded",
    "multi-version diff, g=1",
    "multi-version diff, across a session, rewrite_at_lock on",
    "permutation recovery (spread_s1)",
    "permutation recovery (spread_s2)",
    "permutation recovery (grouped)",
    "query equality (spread_s2), overlap threshold",
    "query equality (grouped), overlap threshold",
    "dedup membership (threshold)",
    "dedup membership (llm)",
]


TEX_ESCAPES = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("$", r"\$"),
    ("{", r"\{"),
    ("}", r"\}"),
)


def tex_text(value: str) -> str:
    """Escape a plain-text cell for LaTeX. Store names like append_control are data, not markup."""
    for a, b in TEX_ESCAPES:
        value = value.replace(a, b)
    return value


def _ledger_row(r: dict) -> str:
    # A bound that holds at the experiment's confidence where an interval exists; the point bound
    # otherwise, marked, because a number without an interval is not the same kind of claim.
    if r["bits_conservative"] is not None:
        bits = f"{r['bits_conservative']:.3f}"
    elif r["status"] == "no interval":
        bits = f"{r['bits']:.3f}$^\\ast$"
    else:
        bits = f"{r['bits']:.3f}"
    if r["mde"] is not None:
        mde = f"{r['mde']:.3f} ({r['metric']})"
    elif r["status"] == "closed":
        mde = "undetermined"  # a zero-width interval says nothing about power; never print 0
    else:
        mde = "--"
    status = {"open": r"\textbf{open}", "closed": "closed", "no interval": "no interval"}[
        r["status"]
    ]
    # Escape first, then add the one piece of markup a channel name is allowed.
    chan = tex_text(r["channel"]).replace("post-tau", r"post-$\tau$")
    return " & ".join([chan, tex_text(r["design"]), r["adversary"], status, bits, mde]) + r" \\"


LEDGER_CAPTION = (
    r"Every channel in one unit. \emph{Bits} is a Fano lower bound on the mutual information "
    r"between the secret and what the adversary observes, per observation of the unit the channel "
    r"is about (a record's topic, a region's partition, one game instance), taken at the lower end "
    r"of the experiment's 95\% interval so that it holds at the experiment's confidence; "
    r"$^\ast$ marks a result with no interval, for which only the point bound exists. A test "
    r"bounds leakage from below, so for a \emph{closed} channel the informative number is the last "
    r"column: the smallest effect, in the experiment's own unit, it had 80\% power to detect."
)


def t_ledger() -> None:
    d = load("ledger.json")
    if not d:
        w("ledger.tex", pending("tab:ledger", "Every channel in one unit."))
        return
    rows = d["rows"]
    main_rows = []
    for key in LEDGER_MAIN:
        main_rows.extend(r for r in rows if r["channel"] == key)
    w(
        "ledger.tex",
        fill(
            TEMPLATE_WIDE,
            label="tab:ledger",
            cols="llllrl",
            head=LEDGER_HEAD,
            body="\n".join(_ledger_row(r) for r in main_rows),
            caption=LEDGER_CAPTION + r" The supplement gives every row.",
        ),
    )
    w(
        "supp_ledger_full.tex",
        fill(
            TEMPLATE_WIDE,
            label="tab:supp-ledger",
            cols="llllrl",
            head=LEDGER_HEAD,
            body="\n".join(_ledger_row(r) for r in rows),
            caption=LEDGER_CAPTION + " Every measured channel, including every ablation.",
        ),
    )


def t_census() -> None:
    d = load("census.json")
    if not d:
        w("supp_census.tex", pending("tab:supp-census", "Agent-memory frameworks, from source."))
        return
    verdict = {"yes": r"\textbf{yes}", "none found": "none found", "not verified": "not verified"}
    body = []
    for r in d["rows"]:
        # \nolinkurl breaks a long path at its slashes; \texttt cannot break it at all.
        ev = r"\nolinkurl{" + r["evidence"][0] + "}" if r["evidence"] else "--"
        body.append(
            " & ".join(
                [
                    tex_text(r["framework"]),
                    tex_text(r["version"]),
                    tex_text(r["default_store"]),
                    verdict.get(r["write_depends_on_existing"], r["write_depends_on_existing"]),
                    ev,
                ]
            )
            + r" \\"
        )
    w(
        "supp_census.tex",
        fill(
            TEMPLATE_WIDE,
            label="tab:supp-census",
            cols="llp{3.6cm}lp{4.9cm}",
            head=(
                "Framework & Version & Default store & Write depends on existing content & Evidence"
            ),
            body="\n".join(body),
            caption=(
                r"Whether a write reaches storage depends on whether the content is already "
                r"stored, read from each framework's shipped source (\texttt{pip download "
                r"--no-deps}, " + d["date"] + r"). \emph{Yes} is recorded only where the "
                r"mechanism was located at the cited file; anything not located is \emph{none "
                r"found} or \emph{not verified}, never inferred from documentation. Where it is "
                r"yes, the precondition of \cref{prop:derived} holds and the deduplication oracle "
                r"of \cref{sec:attack-derived} applies to the framework as shipped."
            ),
        ),
    )


NAMES = {
    "t_e1": ("e1_longitudinal.tex", "tab:e1", "Longitudinal exposure of the earlier cohort."),
    "t_e2": ("e2_agent.tex", "tab:e2", "The attack on records a real agent wrote."),
    "t_e3": ("e3_dedup.tex", "tab:e3", "Deduplication membership inference."),
    "t_e4": ("e4_drift.tex", "tab:e4", "Drift, and an adversary that knows the past."),
    "t_e5": ("e5_game.tex", "tab:e5", "The dual-epoch game."),
    "t_e6": ("e6_stores.tex", "tab:e6", "Deployed vector stores under \\texttt{gocryptfs}."),
    "t_e7": ("e7_permutation.tex", "tab:e7", "Permutation recovery from probe co-occurrence."),
    "t_e8": ("e8_multisnapshot.tex", "tab:e8", "What a sequence of file versions reveals."),
    "t_e9": ("e9_refusal.tex", "tab:e9", "The refusal channel."),
    "t_e10": ("supp_e10_full.tex", "tab:supp-e10", "Cost of the three designs."),
    "t_e11": ("e11_retrieval.tex", "tab:e11", "Retrieval quality on the full BEIR corpora."),
    "t_e12": ("e12_oblivious.tex", "tab:e12", "Oblivious payload fetch."),
    "t_e15": ("supp_e15.tex", "tab:supp-e15", "Query equality under each probe mechanism."),
    "t_ledger": ("ledger.tex", "tab:ledger", "Every channel in one unit."),
    "t_census": ("supp_census.tex", "tab:supp-census", "Agent-memory frameworks, from source."),
}


def main() -> None:
    # A result file in an older shape must produce a visible placeholder, never a half-built table
    # or a crash that leaves the previous round's numbers in place.
    for fn in (
        t_e1,
        t_e2,
        t_e3,
        t_e4,
        t_e5,
        t_e6,
        t_e7,
        t_e8,
        t_e9,
        t_e10,
        t_e11,
        t_e12,
        t_e15,
        t_ledger,
        t_census,
    ):
        try:
            fn()
        except Exception as e:
            spec = NAMES[fn.__name__]
            fname, label, caption = spec[0], spec[1], spec[2]
            extra = spec[3] if len(spec) > 3 else ()
            w(fname, pending(label, caption, extra))
            print(f"  {fn.__name__}: placeholder ({type(e).__name__}: {e})")
    made = sorted(p.name for p in T.glob("*.tex"))
    print(f"wrote {len(made)} table fragments to {T.relative_to(C.repo_root())}:")
    for m in made:
        print(f"  {m}")


if __name__ == "__main__":
    main()
