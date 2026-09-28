"""Reproduction driver -- what a reviewer runs, and what catches us lying to ourselves.

    just repro        the load-bearing experiments on D2. Tens of minutes on a laptop.
    just repro-full   everything, every sweep, all figures. Hours.
    python -m eval.repro --verify   re-check the results already on disk, run nothing

.. warning::
   **The ranges below defend the PREVIOUS paper's claims (2026-09-15 and earlier).** The
   2026-09-16 round retired or reversed several of them, so ``--verify`` here will fail, or
   worse pass, against sentences the manuscript no longer contains. Concretely:

   * ``E4 adv_hist = 0.587`` and ``adv_hist/adv_post = 0.943`` are retired -- the transfer lemma
     replaced the cohort ratio as the headline.
   * ``E9 'does not resolve above chance at any budget tested'`` is **false** for Mnemos-P: the
     write channel is open and stated as such.
   * ``E10 'query equality closed outright at s = 2'`` is **reversed** -- an overlap-threshold
     adversary reaches Youden's J = 0.98.

   The current verifier is :mod:`eval.exp2.verify` (``just verify2``), whose checks cite the
   current paper. This file is kept because its preflight, provenance stamping and the five
   measurement failures documented below are still the reason any of this exists -- and because
   deleting the record of a superseded claim is how a project forgets it ever made one.

Artifact evaluation is a real score awarded by real people with limited patience, so the quick
path exists and has to stay quick. That is the boring half of this file's job.

The other half
==============
**Every number this harness produces is checked against the range the paper claims for it.**
Running an experiment and writing its JSON is not reproduction; it is exercise. A pipeline that
only runs cannot distinguish a correct measurement from a confident, plausible, wrong one -- and
this project has produced five of those, none detectable from its own output:

- E11 measured through ``MnemosMemoryServer.put``, which embeds its argument. Every corpus size
  came out at ~7,600 us/entry, perfectly flat. The flat curve was a measurement of
  ``bge-base-en-v1.5``, and it "confirmed" a linearity claim the write path does not satisfy. The
  paper's own performance table says ``put`` costs ~72 us. **A range check on that one number
  would have failed instantly.**
- E11 measured again while a criterion benchmark held a core: the smallest point came out 6x high
  and dragged the fitted exponent negative, turning a super-linear cost into a sub-linear one.
- E9's trace parser assumed externally-tagged JSON against a flat ``{"kind": ...}`` trace, matched
  nothing, and reported ``Adv = 0.000`` -- which reads exactly like a successful defence.
- E9's intersection attack grouped by topic, but a topic spans many partitions, so the
  intersection was empty by construction. Also 0.000. Also looked like a defence.
- E10's recovery metric was gameable: at ``s = 8`` the probe set covered the whole index and
  "always say the same thing" scored 1.000.

Three of those returned "nothing found" and two returned a clean wrong exponent. Every one was
caught by an **external** number the result had to agree with, never by the result itself. So the
checks below are not smoke tests. Each one states the paper claim it defends, the range, and why
the range is what it is. A check with a range nobody can justify is worse than no check.

Non-negotiables
---------------
**Runs on the public corpus (D2) alone.** No Hermes, no private data, no credentials. If
``just repro`` needs anything a stranger cannot obtain, it is broken.

**Every result carries its provenance**: ``params_hash``, git SHA, seed, machine description,
and -- critically -- the **keystore provider**. A run on ``software-INSECURE`` is not a security
result, and six weeks later nobody will remember which JSON came from which machine.

**Failures are loud.** A missing dataset, a failed experiment or an out-of-range headline number
prints, fails the run and exits non-zero. Silently producing a partial result set is how a figure
ends up in a paper with half its points missing.

**Nothing runs concurrently.** Experiments run one at a time, in-process cores uncontended. This
is not tidiness: see the second bullet above.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mnemos import params

RESULTS = Path("python/eval/results")

#: The public corpus every claim in the quick path is made against.
#:
#: Resolved to an absolute path before it reaches an experiment. Experiments are launched with
#: ``cwd = <root>/python`` so their imports work, and a repo-relative path handed across that
#: boundary resolves to ``python/data/...`` and fails with a `FileNotFoundError` naming a file
#: nobody ever intended to open.
D2_REL = Path("data/d2-beir/notes.jsonl")


def d2_path(root: Path) -> str:
    return str((root / D2_REL).resolve())


# =================================================================================================
# Provenance
# =================================================================================================


@dataclass(frozen=True)
class RunProvenance:
    """Stamped onto every result set."""

    params_hash: str
    git_sha: str
    machine: str
    python_version: str
    keystore_provider: str
    """``software-INSECURE`` here means these are NOT security results."""
    native_extension: str
    """Absolute path of the compiled extension that actually loaded.

    Recorded because it was once wrong and nothing said so. A stale ``mnemos_native.pyd`` sat in
    ``python/`` and shadowed the installed build, since the documented workflow runs from that
    directory and cwd precedes site-packages on ``sys.path``. Experiments ran against a binary
    from a previous build, silently. A results file that cannot say which engine produced it is
    a results file you have to take on trust."""
    native_mtime: str
    """Build time of that extension. A number measured against a week-old engine is not a
    number about the current one, and the two are indistinguishable without this."""
    quick: bool
    started: str


def native_extension() -> tuple[str, str]:
    """Where the compiled extension loaded from, and when it was built."""
    try:
        import mnemos_native

        path = Path(mnemos_native.__file__).resolve()
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime))
        return str(path), stamp
    except Exception:
        return "unknown", "unknown"


def keystore_provider() -> str:
    """What actually guarded the key, asked of the engine rather than assumed.

    A hardcoded string here would be the same class of defect as a leakage table that is not a
    test: a claim about the system, stored next to the system, never compared to it.
    """
    try:
        import mnemos_native

        return str(mnemos_native.keystore_provider())
    except Exception:
        return "unknown"


def provenance(quick: bool) -> RunProvenance:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        sha = "unknown"

    native_path, native_built = native_extension()
    return RunProvenance(
        params_hash=params.params_hash(),
        native_extension=native_path,
        native_mtime=native_built,
        git_sha=sha,
        machine=f"{platform.system()} {platform.release()} {platform.machine()}",
        python_version=platform.python_version(),
        keystore_provider=keystore_provider(),
        quick=quick,
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


# =================================================================================================
# Checks
# =================================================================================================


@dataclass(frozen=True)
class Check:
    """One headline number, the range the paper commits it to, and why."""

    name: str
    claim: str
    """Where in the paper this number appears. If it appears nowhere, it does not belong here."""
    lo: float
    hi: float
    why: str
    """Why *these* bounds. A range nobody can justify is worse than no check at all."""


@dataclass
class CheckResult:
    check: Check
    value: float | None
    ok: bool
    note: str = ""


def _rows(path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["rows"] if isinstance(d, dict) and "rows" in d else d


def _load(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


# --- extractors -----------------------------------------------------------------------------
#
# One per experiment. Each returns {check name: measured value}. Kept separate from the ranges so
# that changing what is measured and changing what is claimed are two different edits.


def extract_e4(results: Path) -> dict[str, float]:
    rows = _rows(results / "e4_transfer_bge_d2-beir.json")
    padded = [r for r in rows if not r["padded"]]
    if not padded:
        return {}
    best = max(padded, key=lambda r: r["adv_hist"])
    return {
        "E4 adv_hist (D2, unpadded)": best["adv_hist"],
        "E4 adv_hist/adv_post": best["ratio"],
        "E4 chance floor": 1.0 / best["n_topics"],
    }


def extract_e7(results: Path) -> dict[str, float]:
    rows = _rows(results / "e7_agent_utility.json")
    by = {r["retriever"]: r for r in rows}
    if "mnemos" not in by or "plaintext-ivf" not in by:
        return {}
    delta = (by["mnemos"]["recall_at_10"] - by["plaintext-ivf"]["recall_at_10"]) * 100.0
    return {
        "E7 Recall@10 (Mnemos)": by["mnemos"]["recall_at_10"],
        "E7 Recall@10 delta (pp)": delta,
    }


def extract_e9(results: Path) -> dict[str, float]:
    rows = _rows(results / "e9_transfer_live.json")
    out: dict[str, float] = {}
    for padded, label in ((True, "padded"), (False, "F5 build")):
        sel = [r for r in rows if r["index_write_padding"] is padded]
        if not sel:
            continue
        # Per budget, then the **maximum over budgets** -- which is what the paper's sentences
        # mean. "reaches adv_hist = 0.367" is a claim about the best budget, and "does not
        # resolve above chance at any budget tested" is a claim about the worst case for the
        # defence. Both are the max. Pooling every budget together instead answers a third
        # question neither sentence asks, and passes a range check while doing it.
        topics = sel[0]["n_topics"]
        best = 0.0
        for budget in sorted({r["budget"] for r in sel}):
            at = [r for r in sel if r["budget"] == budget]
            correct = sum(r["hist_correct"] for r in at)
            total = sum(r["hist_total"] for r in at)
            if total:
                best = max(best, correct / total - 1.0 / topics)
        out[f"E9 adv_hist ({label})"] = max(0.0, best)
    return out


def extract_e10(results: Path) -> dict[str, float]:
    rows = _rows(results / "e10_probe_spread.json")
    by_s = {r["spread"]: r for r in rows if r["retention"] == 1.0}
    out: dict[str, float] = {}
    if 1 in by_s:
        out["E10 query equality, s=1"] = by_s[1]["advantage"]
    if 2 in by_s:
        out["E10 query equality, s=2"] = by_s[2]["advantage"]
        out["E10 read volume factor, s=2"] = (
            by_s[2]["bucket_reads_per_search"] / by_s[1]["bucket_reads_per_search"]
            if 1 in by_s
            else float("nan")
        )
        out["E10 Recall@10 delta, s=2 (pp)"] = by_s[2]["recall_delta_pp"]
    return out


def extract_e11(results: Path) -> dict[str, float]:
    rows = _rows(results / "e11_write_scaling.json")
    if len(rows) < 2:
        return {}
    import math

    xs = [math.log(r["n"]) for r in rows]
    ys = [math.log(r["per_entry_us"]) for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom if denom else 0.0
    return {
        "E11 per-entry write (us)": rows[-1]["per_entry_us"],
        "E11 growth exponent": slope,
    }


def extract_e5(results: Path) -> dict[str, float]:
    d = _load(results / "e5_volumetric.json")
    assert isinstance(d, dict)
    return {
        "E5 volume recovery": d["volume_recovery"],
        "E5 volume variance": d["volume_variance"],
    }


# =================================================================================================
# Experiments
# =================================================================================================


@dataclass(frozen=True)
class Experiment:
    key: str
    module: str
    args: list[str]
    produces: list[str]
    """Result files it must leave behind. A missing one is a failure, not a warning."""
    extract: Callable[[Path], dict[str, float]]
    checks: list[Check] = field(default_factory=list)
    quick: bool = True
    minutes: int = 5


EXPERIMENTS: list[Experiment] = [
    Experiment(
        key="e5",
        module="eval.experiments.e5_volumetric",
        args=[],
        produces=["e5_volumetric.json"],
        extract=extract_e5,
        minutes=2,
        checks=[
            Check(
                name="E5 volume recovery",
                claim="sec:e5 -- 'capped near 1/n_topics by construction'",
                lo=0.0,
                hi=0.20,
                why="Constant-volume retrieval means the volume channel carries nothing, so "
                "recovery must sit at the chance floor for 12 queries (0.083). Anything above "
                "0.20 means a search became data-dependent and the invariant is broken.",
            ),
            Check(
                name="E5 volume variance",
                claim="tab:leakage -- 'Search volume: constant, query-independent'",
                lo=0.0,
                hi=0.0,
                why="Exactly zero, not approximately. Variance in read volume across queries is "
                "the volumetric channel; a single non-zero digit here is a defect.",
            ),
        ],
    ),
    Experiment(
        key="e4",
        module="eval.experiments.e4_transfer",
        args=["--backend", "bge", "--corpus", "{D2}"],
        produces=["e4_transfer_bge_d2-beir.json"],
        extract=extract_e4,
        minutes=25,
        checks=[
            Check(
                name="E4 adv_hist (D2, unpadded)",
                claim="abstract -- 'an advantage of adv_hist = 0.587'",
                lo=0.45,
                hi=0.70,
                why="The headline attack result. Bounds are the cluster-bootstrap interval "
                "widened to absorb a different k-means seed, which moves partition boundaries "
                "and therefore the map. Below 0.45 the attack no longer supports the paper's "
                "claim; above 0.70 something is leaking that was not leaking before.",
            ),
            Check(
                name="E4 adv_hist/adv_post",
                claim="abstract -- 'adv_hist/adv_post = 0.943 [0.916, 0.970]'",
                lo=0.85,
                hi=1.05,
                why="The paper's actual contribution: being written before the adversary "
                "arrived buys almost nothing. A ratio near 1.0 is the finding. If this drops "
                "below 0.85 the historical cohort is meaningfully safer and the claim weakens; "
                "it can legitimately exceed 1.0, which is why the upper bound is not 1.0.",
            ),
            Check(
                name="E4 chance floor",
                claim="abstract -- 'against a 0.125 chance floor'",
                lo=0.124,
                hi=0.126,
                why="D2 has exactly 8 BEIR sub-corpora, one topic each. This is not really a "
                "measurement -- it is a guard against the corpus silently changing shape under "
                "an experiment whose numbers are quoted relative to it.",
            ),
        ],
    ),
    Experiment(
        key="e9",
        module="eval.experiments.e9_transfer_live",
        args=[],
        produces=["e9_transfer_live.json"],
        extract=extract_e9,
        minutes=20,
        checks=[
            Check(
                name="E9 adv_hist (padded)",
                claim="abstract -- 'does not resolve above chance at any budget tested'",
                lo=0.0,
                hi=0.05,
                why="A4 driven against the running engine, with index-write padding on. This is "
                "the defence claim, and it is the one number in the paper where a *low* value "
                "is the result -- so it is also the one most easily produced by a broken "
                "harness. E9's parser once reported 0.000 because it matched nothing at all. "
                "The paired check below is what makes this one mean something.",
            ),
            Check(
                name="E9 adv_hist (F5 build)",
                claim="abstract -- 'the same attack reaches adv_hist = 0.367 against that build'",
                lo=0.20,
                hi=0.55,
                why="The ablation with index_write_padding disabled, reproducing finding F5. "
                "**This check exists to validate the one above it.** If the attack fails here "
                "too, the harness is broken rather than the attack defeated -- which is exactly "
                "how a parser bug once read as a successful defence.",
            ),
        ],
    ),
    Experiment(
        key="e10",
        module="eval.experiments.e10_probe_spread",
        args=[],
        produces=["e10_probe_spread.json"],
        extract=extract_e10,
        minutes=15,
        checks=[
            Check(
                name="E10 query equality, s=1",
                claim="sec:mitigation -- 'A5 recovers query equality at 1.000'",
                lo=0.90,
                hi=1.00,
                why="The residual channel the mitigation exists to close, measured with the "
                "mitigation off. Youden's J, so 1.000 means perfect separation and 0 means "
                "chance. If this is not near 1.0 the mitigation is solving a problem the "
                "measurement cannot see.",
            ),
            Check(
                name="E10 query equality, s=2",
                claim="sec:mitigation -- 'closed outright at s = 2'",
                lo=0.0,
                hi=0.10,
                why="The shipped default. Pairwise query equality must be gone, not reduced.",
            ),
            Check(
                name="E10 Recall@10 delta, s=2 (pp)",
                claim="sec:mitigation -- the mitigation is shipped on by default",
                lo=-1.0,
                hi=10.0,
                why="The mitigation is the *default* in params.toml, so it has to be shown not "
                "to cost utility. Widening the probe set reads strictly more partitions, so "
                "recall should hold or improve; a negative delta beyond noise would mean the "
                "randomisation is dropping real probes, which is `probe_retention`'s job and "
                "not this knob's.",
            ),
            Check(
                name="E10 read volume factor, s=2",
                claim="sec:mitigation -- 'costs 2x read volume'",
                lo=2.0,
                hi=2.0,
                why="Exactly 2x. The mitigation multiplies the read set by a constant; if this "
                "is not exact, the count has become query-dependent and a query-equality leak "
                "has been traded for a volumetric one.",
            ),
        ],
    ),
    Experiment(
        key="e7",
        module="eval.experiments.e7_agent_utility",
        args=["--corpus", "{D2}"],
        produces=["e7_agent_utility.json"],
        extract=extract_e7,
        minutes=20,
        checks=[
            Check(
                name="E7 Recall@10 delta (pp)",
                claim="abstract -- 'retrieves within 0.7 percentage points of a plaintext index'",
                lo=-2.0,
                hi=5.0,
                why="The utility budget declared in params.toml is 2 percentage points, so the "
                "lower bound is the budget itself rather than the measured -0.7. The upper "
                "bound is loose on purpose: beating plaintext is possible (probe widening "
                "reads more partitions) and is not a failure.",
            ),
            Check(
                name="E7 Recall@10 (Mnemos)",
                claim="tab:utility",
                lo=0.60,
                hi=1.00,
                why="An absolute floor under the relative check above. Two retrievers that are "
                "both broken agree with each other perfectly, and the delta check alone would "
                "pass.",
            ),
        ],
    ),
    Experiment(
        key="e11",
        module="eval.experiments.e11_write_scaling",
        args=[],
        produces=["e11_write_scaling.json"],
        extract=extract_e11,
        quick=False,
        minutes=10,
        checks=[
            Check(
                name="E11 per-entry write (us)",
                claim="tab:perf -- 'put (amortised), 72.3 us'",
                lo=20.0,
                hi=500.0,
                why="**The check that would have caught the worst measurement error in this "
                "project.** E11 once measured 7,600 us/entry, flat across every corpus size, "
                "because it went through a code path that embeds its argument -- so the curve "
                "was a measurement of bge-base-en-v1.5 rather than of the write path. The "
                "criterion benchmark says ~72 us. Anything two orders of magnitude away from "
                "that is measuring something else, whatever it looks like.",
            ),
            Check(
                name="E11 growth exponent",
                claim="sec:e11 -- 'per-entry cost goes as n^0.07'",
                lo=0.0,
                hi=0.5,
                why="Per-entry write cost is predicted to grow slowly but *not* to shrink. A "
                "negative exponent is the signature of a contended machine inflating the "
                "smallest point, which is how this was measured wrongly the first time. Run "
                "this on an idle machine or do not run it.",
            ),
        ],
    ),
]


# =================================================================================================
# Driver
# =================================================================================================


def preflight(root: Path) -> list[str]:
    """Things that make a run pointless, checked before spending an hour discovering them."""
    problems: list[str] = []

    if not (root / D2_REL).exists():
        problems.append(
            f"missing corpus {D2_REL}. Build it with `python data/fetch_d2.py` -- it downloads "
            "public BEIR sub-corpora and needs no credentials."
        )

    try:
        import mnemos_native  # noqa: F401
    except ImportError:
        problems.append(
            "the native extension is not importable. Build it with `just build` "
            "(maturin develop --release -m python/Cargo.toml)."
        )

    try:
        params.params_hash()
    except Exception as exc:
        problems.append(f"params.toml is unreadable: {exc}")

    # The embedding backend, checked here rather than discovered later.
    #
    # Four of the five experiments below need a real model, and each one raises only when it
    # reaches its first embedding call. In a run ordered as this one is, that is twenty-five
    # minutes of E4 before the first failure -- and the failure is an environment problem, not
    # a result. A preflight that checks the corpus and the extension but not the model checks
    # two thirds of what a run needs.
    try:
        from mnemos.embed import embed, require_semantic_backend

        # Embed a one-token probe *first*. `ACTIVE_BACKEND` is set by `embed()`, so asking
        # `require_semantic_backend()` before anything has been embedded always answers 'none'
        # -- a preflight that fails every run, including the good ones, while looking correct
        # because the failure it reports is the one you were expecting to see.
        #
        # This also does the real work: it forces the model to load and download if absent, so
        # a first run pays that cost here rather than twenty minutes into E4.
        embed(["preflight"])
        require_semantic_backend()
    except ImportError:
        problems.append("mnemos.embed is not importable; is the package installed?")
    except Exception as exc:
        problems.append(
            f"no real embedding backend: {exc} "
            "E5 alone would still be valid (it is a security measurement and does not embed); "
            "run `python -m eval.repro --quick --only e5` for that."
        )

    return problems


def run_experiment(exp: Experiment, root: Path) -> tuple[bool, float, str]:
    """Run one experiment to completion. Returns (ok, seconds, message)."""
    # `{D2}` is substituted here rather than at import time, because the corpus path has to be
    # absolute and the repo root is not known to a module-level constant.
    cmd = [sys.executable, "-m", exp.module, *(a.format(D2=d2_path(root)) for a in exp.args)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=root / "python", check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        return False, elapsed, f"exit {proc.returncode}"
    missing = [f for f in exp.produces if not (root / RESULTS / f).exists()]
    if missing:
        return False, elapsed, f"produced no {missing}"
    return True, elapsed, ""


def verify(exp: Experiment, root: Path) -> list[CheckResult]:
    """Compare an experiment's headline numbers against the ranges the paper claims."""
    try:
        measured = exp.extract(root / RESULTS)
    except FileNotFoundError as exc:
        return [CheckResult(c, None, False, f"no results: {exc}") for c in exp.checks]
    except (KeyError, ValueError, TypeError) as exc:
        # A malformed result file is a failure, never a skip. This is where "the parser matched
        # nothing" has to stop being indistinguishable from "the attack found nothing".
        return [CheckResult(c, None, False, f"unreadable results: {exc}") for c in exp.checks]

    out: list[CheckResult] = []
    for c in exp.checks:
        v = measured.get(c.name)
        if v is None:
            out.append(CheckResult(c, None, False, "not present in results"))
            continue
        out.append(CheckResult(c, v, c.lo <= v <= c.hi))
    return out


def report(results: list[CheckResult]) -> bool:
    width = max((len(r.check.name) for r in results), default=10)
    print(f"\n{'':2} {'claim':<{width}}  {'measured':>10}  {'expected':>18}")
    print("  " + "-" * (width + 34))
    for r in results:
        mark = "ok" if r.ok else "!!"
        val = "--" if r.value is None else f"{r.value:.4f}"
        rng = f"[{r.check.lo:.3f}, {r.check.hi:.3f}]"
        print(f"{mark:2} {r.check.name:<{width}}  {val:>10}  {rng:>18}")
    failed = [r for r in results if not r.ok]
    if failed:
        print("\nOUT OF RANGE -- these numbers do not reproduce the paper's claims:\n")
        for r in failed:
            print(f"  {r.check.name}")
            print(f"    claim:    {r.check.claim}")
            print(f"    measured: {'--' if r.value is None else r.value}  {r.note}")
            print(f"    why the range is what it is:\n      {r.check.why}\n")
    return not failed


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce and verify Mnemos experiments.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="the load-bearing experiments on D2")
    mode.add_argument("--full", action="store_true", help="everything; hours")
    mode.add_argument(
        "--verify",
        action="store_true",
        help="check the results already on disk against the paper's claims; run nothing",
    )
    ap.add_argument("--only", default=None, help="comma-separated experiment keys, e.g. e9,e10")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out = root / RESULTS
    out.mkdir(parents=True, exist_ok=True)

    selected = [e for e in EXPERIMENTS if args.full or args.verify or e.quick]
    if args.only:
        keys = {k.strip() for k in args.only.split(",")}
        selected = [e for e in selected if e.key in keys]
    if not selected:
        sys.exit("no experiments selected")

    if not args.verify:
        problems = preflight(root)
        if problems:
            print("cannot run:\n")
            for p in problems:
                print(f"  - {p}")
            sys.exit(1)

        prov = provenance(quick=args.quick)
        (out / "provenance.json").write_text(json.dumps(asdict(prov), indent=2), encoding="utf-8")
        print(f"params  {prov.params_hash}")
        print(f"git     {prov.git_sha}")
        print(f"machine {prov.machine}")
        print(f"keys    {prov.keystore_provider}")
        print(f"native  {prov.native_extension}  (built {prov.native_mtime})")
        if prov.keystore_provider != "windows-cng-pcp":
            print(
                "\n  NOTE: not a hardware keystore. These are performance and leakage results,\n"
                "  not key-custody results. Every figure produced here carries this line in\n"
                "  provenance.json so the distinction survives the run.\n"
            )
        budget = sum(e.minutes for e in selected)
        print(f"\nrunning {len(selected)} experiments, roughly {budget} minutes, one at a time.")
        print("Do not run anything else on this machine. A contended core once turned a")
        print("super-linear cost into a sub-linear one without any other symptom.\n")

    all_checks: list[CheckResult] = []
    failures: list[str] = []

    for exp in selected:
        if not args.verify:
            print(f"  {exp.key} ...", end="", flush=True)
            ok, secs, msg = run_experiment(exp, root)
            print(f" {secs / 60:.1f} min" + ("" if ok else f"  FAILED: {msg}"))
            if not ok:
                failures.append(f"{exp.key}: {msg}")
                continue
        all_checks.extend(verify(exp, root))

    ok = report(all_checks)

    if not args.verify:
        print("\nregenerating figures ...")
        fig = subprocess.run(
            [sys.executable, "-m", "eval.figures.emit_all"], cwd=root / "python", check=False
        )
        if fig.returncode != 0:
            failures.append("figure generation failed")

    if failures:
        print("\nfailed experiments:")
        for f in failures:
            print(f"  - {f}")

    if failures or not ok:
        print("\nREPRODUCTION FAILED. The numbers on disk do not match the paper's claims.")
        sys.exit(1)
    print("\nAll headline numbers reproduce within their claimed ranges.")


if __name__ == "__main__":
    main()
