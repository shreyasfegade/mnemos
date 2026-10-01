# Mnemos task runner.  https://github.com/casey/just
#
#   just              list recipes
#   just setup        one-time toolchain + deps
#   just test         everything CI runs
#   just repro        E1-E6 on the public corpus at N=1k, < 30 min, regenerates figures
#   just demo         the live compromise visualisation

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

default:
    @just --list

# ---------------------------------------------------------------- setup

# One command, from a fresh clone, to a state where `just test` and `just repro` both work.
#
# Builds into a project-local `.venv` on purpose. The evaluation harness needs torch and
# sentence-transformers at specific versions, and installing those into a machine-wide Python is
# how you end up with a broken `transformers` that no experiment can import -- which is exactly
# what happened here, and cost a full reproduction run to diagnose.
setup:
    cargo fetch
    python -m venv .venv
    .venv/Scripts/python.exe -m pip install --upgrade pip
    .venv/Scripts/python.exe -m pip install -r requirements.lock
    .venv/Scripts/python.exe -m pip install -e ".[dev,eval]"
    .venv/Scripts/python.exe -m pip install maturin
    .venv/Scripts/python.exe -m maturin develop --release -m crates/mnemos-ffi/Cargo.toml
    @echo "Ready. 'just test' to verify, 'just verify' to check the paper's numbers."

# The same, for POSIX shells.
setup-posix:
    cargo fetch
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.lock
    .venv/bin/python -m pip install -e ".[dev,eval]" maturin
    .venv/bin/python -m maturin develop --release -m crates/mnemos-ffi/Cargo.toml

# ---------------------------------------------------------------- build

build:
    cargo build --workspace --release

# The FFI extension module. Needed from Stage 4 (the MCP server).
build-py:
    maturin develop --release -m crates/mnemos-ffi/Cargo.toml

# ---------------------------------------------------------------- test

test: test-rust test-python lint

test-rust:
    cargo test --workspace

# The history-independence property test is the machine-checked form of Definition 1(iii) --
# the paper's core hypothesis. It gets more cases than the default and runs on its own so a
# failure is impossible to miss in the output.
test-hi:
    cargo test -p mnemos-store --release -- --ignored layout_is_independent_of_insertion_order

# Asserts the emitted LeakEvent set matches docs/leakage-profile.md, row by row.
# If this fails, the implementation leaks something the paper does not admit to.
test-leakage:
    cargo test -p mnemos-engine --test leakage_profile

# Asserts the leakage model against the DISK rather than against itself: snapshot the store,
# perform one operation, diff it byte-for-byte, and fail if any changed range has no event
# describing it. Findings F5 and F6 were both channels that emitted nothing, and a test suite
# that reads events cannot see those. Each test here carries a negative control.
test-channels:
    cargo test -p mnemos-engine --test channel_enumeration

test-python:
    pytest python/tests -v

# ---------------------------------------------------------------- quality

lint: lint-rust lint-python lint-paper

# Catches LaTeX source that compiles with zero errors and renders wrong: escapes eaten by a
# shell heredoc (`	imes` -> a literal tab, so `$2	imes$` typesets as "2imes"), markdown that
# means nothing to LaTeX, and unresolved draft notes. Three of these shipped into a build that
# had just been verified as "0 errors, 0 undefined references".
lint-paper:
    python -m eval.lint_paper

lint-rust:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings

lint-python:
    ruff check python
    ruff format --check python
    mypy python/mnemos

audit:
    cargo audit
    cargo deny check

# ---------------------------------------------------------------- experiments

# E2 -- performance. Feeds every latency number in the paper.
# The paper's performance numbers, at the paper's configuration.
#
# The dimension and the corpus sizes are pinned here rather than left to the binary's defaults,
# because the defaults are dim=128 and the paper reports dim=768 -- and a table whose caption
# says one thing while its numbers were measured at another is not a hypothetical failure here,
# it is what the write row of `tab:perf` did until 2026-08-29.
#
# Two baselines: `frozen-s1` is the pre-mitigation configuration the security results were
# measured against, `frozen-s2` the shipped default. `just perf-table` reads both.
#
# Run on an idle machine. A contended core once turned a super-linear cost into a sub-linear one.
bench:
    MNEMOS_BENCH_DIM=768 MNEMOS_BENCH_SIZES=1000,10000 MNEMOS_BENCH_SPREAD=1 cargo bench -p mnemos-bench --bench search -- --save-baseline frozen-s1
    MNEMOS_BENCH_DIM=768 MNEMOS_BENCH_SIZES=1000,10000 MNEMOS_BENCH_SPREAD=2 cargo bench -p mnemos-bench --bench search -- "search/mnemos" --save-baseline frozen-s2

# Regenerate paper/tables/*.tex from criterion's output. Never edit those files by hand.
perf-table:
    python -m eval.figures.emit_perf_table --baseline frozen-s1 --baseline-s2 frozen-s2

# E3 -- attack A3 on the forward-private baselines vs Mnemos. GATE G2 and G4.
e3:
    python -m eval.experiments.e3_snapshot

# E6 -- cohort asymmetry with confidence intervals. A view over E4's bootstrap, not a
# second measurement of the same quantity.
e6:
    python -m eval.experiments.e6_delta

# Reproduction: the load-bearing experiments on the public corpus (D2).
# This is what a reviewer runs, so it must stay working. Every headline number it produces is
# checked against the range the paper claims for it -- running is not reproducing.
repro:
    python -m eval.repro --quick

# Everything: all N, all sweeps, all figures. Hours.
repro-full:
    python -m eval.repro --full

# Check the results already on disk against the paper's claims. Runs nothing, takes a second.
# Run this before building the paper. This is eval.exp2.verify: eval.repro --verify encodes the
# claims of the earlier E1-E6 harness (adv_hist = 0.587, query equality closed at s = 2), which
# the paper no longer makes, so a pass there certifies nothing about the paper as it stands.
verify:
    python -m eval.exp2.verify

# Regenerate every figure from the results on disk.
figures:
    python -m eval.figures.emit_all

# ---------------------------------------------------------------- review round 2 (2026-09-16)
#
# The experiment set for the current paper. Numbering follows the review notes,
# which is NOT the numbering in the evaluation plan or in round 1 -- see the banner at the top of
# the evaluation plan for the mapping between the three.

# Everything that needs no LLM and no WSL. Hours; safe to run unattended.
exp2:
    python -m eval.exp2.e4_drift
    python -m eval.exp2.e5_game --instances 1000
    python -m eval.exp2.e7_permutation
    python -m eval.exp2.e8_multisnapshot
    python -m eval.exp2.e9_refusal
    python -m eval.exp2.e11_retrieval
    python -m eval.exp2.e13_artifacts

# Latency-sensitive. Run these on an OTHERWISE IDLE machine or the numbers are contention, not cost.
exp2-perf:
    python -m eval.exp2.e10_performance
    python -m eval.exp2.e12_oblivious

# Needs a local model (Ollama). Generation is the long pole; the attack then replays the cache.
exp2-agent:
    python -m eval.exp2.e2_agent_attack --generate --turns 600
    python -m eval.exp2.e2_agent_attack
    python -m eval.exp2.e3_dedup
    python -m eval.exp2.e1_longitudinal

# Needs WSL with gocryptfs on PATH (~/bin/gocryptfs). Deployed stores under a FUSE mount.
exp2-stores:
    python -m eval.exp2.e6_stores --n 3000 --hist 300 --budget 30

# Gather every round-2 result and audit that each point estimate carries an interval.
exp2-collect:
    python -m eval.exp2.collect

# Kept as an alias so older notes that say `just verify2` still work; `just verify` is the same
# check now that it no longer runs the earlier harness's ranges.
verify2: verify

# ---------------------------------------------------------------- paper
#
# NOTHING in the paper is typed by hand: an experiment that has not run emits a visible
# placeholder rather than a stale number.
#
# Regenerate every table and figure from the result JSONs, then build and CHECK the build.
# `eval.build_paper` runs pdflatex/bibtex/pdflatex x2 and the supplement (which reads main.aux via
# xr), then fails unless each log was written by this run, no LaTeX error was recovered from, and
# no reference or citation is undefined. Not latexmk: MiKTeX ships no Perl, and a latexmk that
# cannot start leaves the previous log behind, so a stale page count reads as a fresh build.
paper:
    python -m eval.exp2.ledger
    python -m eval.exp2.make_tables
    python -m eval.exp2.numbers
    python -m eval.exp2.make_figures
    python -m eval.build_paper

# The submission bundle: the two PDFs and the LaTeX sources they were built from, and nothing
# else. It refuses to exist unless the verifier passes, the lint gate is clean and every number
# the prose quotes has been measured (--strict), so a stale export -- which is what paper/dist/
# was, with retracted numbers and revision-history language in it -- cannot be produced again.
dist: verify lint-paper paper
    python -m eval.exp2.numbers --strict
    python -m eval.check_response
    python -m eval.make_dist
    python -m eval.make_docx

# ---------------------------------------------------------------- artifacts

# The Stage 3 deliverable: side-by-side live compromise, three minutes, unattended.
demo:
    cargo run --release -p mnemos-cli -- demo --port 8080

# Regenerate the paper's configuration table from params.toml, so the paper and the
# implementation cannot drift apart on a number.
paper-params:
    cargo run --release -p mnemos-cli -- params > paper/generated/params-table.txt

clean:
    cargo clean
    rm -rf python/eval/results/*.json paper/figures/*.pdf
