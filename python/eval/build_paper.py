"""Build the paper and the supplement, and fail on anything that is not a clean build.

Three failures shipped past a hand-run build in one session, and none showed in the exit code:

* ``latexmk`` could not run (MiKTeX ships no Perl) and left the previous ``main.log`` in place, so
  grepping it for the page count reported the stale build as current;
* a command line carrying a backslashed regex was blocked before it ran, with the same result;
* a generated table put an unescaped ``_`` into text mode, and pdflatex recovered from the
  "Missing $" error and wrote a PDF anyway.

So this script checks what the exit code does not: that each log was written by *this* run, that
no LaTeX error was recovered from, and that no reference or citation is undefined. It reports the
page counts it verified and the overfull boxes it found.

Usage::

    cd python && python -m eval.build_paper
"""

from __future__ import annotations

import re
import subprocess
import sys
import time

from eval.lab import corpus as C

PAPER = C.repo_root() / "paper"

STEPS = (
    ("pdflatex", "main"),
    ("bibtex", "main"),
    ("pdflatex", "main"),
    ("pdflatex", "main"),
    # The supplement reads main.aux through xr, so it is built after main is final.
    ("pdflatex", "supplement"),
    ("pdflatex", "supplement"),
)


def run(tool: str, job: str) -> None:
    args = [tool, job] if tool == "bibtex" else [tool, "-interaction=nonstopmode", f"{job}.tex"]
    subprocess.run(
        args, cwd=PAPER, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )


def check(job: str, started: float) -> tuple[list[str], str]:
    log = PAPER / f"{job}.log"
    problems: list[str] = []
    if not log.exists():
        return [f"{job}.log does not exist: the build did not run"], ""
    if log.stat().st_mtime < started:
        return [f"{job}.log predates this build: the build did not run, and the log is stale"], ""
    text = log.read_text(encoding="utf-8", errors="ignore")
    for m in re.finditer(r"^! (.+)$", text, re.M):
        ctx = text[m.end() : m.end() + 160].strip().splitlines()
        where = next((line for line in ctx if line.startswith("l.")), "")
        problems.append(f"{job}: LaTeX error '{m.group(1)}' {where[:100]}")
    for m in re.finditer(r"(Reference|Citation) [`'](.+?)' on page \d+ undefined", text):
        problems.append(f"{job}: undefined {m.group(1).lower()} {m.group(2)}")
    if "multiply defined" in text:
        problems.append(f"{job}: multiply-defined labels")
    pages = re.search(r"Output written on \S+ \((\d+) pages?", text)
    overfull = len(re.findall(r"^Overfull \\hbox", text, re.M))
    summary = f"{job}.pdf: {pages.group(1) if pages else '?'} pages, {overfull} overfull boxes"
    if not pages:
        problems.append(f"{job}: no PDF was written")
    return problems, summary


def main() -> None:
    started = time.time() - 1.0
    for tool, job in STEPS:
        run(tool, job)
    problems: list[str] = []
    for job in ("main", "supplement"):
        p, summary = check(job, started)
        problems.extend(p)
        if summary:
            print(summary)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    print("clean build: fresh logs, no LaTeX errors, no undefined references")


if __name__ == "__main__":
    main()
