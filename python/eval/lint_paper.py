"""Lint the paper source for defects that compile cleanly and render wrong.

LaTeX's failure mode of interest is not the error. It is the source that builds with zero errors
and puts something false on the page, which nobody sees unless they read the PDF character by
character.

Three of these shipped into a build here. All three came from the same place: writing LaTeX
through a shell heredoc, where ``\\times`` becomes a literal tab and ``\\to`` becomes tab-o. The
compiler is content --- a tab is whitespace, ``imes`` is a word --- so ``$2\\times$`` renders as
``2imes`` and ``$1.000 \\to 0$`` as ``1.000 o 0``. Both sat in a 35-page PDF that had just been
verified as "0 errors, 0 undefined references".

Usage::

    cd python && python -m eval.lint_paper

Exit code is non-zero if anything is found, so it belongs in ``just lint`` rather than in
somebody's memory.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from eval.lab import corpus as C

#: Commands whose backslash a shell or Python string can silently eat, leaving a control
#: character and a word fragment that still typesets. The list is the ones that actually bit us
#: plus the rest of the single-letter escape space that behaves the same way.
FRAGMENTS = {
    "imes": r"\times",
    "o ": r"\to",
    "au": r"\tau",
    "extbf": r"\textbf",
    "extit": r"\textit",
    "ho": r"\rho",
    "n ": r"\n as a newline escape",
    "rac": r"\frac",
    "elta": r"\delta",
    "lpha": r"\alpha",
}

#: Markdown that means nothing to LaTeX and renders literally.
MARKDOWN = [
    (re.compile(r"\*\*[^*\n]{2,60}\*\*"), "markdown bold; use \\textbf{...}"),
    (re.compile(r"(?<![\w`])_[a-z][\w ]{2,40}_(?![\w`])"), "markdown italics; use \\emph{...}"),
    (re.compile(r"^#{1,6} ", re.M), "markdown heading; use \\section/\\paragraph"),
]

#: Revision-history language. A journal paper must not refer to its own unpublished earlier
#: versions: the reader has not seen them, and the reference reads as an admission that the
#: present claim is replacing a wrong one.
#:
#: This is a rule rather than a habit because the habit failed twice. Sixteen sites were
#: reintroduced while writing the corrections that were supposed to remove them, and six more
#: survived that sweep -- one of them inside a *generated* table caption, where no amount of
#: reading the .tex sources would have found it, because the offending words lived in the
#: generator's f-string.
#:
#: The patterns are deliberately narrow, and the narrowness is the point. "The earlier cohort" is
#: this paper's own name for cohort H and appears on nearly every page; "whatever earlier versions
#: the environment retained" is about file versions and is the multi-snapshot channel, not a
#: draft. A linter that fires on those gets switched off, which is worse than not having one. So
#: "earlier" is only a finding when an article precedes it and a manuscript-artifact noun follows.
REVISION = [
    (
        re.compile(
            r"\b(?:an?|the)\s+earlier\s+"
            r"(?:version|draft|statement|experiment|measurement|table|finding|pass|round"
            # Nouns that in a manuscript can only mean the manuscript. Added after "an earlier
            # formulation of this attack" was written into the supplement and passed this rule.
            r"|formulation|submission|manuscript|revision)\b",
            re.I,
        ),
        "refers to an earlier version of this work",
    ),
    (
        re.compile(r"\b(?:a|the)\s+previous\s+(?:version|draft|round)\b", re.I),
        "refers to a previous version of this work",
    ),
    (
        re.compile(r"\bin\s+th(?:is|e\s+current)\s+round\b", re.I),
        "refers to this revision round",
    ),
    (
        re.compile(r"\bpreviously\s+reported\b", re.I),
        "refers to what an earlier version reported",
    ),
    (
        re.compile(r"\bthe\s+earlier\s+numbers\b", re.I),
        "refers to an earlier version's numbers",
    ),
    (
        # "That last channel is where the paper's earlier claim was too strong" sat in the
        # conclusion and passed every rule above: the article is a possessive, and "claim" was not
        # on the noun list. The possessive form is only ever about the manuscript itself.
        re.compile(
            r"\b(?:the\s+paper|this\s+paper|our)(?:'s)?\s+(?:earlier|previous|original|"
            r"former)\s+(?:claim|version|statement|result|conclusion)s?\b",
            re.I,
        ),
        "refers to what an earlier version of this work claimed",
    ),
    (
        # "This revises the claim that the attack applies unmodified to deployed stores" opened a
        # paragraph of the evaluation: a correction narrated as a correction, to a claim the
        # reader never saw.
        re.compile(
            r"\b(?:revises?|retracts?|corrects?)\s+(?:the|our)\s+(?:earlier\s+)?claim\b", re.I
        ),
        "narrates a correction to a claim the reader never saw",
    ),
    (
        # "The submitted proof closed this case with an admissibility assumption ... That
        # assumption is now removed" sat in the supplement's proof and passed every rule above.
        re.compile(
            r"\bthe\s+submitted\s+(?:proof|ledger|version|paper|manuscript|table|analysis|draft)\b"
            r"|\bis\s+now\s+removed\b",
            re.I,
        ),
        "narrates the submission history",
    ),
]


#: Stylistic tics that make prose read as boilerplate. Each
#: distinctive phrase is flagged wherever it occurs, so the human-plainness pass is enforced rather
#: than remembered. These are deliberately narrow: a phrase here is one a careful human editor would
#: not reach for twice in a paper, not ordinary connective tissue.
STYLE_TICS = [
    (re.compile(r"\bthe honest reading\b", re.I), "tic: 'the honest reading'"),
    (re.compile(r"\bworth stating plainly\b", re.I), "tic: 'worth stating plainly'"),
    (re.compile(r"\beasy to misattribute\b", re.I), "tic: 'easy to misattribute'"),
    (re.compile(r"\bthe honest (?:number|answer|split|reading)\b", re.I), "tic: 'the honest X'"),
    (re.compile(r"\bis the one that\b", re.I), "tic: 'is the one that'"),
    (re.compile(r"\bnot an art[ei]fact of\b", re.I), "tic: 'not an artifact of'"),
]

#: Phrases that are fine once or twice but read as a verbal habit past a budget per file.
STYLE_BUDGETS = [
    (re.compile(r"\bwhich is why\b", re.I), "which is why", 3),
    (re.compile(r"\bnot\b[^.,;]{1,40}\bbut\b", re.I), "not X but Y", 6),
]


def check(path: Path, root: Path) -> list[str]:
    problems: list[str] = []
    text = path.read_text(encoding="utf-8")
    # A relative path, not a bare filename: dist/ carries copies of the section files under the
    # same names, and "07-evaluation.tex:93" that could mean either is not an actionable report.
    name = path.relative_to(root).as_posix()

    for n, line in enumerate(text.split("\n"), 1):
        # Control characters. A tab in LaTeX source is legal and almost always means an escape
        # was eaten on the way in.
        for ch in line:
            if ch == "\t" or ord(ch) < 32:
                tail = line[line.index(ch) + 1 :][:6]
                guess = next(
                    (cmd for frag, cmd in FRAGMENTS.items() if tail.startswith(frag)),
                    "an eaten backslash",
                )
                problems.append(
                    f"{name}:{n}: control char U+{ord(ch):04X} followed by {tail!r} "
                    f"-- almost certainly {guess}. Renders without erroring.\n"
                    f"    {line.strip()[:100]}"
                )
                break

    for pattern, why in MARKDOWN:
        for m in pattern.finditer(text):
            n = text[: m.start()].count("\n") + 1
            problems.append(f"{name}:{n}: {why}: {m.group(0)[:60]!r}")

    for pattern, why in REVISION:
        for m in pattern.finditer(text):
            n = text[: m.start()].count("\n") + 1
            problems.append(f"{name}:{n}: {why}: {m.group(0)[:60]!r}")

    for pattern, why in STYLE_TICS:
        for m in pattern.finditer(text):
            n = text[: m.start()].count("\n") + 1
            problems.append(f"{name}:{n}: {why}: {m.group(0)[:60]!r}")

    for pattern, label, budget in STYLE_BUDGETS:
        hits = list(pattern.finditer(text))
        if len(hits) > budget:
            lines = ", ".join(str(text[: m.start()].count("\n") + 1) for m in hits[:12])
            problems.append(
                f"{name}: '{label}' used {len(hits)} times (budget {budget}); lines {lines}"
            )

    # Draft markers. These are supposed to be grepped for before submission rather than
    # remembered, which only works if something greps.
    for n, line in enumerate(text.split("\n"), 1):
        # The macro's own \newcommand is not a use of it.
        if "\\todonote" in line and "newcommand" not in line:
            problems.append(f"{name}:{n}: unresolved \\todonote")

    return problems


def main() -> None:
    root = C.repo_root()
    # paper/dist/ is build output (eval.make_dist zips these very sources, so linting it would
    # report every finding twice), and paper/_superseded/ holds the old hand-made export and
    # orphaned tables, kept only as history. Neither is anything a person maintains, and a linter
    # whose findings cannot be fixed by editing a source is a linter that gets switched off.
    skip = {"dist", "_superseded"}
    sources = sorted(
        p for p in (root / "paper").rglob("*.tex") if not skip & set(p.relative_to(root).parts)
    )
    if not sources:
        sys.exit("no .tex files found under paper/")

    problems: list[str] = []
    for path in sources:
        problems.extend(check(path, root))

    # Every result macro the prose uses must be one eval.exp2.numbers generates. The build would
    # catch an undefined one, but only when someone builds; a renamed macro in numbers.py would
    # otherwise leave the prose pointing at nothing until the next full build.
    numbers = root / "paper" / "tables" / "numbers.tex"
    defined = (
        set(re.findall(r"\\newcommand\{\\(R[A-Za-z]+)\}", numbers.read_text(encoding="utf-8")))
        if numbers.exists()
        else set()
    )
    for path in sources:
        if path == numbers:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            for used in re.findall(r"\\(R[A-Z][A-Za-z]+)", line):
                if used not in defined:
                    problems.append(
                        f"{path.relative_to(root).as_posix()}:{n}: \\{used} is not generated by "
                        "eval.exp2.numbers"
                    )

    if problems:
        print(f"{len(problems)} problem(s) in the paper source:\n")
        for p in problems:
            print(f"  {p}")
        print(
            "\nThese compile without error. That is the point: a clean build is not evidence "
            "that the page says what you wrote."
        )
        sys.exit(1)

    print(f"paper source clean ({len(sources)} files)")


if __name__ == "__main__":
    main()
