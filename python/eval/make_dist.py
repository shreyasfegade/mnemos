"""Build the submission bundle in ``paper/dist/`` from the sources as they stand.

    cd python && python -m eval.make_dist      (or: just dist, which gates it first)

The previous ``paper/dist/`` was a hand-made export that outlived the text it was made from: it
carried retracted numbers and sixteen instances of revision-history language, and nothing
regenerated it. This module makes the bundle a build product. It copies only what a reviewer or an
editor needs -- the two PDFs, and a source archive holding exactly the files the two ``.tex``
roots include -- and refuses to run on a build that is older than its sources, so the bundle can
never again describe a paper that no longer exists.

It does not gate on the verifier or the lint itself; ``just dist`` runs those first, and runs this
only if they pass.
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

from eval.lab import corpus as C

PAPER = C.repo_root() / "paper"
DIST = PAPER / "dist"
ROOTS = ("main", "supplement")
INCLUDE = re.compile(r"\\(?:input|include|includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}")


def included(tex: Path, seen: set[Path]) -> None:
    """Every file ``tex`` pulls in, recursively, as paths relative to paper/."""
    for m in INCLUDE.finditer(tex.read_text(encoding="utf-8")):
        name = m.group(1)
        for cand in (PAPER / name, PAPER / f"{name}.tex", PAPER / f"{name}.pdf"):
            if cand.is_file():
                # Already collected through another root (numbers.tex is in both preambles).
                if cand not in seen:
                    seen.add(cand)
                    if cand.suffix == ".tex":
                        included(cand, seen)
                break
        else:
            raise SystemExit(f"{tex.relative_to(PAPER)} includes {name}, which does not exist")


def main() -> None:
    files: set[Path] = set()
    for root in ROOTS:
        tex = PAPER / f"{root}.tex"
        files.add(tex)
        included(tex, files)
    files.add(PAPER / "refs.bib")

    newest_source = max(f.stat().st_mtime for f in files)
    for root in ROOTS:
        pdf = PAPER / f"{root}.pdf"
        if not pdf.exists() or pdf.stat().st_mtime < newest_source:
            sys.exit(f"{pdf.name} is older than its sources; build the paper first (just paper)")

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    for root in ROOTS:
        shutil.copy2(PAPER / f"{root}.pdf", DIST / f"{root}.pdf")
    with zipfile.ZipFile(DIST / "mnemos-paper-source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, f.relative_to(PAPER).as_posix())
    print(f"wrote {DIST.relative_to(C.repo_root())}: 2 PDFs and {len(files)} source files")


if __name__ == "__main__":
    main()
