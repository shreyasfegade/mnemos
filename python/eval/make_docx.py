"""Word (.docx) versions of the paper and the supplement, from the same LaTeX sources.

    cd python && python -m eval.make_docx          (after python -m eval.build_paper)

Pandoc reads LaTeX but not IEEEtran, cleveref or BibTeX output, so the sources are flattened and
normalised first. Everything that carries a number is resolved from the *built* paper rather than
re-derived, so the Word file and the PDF cannot disagree:

* ``\\cref``/``\\Cref``/``\\ref`` are resolved from ``main.aux`` and ``supplement.aux`` (cleveref's
  ``@cref`` entries give the type and number LaTeX actually assigned);
* ``\\cite`` becomes the ``[n]`` IEEEtran assigned (``\\bibcite`` in ``main.aux``), and the
  reference
  list is taken from ``main.bbl``;
* result macros come from ``tables/numbers.tex`` exactly as in the PDF;
* figures use the PNG each figure script writes beside its PDF.

It refuses to run on a build older than its sources, for the same reason ``make_dist`` does.
"""

from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

import pypandoc

from eval.lab import corpus as C

PAPER = C.repo_root() / "paper"
DIST = PAPER / "dist"

PREAMBLE = r"""
\documentclass{article}
\usepackage{amsmath,amssymb,amsthm}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
\newtheorem{proposition}{Proposition}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newcommand{\advhist}{\mathrm{Adv}_{\mathrm{hist}}}
\newcommand{\advpost}{\mathrm{Adv}_{\mathrm{post}}}
\newcommand{\AdvH}{\mathsf{Adv}^{\mathrm{H}}}
\newcommand{\AdvR}{\mathsf{Adv}^{\mathrm{R}}}
\newcommand{\negl}{\mathsf{negl}}
\newcommand{\pt}[1]{\par\textbf{#1}\par}
\newcommand{\done}{\textbf{[done]}}
\newcommand{\pdone}{\textbf{[partly done]}}
\newcommand{\planned}{\textbf{[in progress]}}
"""

TYPE_NAMES = {
    "section": ("Section", "Sections"),
    "subsection": ("Section", "Sections"),
    "subsubsection": ("Section", "Sections"),
    "appendix": ("Appendix", "Appendices"),
    "table": ("Table", "Tables"),
    "figure": ("Figure", "Figures"),
    "theorem": ("Theorem", "Theorems"),
    "lemma": ("Lemma", "Lemmas"),
    "proposition": ("Proposition", "Propositions"),
    "corollary": ("Corollary", "Corollaries"),
    "definition": ("Definition", "Definitions"),
    "equation": ("Eq.", "Eqs."),
    "enumi": ("item", "items"),
}


def labels_from(aux: Path) -> dict[str, tuple[str, str]]:
    """label -> (cleveref type, number as printed), from a built .aux file."""
    out: dict[str, tuple[str, str]] = {}
    if not aux.exists():
        return out
    text = aux.read_text(encoding="latin-1")
    for m in re.finditer(
        r"\\newlabel\{([^}]+)@cref\}\{\{\[([^\]]*)\]\[[^\]]*\]\[[^\]]*\]([^}]*)\}", text
    ):
        out[m.group(1)] = (m.group(2), m.group(3))
    return out


def citations_from(aux: Path) -> dict[str, str]:
    text = aux.read_text(encoding="latin-1") if aux.exists() else ""
    return dict(re.findall(r"\\bibcite\{([^}]+)\}\{([^}]+)\}", text))


def inline_inputs(text: str, seen: int = 0) -> str:
    if seen > 10:
        raise SystemExit("\\input nesting too deep")

    def rep(m: re.Match[str]) -> str:
        name = m.group(1)
        if name == "tables/numbers":
            return ""  # macros go in the preamble
        path = PAPER / (name if name.endswith(".tex") else name + ".tex")
        return inline_inputs(path.read_text(encoding="utf-8"), seen + 1)

    return re.sub(r"\\input\{([^}]+)\}", rep, text)


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.split("\n"))


def resolve_refs(text: str, labels: dict[str, tuple[str, str]]) -> str:
    missing: set[str] = set()

    def name_of(key: str, capital: bool) -> tuple[str, str, str]:
        if key not in labels:
            missing.add(key)
            return ("", "??", "")
        typ, num = labels[key]
        one, many = TYPE_NAMES.get(typ, (typ.capitalize(), typ.capitalize() + "s"))
        return (one, num, many)

    def cref(m: re.Match[str]) -> str:
        capital = m.group(1) == "C"
        keys = [k.strip() for k in m.group(2).split(",") if k.strip()]
        parts = [name_of(k, capital) for k in keys]
        if len({p[0] for p in parts}) == 1 and len(parts) > 1:
            nums = [p[1] for p in parts]
            joined = ", ".join(nums[:-1]) + " and " + nums[-1]
            return f"{parts[0][2]} {joined}"
        words = [f"{p[0]} {p[1]}".strip() for p in parts]
        return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]

    text = re.sub(r"\\([cC])ref\{([^}]+)\}", cref, text)
    text = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), ("", "??"))[1], text)
    if missing:
        raise SystemExit(f"unresolved references: {sorted(missing)}")
    return text


def resolve_cites(text: str, cites: dict[str, str]) -> str:
    def rep(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        bad = [k for k in keys if k not in cites]
        if bad:
            raise SystemExit(f"uncited keys: {bad}")
        return ", ".join(f"[{cites[k]}]" for k in keys)

    return re.sub(r"\\cite\{([^}]+)\}", rep, text)


def references_from_bbl(bbl: Path, cites: dict[str, str]) -> str:
    text = bbl.read_text(encoding="utf-8")
    body = text[text.index("\\bibitem") :]
    body = body[: body.index("\\end{thebibliography}")]
    items = re.split(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", body)[1:]
    out = ["\\section*{References}"]
    for key, entry in zip(items[0::2], items[1::2], strict=True):
        e = entry
        e = re.sub(r"\\BIBentry(?:STD|ALT)interwordspacing", "", e)
        e = e.replace("\\newblock", "")
        e = re.sub(r"\\bibinfo\{[^}]*\}", "", e)  # \bibinfo{field}{value} -> {value}
        e = re.sub(r"\\url\{([^}]*)\}", r"\\texttt{\1}", e)
        e = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", e)
        e = " ".join(e.split())
        out.append(f"\\noindent [{cites.get(key, '?')}] {e}\n")
    return "\n\n".join(out)


def normalise(body: str) -> str:
    body = re.sub(r"\\IEEEPARstart\{([^}]*)\}\{([^}]*)\}", r"\1\2", body)
    body = re.sub(r"\\texorpdfstring\{([^}]*)\}\{[^}]*\}", r"\1", body)
    body = re.sub(r"\\nolinkurl\{([^}]*)\}", r"\\texttt{\1}", body)
    body = re.sub(r"\\bstctlcite\{[^}]*\}", "", body)
    body = body.replace("\\IEEEpeerreviewmaketitle", "").replace(
        "\\IEEEdisplaynontitleabstractindextext", ""
    )
    body = body.replace("\\maketitle", "")
    # Footnote marks: texmath has no \S, and these are text symbols anyway.
    for sym, ch in (("\\S", "§"), ("\\ddagger", "‡"), ("\\dagger", "†"), ("\\ast", "*")):
        body = body.replace(f"$^{sym}$", f"\\textsuperscript{{{ch}}}")
    # Run-in paragraph heads, as in the PDF, rather than Word headings.
    body = re.sub(r"\\paragraph\{([^}]*)\}", r"\\textbf{\1.} ", body)
    # Double-column floats are ordinary floats in a single-column Word file.
    body = body.replace("\\begin{table*}", "\\begin{table}").replace(
        "\\end{table*}", "\\end{table}"
    )
    body = body.replace("\\begin{figure*}", "\\begin{figure}").replace(
        "\\end{figure*}", "\\end{figure}"
    )
    body = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?\{[^}]*?)\.pdf\}", r"\1.png}", body)
    body = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", body)
    # Pandoc's math writer falls back to raw TeX on fixed-size delimiters and \tfrac in some
    # contexts; the stretchy forms render identically in Word's equation editor.
    for a, b in (
        (r"\bigl|", r"\left|"),
        (r"\bigr|", r"\right|"),
        (r"\Bigl(", r"\left("),
        (r"\Bigr)", r"\right)"),
        (r"\bigl(", r"\left("),
        (r"\bigr)", r"\right)"),
        (r"\tfrac", r"\frac"),
    ):
        body = body.replace(a, b)
    body = re.sub(
        r"\\begin\{IEEEbiographynophoto\}\{([^}]*)\}",
        r"\\section*{Author}\n\\textbf{\1} ",
        body,
    )
    body = body.replace("\\end{IEEEbiographynophoto}", "")
    for cmd in ("\\vfill", "\\clearpage", "\\newpage"):
        body = body.replace(cmd, "")
    return body


def polish(path: Path) -> None:
    """Journal typography: a serif body, and tables ruled like booktabs rather than unruled."""
    import docx
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    d = docx.Document(str(path))
    for name, size in (("Normal", 10), ("Body Text", 10), ("First Paragraph", 10), ("Compact", 9)):
        try:
            st = d.styles[name]
        except KeyError:
            continue
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    from docx.shared import RGBColor

    for name, size in (("Title", 20), ("Heading 1", 13), ("Heading 2", 11), ("Heading 3", 10)):
        with contextlib.suppress(KeyError):
            st = d.styles[name]
            st.font.name = "Times New Roman"
            st.font.size = Pt(size)
            st.font.bold = True
            st.font.color.rgb = RGBColor(0, 0, 0)
            rpr = st.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rpr.append(fonts)
            for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                fonts.set(qn(attr), "Times New Roman")
            for theme in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                fonts.attrib.pop(qn(theme), None)
    for name in ("Table Caption", "Image Caption"):
        with contextlib.suppress(KeyError):
            d.styles[name].font.size = Pt(9)

    def rule(tag: str, sz: int) -> OxmlElement:
        e = OxmlElement(f"w:{tag}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), "000000")
        return e

    for table in d.tables:
        tbl_pr = table._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        borders.append(rule("top", 12))
        borders.append(rule("bottom", 12))
        tbl_pr.append(borders)
        if table.rows:
            for cell in table.rows[0].cells:
                tc_pr = cell._tc.get_or_add_tcPr()
                cb = OxmlElement("w:tcBorders")
                cb.append(rule("bottom", 6))
                tc_pr.append(cb)
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(8.5)
    d.save(str(path))


def build(root: str, title: str, author: str, abstract: str | None, keywords: str | None) -> Path:
    tex = PAPER / f"{root}.tex"
    pdf = PAPER / f"{root}.pdf"
    src = strip_comments(tex.read_text(encoding="utf-8"))
    full = inline_inputs(src)
    start = full.index("\\begin{document}") + len("\\begin{document}")
    end = full.index("\\end{document}")
    body = full[start:end]
    # Title block: everything up to where the text proper starts is IEEEtran front matter.
    for marker in ("\\IEEEpeerreviewmaketitle", "\\maketitle"):
        if marker in body:
            body = body[body.index(marker) + len(marker) :]
            break

    labels = labels_from(PAPER / f"{root}.aux")
    if root != "main":
        labels = {**labels_from(PAPER / "main.aux"), **labels}
    cites = citations_from(PAPER / "main.aux")

    # Number captions as the PDF does; pandoc's Word writer does not number floats.
    def number_float(m: re.Match[str]) -> str:
        block = m.group(0)
        lab = re.search(r"\\label\{([^}]+)\}", block)
        if not lab or lab.group(1) not in labels:
            return block
        word = "Table" if m.group(1) == "table" else "Figure"
        num = labels[lab.group(1)][1]
        return block.replace("\\caption{", f"\\caption{{\\textbf{{{word} {num}.}} ", 1)

    body = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", number_float, body, flags=re.S)
    body = resolve_refs(body, labels)
    body = resolve_cites(body, cites)
    if "\\bibliography{refs}" in body:
        body = body.replace("\\bibliographystyle{IEEEtran}", "")
        body = body.replace("\\bibliography{refs}", references_from_bbl(PAPER / "main.bbl", cites))
    body = normalise(body)

    front = ""
    if abstract:
        front += "\\begin{abstract}\n" + normalise(
            resolve_cites(resolve_refs(abstract, labels), cites)
        )
        front += "\n\\end{abstract}\n"
    if keywords:
        front += f"\\noindent\\textbf{{Index Terms}}---{keywords}\n\n"

    numbers = (PAPER / "tables" / "numbers.tex").read_text(encoding="utf-8")
    doc = PREAMBLE + numbers + "\n\\begin{document}\n" + front + body + "\n\\end{document}\n"
    if "[not run]" in numbers:
        sys.exit("numbers.tex has unmeasured values; run the experiments first")

    newest = max(
        p.stat().st_mtime for p in [tex, *PAPER.glob("sections/*.tex"), *PAPER.glob("tables/*.tex")]
    )
    if not pdf.exists() or pdf.stat().st_mtime < newest:
        sys.exit(f"{pdf.name} is older than its sources; build the paper first")

    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / f"{root}.docx"
    (DIST / f"{root}.flat.tex").write_text(doc, encoding="utf-8")
    pypandoc.convert_text(
        doc,
        "docx",
        format="latex",
        outputfile=str(out),
        extra_args=[
            f"--resource-path={PAPER}",
            "--number-sections",
            f"--metadata=title:{title}",
            f"--metadata=author:{author}",
        ],
    )
    polish(out)
    if "--keep-flat" not in sys.argv:
        (DIST / f"{root}.flat.tex").unlink()
    return out


def main() -> None:
    main_src = (PAPER / "main.tex").read_text(encoding="utf-8")
    # The title carries a LaTeX line break for the PDF; Word metadata wants one line.
    title = re.sub(r"\s*\\\\\s*", " ", re.search(r"\\title\{([^}]*)\}", main_src).group(1))
    keywords = " ".join(
        re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", main_src, re.S)
        .group(1)
        .split()
    )
    abstract = strip_comments((PAPER / "sections" / "00-abstract.tex").read_text(encoding="utf-8"))
    author = "Shreyas Fegade"
    print("wrote", build("main", title, author, abstract, keywords).relative_to(C.repo_root()))
    print(
        "wrote",
        build(
            "supplement", f"Supplementary Material for \u201c{title}\u201d", author, None, None
        ).relative_to(C.repo_root()),
    )
    print(
        "wrote",
        build(
            "response", f"Response to Reviewers: \u201c{title}\u201d", author, None, None
        ).relative_to(C.repo_root()),
    )


if __name__ == "__main__":
    main()
