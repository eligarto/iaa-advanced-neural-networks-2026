#!/usr/bin/env python3
"""Static checks on the workbook sources, before LaTeX runs.

The workbook is 17 chapter files, 60-odd figures, 50 bibliography entries and
several hundred cross-references. LaTeX reports dangling references only after
a full build, and it reports some of them (an unclosed environment, a stray
math delimiter) as hundreds of cascading errors rather than one. This script
catches the whole class of them in a second:

  * every \\citep/\\citet key has an entry in references.bib, and reports the
    entries that are never referenced;
  * every \\ref/\\eqref has a matching \\label;
  * every \\includegraphics target (and every \\fig{...}) exists in figures/;
  * environments are balanced per file (an unclosed `issues` box once broke a
    build 40 files later);
  * math delimiters are balanced per file;
  * no HTML-style closing tags (`</issues>`) survive from editing.

Exit code 0 means the sources are internally consistent. It says nothing about
whether the numbers in them are right --- that is what the repository's tests
and `cluster head-to-head` are for.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parent
CHAPTERS = ARTICLE / "chapters"
FIGURES = ARTICLE / "figures"


def read_sources() -> dict[str, str]:
    files = {str(p): p.read_text() for p in sorted(CHAPTERS.glob("*.tex"))}
    files[str(ARTICLE / "workbook.tex")] = (ARTICLE / "workbook.tex").read_text()
    return files


def check(verbose: bool = False) -> int:
    files = read_sources()
    body = "\n".join(files.values())
    problems: list[str] = []

    # --- citations ---------------------------------------------------------
    cited: set[str] = set()
    for grp in re.findall(r"\\cite[a-z]*\{([^}]*)\}", body):
        cited.update(k.strip() for k in grp.split(",") if k.strip())
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", (ARTICLE / "references.bib").read_text()))
    for miss in sorted(cited - bib_keys):
        problems.append(f"citation key not in references.bib: {miss}")
    unused = sorted(bib_keys - cited)
    if verbose:
        print(f"citations: {len(cited)} used, {len(bib_keys)} in the bib")
        if unused:
            print(f"  unused entries (kept out of the printed list): {', '.join(unused)}")

    # --- cross-references --------------------------------------------------
    labels = set(re.findall(r"\\label\{([^}]*)\}", body))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", body))
    for bad in sorted(refs - labels):
        problems.append(f"dangling reference: \\ref{{{bad}}} has no \\label")

    # --- figures -----------------------------------------------------------
    figs = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", body))
    figs |= set(re.findall(r"\\fig(?:\[[^\]]*\])?\{([^}]+)\}", body))
    # macro *definitions* use #1/#2 placeholders; they are not file names
    figs = {f for f in figs if "#" not in f and "<" not in f}
    for bad in sorted(figs):
        if not (FIGURES / bad).exists():
            problems.append(f"figure not vendored: figures/{bad}")

    # --- per-file structural checks ---------------------------------------
    for path, text in files.items():
        name = Path(path).name
        begins = Counter(re.findall(r"\\begin\{(\w+\*?)\}", text))
        ends = Counter(re.findall(r"\\end\{(\w+\*?)\}", text))
        for env in set(begins) | set(ends):
            if begins[env] != ends[env]:
                problems.append(
                    f"{name}: unbalanced environment {env!r} "
                    f"({begins[env]} begin, {ends[env]} end)"
                )
        if "</" in text:
            for tag in re.findall(r"</(\w+)>", text):
                problems.append(f"{name}: HTML-style closing tag </{tag}>")
        if text.replace("\\$", "").replace("\\%", "").count("$") % 2:
            problems.append(f"{name}: odd number of $ delimiters (math mode left open)")

    if problems:
        print(f"FAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(
        f"OK — {len(files)} source files, {len(cited)} citations, "
        f"{len(refs)} cross-references, {len(figs)} figures, all consistent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(check(verbose="--verbose" in sys.argv or "-v" in sys.argv))
