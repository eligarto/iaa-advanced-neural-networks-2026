"""Citation keys for the exercise answers, resolved against the workbook's bib.

An answer that asserts something the literature established should say where
that comes from, in a form the student can follow to the paper. The rule here
is that a citation is a *key into* ``article/references.bib``, never a
free-typed author-year string, so:

* a key that is not in the bibliography raises immediately, instead of
  printing a plausible-looking reference that resolves to nothing;
* the workbook and the exercises cite the same paper by the same key, so a
  student who follows ``(Campello et al. 2013)`` from an answer lands on the
  entry the chapter cites;
* the rendered text is generated from the bib file, so it cannot drift from
  the bibliography the way a hand-typed citation does.

Write ``cite("Campello:13")`` in an answer and it renders as
``(Campello et al. 2013)``; ``cite("Ester:96", parenthetical=False)`` renders
``Ester et al. (1996)`` for use as a sentence subject.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

__all__ = [
    "BIB_PATH",
    "Reference",
    "bibliography",
    "cite",
    "reference_list",
]

#: The workbook's bibliography. The single source of truth for what may be
#: cited. Exercises share it with the chapters so the keys agree.
BIB_PATH = Path(__file__).resolve().parents[2] / "article" / "references.bib"


class Reference(NamedTuple):
    """One bibliography entry, as much as the bib file records of it."""

    key: str
    authors: tuple[str, ...]
    year: str
    journal: str
    doi: str
    note: str
    howpublished: str

    @property
    def author_text(self) -> str:
        """``Ester et al.``, ``Cover and Hart``, ``Lloyd``: citation style."""
        names = self.authors
        if not names:
            return self.key.split(":")[0]
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return f"{names[0]} et al."

    def render(self, *, parenthetical: bool = True) -> str:
        """``(Ester et al. 1996)`` or ``Ester et al. (1996)``."""
        if parenthetical:
            return f"({self.author_text} {self.year})"
        return f"{self.author_text} ({self.year})"

    @property
    def bare(self) -> str:
        """``Ester et al. 1996``, no brackets at all.

        For use *inside* an existing parenthesis, where both bracketed forms
        would nest: ``(Ester et al. 1996, S 6.3)`` rather than the doubled
        ``((Ester et al. 1996), S 6.3)`` or the odd ``(Ester et al. (1996),
        S 6.3)``.
        """
        return f"{self.author_text} {self.year}"


def _field(entry: str, name: str) -> str:
    """Read one brace-balanced field out of a bib entry."""
    match = re.search(name + r"\s*=\s*\{", entry, re.I)
    if match is None:
        return ""
    depth = 1
    index = match.end()
    chars: list[str] = []
    while index < len(entry) and depth:
        char = entry[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if not depth:
                break
        chars.append(char)
        index += 1
    return " ".join("".join(chars).split())


def _detex(text: str) -> str:
    """Strip the LaTeX a bib field may carry, leaving printable prose.

    Answers are printed to a terminal, not typeset, so ``\\url{...}``,
    brace-protection around corporate authors and accent macros have to come
    out: otherwise a student reads ``{Tutte Institute}`` or a raw ``\\url``.
    """
    text = re.sub(r"\\url\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r'\\[\'"`^~=.]\{?([a-zA-Z])\}?', r"\1", text)
    text = text.replace("\\&", "&").replace("\\_", "_")
    return " ".join(text.replace("{", "").replace("}", "").split())


def _surnames(author_field: str) -> tuple[str, ...]:
    """Surnames in citation order, tolerant of the bib file's two styles.

    Entries are written either ``{Surname, I. N. and ...}`` or as a corporate
    name wrapped in braces (``{Gaia Collaboration}``). ``and others`` is the
    bib spelling of *et al.* and is dropped: :meth:`Reference.author_text`
    re-derives it from the count.

    A brace-protected name is taken whole, which is what the braces are for:
    ``{Bland-Hawthorn, J.}`` is one surname, and splitting it on the comma
    the way a plain ``Surname, Initials`` entry is split would be wrong.
    """
    if not author_field:
        return ()
    names: list[str] = []
    for chunk in re.split(r"\s+and\s+(?![^{]*\})", author_field):
        chunk = chunk.strip()
        if not chunk or chunk == "others":
            continue
        if chunk.startswith("{") and chunk.endswith("}"):
            names.append(_detex(chunk))
            continue
        names.append(_detex(chunk.split(",")[0]))
    return tuple(names)


@lru_cache(maxsize=1)
def bibliography() -> dict[str, Reference]:
    """Every entry of ``article/references.bib``, keyed by citation key."""
    if not BIB_PATH.is_file():
        raise FileNotFoundError(f"bibliography not found: {BIB_PATH}")
    text = BIB_PATH.read_text(encoding="utf-8")
    starts = [m.start() for m in re.finditer(r"^@", text, re.M)] + [len(text)]
    entries: dict[str, Reference] = {}
    for start, stop in zip(starts, starts[1:]):
        entry = text[start:stop]
        head = re.match(r"@\w+\{([^,]+),", entry)
        if head is None:
            continue
        key = head.group(1).strip()
        entries[key] = Reference(
            key=key,
            authors=_surnames(_field(entry, "author")),
            year=_field(entry, "year"),
            journal=_detex(_field(entry, "journal") or _field(entry, "booktitle")),
            doi=_field(entry, "doi"),
            note=_detex(_field(entry, "note")),
            howpublished=_detex(_field(entry, "howpublished")),
        )
    return entries


def cite(*keys: str, parenthetical: bool = True, bare: bool = False) -> str:
    """Render one or more citation keys, failing loudly on an unknown key.

    The failure is the point: an answer that cites ``Campelo:13`` (one ``l``)
    should break a test run, not ship a reference a student cannot find.

    ``bare=True`` drops the brackets entirely, for citing inside a
    parenthesis that is already open.
    """
    bib = bibliography()
    unknown = [key for key in keys if key not in bib]
    if unknown:
        raise KeyError(
            f"not in {BIB_PATH.name}: {', '.join(unknown)}. "
            f"Add the entry to the bibliography before citing it.",
        )
    refs = [bib[key] for key in keys]
    if bare:
        return "; ".join(ref.bare for ref in refs)
    if len(refs) == 1:
        return refs[0].render(parenthetical=parenthetical)
    inner = "; ".join(ref.bare for ref in refs)
    if parenthetical:
        return f"({inner})"
    return inner


def reference_list(*keys: str) -> dict[str, str]:
    """``{'Ester:96': 'Ester et al. (1996), KDD-96, doi:...'}`` for an answer.

    Exercises put this under a ``"references"`` entry so the student sees the
    full pointer (journal and DOI) rather than only an author-year tag.
    """
    bib = bibliography()
    unknown = [key for key in keys if key not in bib]
    if unknown:
        raise KeyError(f"not in {BIB_PATH.name}: {', '.join(unknown)}")
    out: dict[str, str] = {}
    for key in keys:
        ref = bib[key]
        # Title first when the bib records one only as a note (the arXiv and
        # software entries), then where to find it: journal, arXiv id or URL,
        # and the DOI last as the durable handle.
        parts = [ref.render(parenthetical=False)]
        if ref.note:
            parts.append(ref.note)
        if ref.journal:
            parts.append(ref.journal)
        if ref.howpublished and ref.howpublished != ref.journal:
            parts.append(ref.howpublished)
        if ref.doi:
            parts.append(f"doi:{ref.doi}")
        out[key] = ", ".join(parts)
    return out
