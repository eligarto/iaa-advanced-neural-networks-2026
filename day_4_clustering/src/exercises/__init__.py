"""Worked solutions to the companion-workbook exercises.

One module per exercise, named ``exercise_<chapter>_<n>`` after the workbook
(``article/workbook.tex``): :mod:`exercises.exercise_04_2` is chapter 4,
exercise 2. Every module exposes the same two entry points:

``ANSWER``
    A ``dict`` of the exercise's findings. Theory exercises answer entirely
    through it; the notebooks print one entry at a time so a student can
    reveal the answer without reading the code.
``solve()``
    Recomputes the result and returns it. Code exercises do their real work
    here, reading the data through :mod:`exercises.utils`.

The notebooks under ``notebooks/`` are presentation only: they import these
modules, show a clean cell for the student to try, and then reveal
``ANSWER``/``solve()``. Nothing is computed in a notebook cell that is not
also computed here.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from types import ModuleType

__all__ = [
    "CHAPTERS",
    "EXERCISES",
    "answer_of",
    "chapter_of",
    "iter_exercises",
    "load",
]

#: Workbook chapter number -> (tex file stem, title, ``\label{sec:...}``).
CHAPTERS: dict[int, tuple[str, str, str]] = {
    1: ("01_introduction", "Introduction", "intro"),
    2: ("02_data", "The data", "data"),
    3: ("03_fundamentals", "Clustering fundamentals", "fundamentals"),
    4: ("04_kmeans", "K-means: the default tool", "kmeans"),
    5: ("05_knn", "The kNN primitive", "knn"),
    6: ("06_dbscan", "DBSCAN: following the density", "dbscan"),
    7: ("07_hdbscan", "HDBSCAN*: a hierarchy over every density", "hdbscan"),
    8: ("08_plscan", "PLSCAN: dropping the size floor", "plscan"),
    9: ("09_validation", "Validation: knowing when a score lies", "validation"),
    10: ("10_tsne", "t-SNE: a map of neighbourhoods", "tsne"),
    11: ("11_umap", "UMAP: a graph, then a layout", "umap"),
    12: ("12_evoc", "EVoC: one fit, three stages", "evoc"),
    13: ("13_benchmark", "The benchmark: what we measured", "benchmark"),
    14: ("14_spectral", "Spectra instead of element ratios", "spectral"),
    15: ("15_physics", "What membership is for", "physics"),
    16: ("16_assignment", "The assignment", "assignment"),
}

#: Number of exercises per chapter, as printed in the workbook.
EXERCISES: dict[int, int] = {
    1: 3, 2: 3, 3: 3, 4: 4, 5: 3, 6: 3, 7: 3, 8: 3,
    9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 16: 3,
}

_NAME = re.compile(r"^exercise_(\d{2})_(\d+)$")


def module_name(chapter: int, number: int) -> str:
    """``(4, 2) -> 'exercise_04_2'`` — the workbook's own numbering."""
    return f"exercise_{chapter:02d}_{number}"


def load(chapter: int, number: int) -> ModuleType:
    """Import one exercise module by its workbook coordinates."""
    return importlib.import_module(f"{__name__}.{module_name(chapter, number)}")


def chapter_of(module: ModuleType) -> tuple[int, int]:
    """Recover ``(chapter, number)`` from an exercise module."""
    match = _NAME.match(module.__name__.rsplit(".", 1)[-1])
    if match is None:
        raise ValueError(f"{module.__name__} is not an exercise module")
    return int(match.group(1)), int(match.group(2))


def iter_exercises() -> list[tuple[int, int, str]]:
    """Every exercise module on disk, sorted by (chapter, number)."""
    found: list[tuple[int, int, str]] = []
    for info in pkgutil.iter_modules(__path__):
        match = _NAME.match(info.name)
        if match is not None:
            found.append((int(match.group(1)), int(match.group(2)), info.name))
    return sorted(found)


def answer_of(chapter: int, number: int) -> dict[str, object]:
    """The ``ANSWER`` dict of one exercise, without importing it by hand."""
    return dict(load(chapter, number).ANSWER)
