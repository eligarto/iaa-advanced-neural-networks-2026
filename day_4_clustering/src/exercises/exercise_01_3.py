"""Chapter 1, exercise 3 — Hogg (2016) vs Casamiquela (2021).

    Read Hogg et al. (2016) and Casamiquela et al. (2021) side by side: one
    is optimistic about strong chemical tagging and one is not. Write two
    sentences each on which *measurement* they disagree about.

The exercise is a reading exercise, and the answer below is deliberately
restricted to what this workbook itself can support: \\S 1.3 for Hogg and
\\S 13.7 (``sec:literature``) for Casamiquela, both of which the repository
quotes with a source file behind them. The lesson is \\S 1.3's closing line
--- the scope of any result in this field is *these abundances, these
clusters* --- so the disagreement resolves into three measured quantities
rather than a verdict.

This module deliberately defines no ``solve()``: nothing here is computed
from the catalogue, and inventing a computation would misrepresent a
literature question. The numbers quoted are the workbook's own, and
:func:`sources` says where each one lives.
"""

from __future__ import annotations

import pandas as pd

from exercises.citations import cite, reference_list

#: Per-element abundance precision, in dex, as the workbook states it.
PRECISION_DEX: dict[str, float] = {
    "Hogg 2016 (Cannon, APOGEE)": 0.04,
    "Casamiquela 2021 (differential, red clump)": 0.03,
    "this workbook (DR19 Astra ASPCAP)": 0.05,
}


def comparison() -> pd.DataFrame:
    """The two papers side by side on the axes the workbook records.

    Every cell is sourced from ``article/chapters/01_introduction.tex`` or
    ``article/chapters/13_benchmark.tex`` (Sect. "Where this sits in the
    literature"); nothing is added from memory.
    """
    rows = [
        {
            "axis": "sample size",
            "Hogg 2016": "~10^5 APOGEE stars",
            "Casamiquela 2021": "175 red-clump stars",
        },
        {
            "axis": "abundance dimension",
            "Hogg 2016": "15 Cannon abundances",
            "Casamiquela 2021": "16 elements, differential",
        },
        {
            "axis": "precision (dex/element)",
            "Hogg 2016": "~0.04",
            "Casamiquela 2021": "~0.03",
        },
        {
            "axis": "field contamination",
            "Hogg 2016": "field present (no positional info used)",
            "Casamiquela 2021": "no field stars at all",
        },
        {
            "axis": "method",
            "Hogg 2016": "K-means on abundances",
            "Casamiquela 2021": "HDBSCAN on abundances, tuned grid",
        },
        {
            "axis": "what was verified",
            "Hogg 2016": "abundance overdensities are phase-space clusters",
            "Casamiquela 2021": "named birth clusters recovered as groups",
        },
        {
            "axis": "headline result",
            "Hogg 2016": "positive: overdensities are real structures",
            "Casamiquela 2021": "h=0.49 c=0.63 V=0.55, RF40=0.29 (9 of 31)",
        },
        {
            "axis": "their own caveat",
            "Hogg 2016": "(workbook: defines what overdensities can be at 0.04 dex)",
            "Casamiquela 2021": ">70% of groups are statistical, mixing clusters",
        },
    ]
    return pd.DataFrame(rows)


def sources() -> dict[str, str]:
    """Where each claim in :data:`ANSWER` comes from, so it can be checked."""
    return {
        "Hogg 2016 description": "article/chapters/01_introduction.tex, §1.3",
        "Hogg 2016 in the K-means chapter": "article/chapters/04_kmeans.tex, §4.5",
        "Casamiquela 2021 numbers": (
            "article/chapters/13_benchmark.tex, §13.7 sec:literature, "
            "Table tab:literature"
        ),
        "recovery fraction definition": "article/chapters/09_validation.tex, eq:rf",
        "bibliography": (
            "article/references.bib — Hogg:16 = ApJ 833, 262 "
            "(doi 10.3847/1538-4357/833/2/262); Casamiquela:21 = A&A 654, "
            "A151 (doi 10.1051/0004-6361/202141779)"
        ),
        "our own replication": "results/casamiquela_comparison.csv",
    }


ANSWER: dict[str, object] = {
    "Hogg 2016, in two sentences": (
        f"{cite('Hogg:16', parenthetical=False)} clustered ~10^5 APOGEE "
        "stars in 15 Cannon "
        "abundances with K-means, using no positional information at all, "
        "and showed that the overdensities K-means finds in abundance space "
        "really are phase-space structures — a positive result about "
        "abundance space having tag-able structure in it. Their measurement "
        "is therefore *existence*: at ~0.04 dex per element, chemistry alone "
        "carries enough information that a chemically defined group is also "
        "a kinematically coherent one."
    ),
    "Casamiquela 2021, in two sentences": (
        f"{cite('Casamiquela:21', parenthetical=False)} measured 16 "
        "elements *differentially* in 175 red-clump members of 31 thin-disc "
        "open clusters — internal coherence ~0.03 dex, no field stars, which "
        "they call their best-case scenario — then clustered the abundances "
        "directly with HDBSCAN and tuned its parameters to maximise the "
        "number of clusters recovered. Their measurement is *identification*: "
        "h = 0.49, c = 0.63, V = 0.55, with a recovery fraction of 0.29 at "
        "the 40% threshold (9 of 31 clusters) and 0.03 at 70% (one cluster, "
        "NGC 2682), and their own summary is that more than 70% of the "
        "groups found are statistical, mixing stars from several real "
        "clusters."
    ),
    "the measurement they disagree about": (
        "Not precision, and not method — it is the *question*. Hogg asks "
        "whether abundance-space overdensities correspond to real phase-space "
        "structure (an existence claim about the signal, verified against "
        "kinematics), and answers yes. Casamiquela asks whether a specific "
        "named birth cluster can be pulled back out as a group (an "
        "identification claim about individual objects, scored with recovery "
        "fraction), and answers mostly no. Both can be true at once: "
        "chemistry can carry real structure while being unable to resolve "
        "which of 31 similar open clusters a star came from. That is exactly "
        "the weak-tagging / strong-tagging distinction of §1.3's margin note, "
        "which goes back to the programme as originally set out "
        f"{cite('Freeman:02', 'BlandHawthorn:16')}."
    ),
    "the second disagreement: what counts as success": (
        "Hogg's success criterion is agreement with an *independent* signal "
        "(phase space), so a group that is chemically coherent and "
        "kinematically coherent passes even if it is not one named cluster. "
        "Casamiquela's criterion is a two-sided per-cluster threshold: the "
        "matched group must hold >=40% of the cluster *and* be >=40% that "
        "cluster (§9's recovery fraction, eq:rf), so a group that merges two "
        "real clusters fails even though it is chemically real. Swap the "
        "criteria and the verdicts would move toward each other — which is "
        "why the workbook reports the metric pair and the recovery fraction "
        "side by side rather than choosing one."
    ),
    "the third disagreement: the sample, not the chemistry": (
        "Hogg's 10^5 stars span the disc and halo, so his groups can differ "
        "in metallicity by dex. Casamiquela's 31 clusters are all thin-disc, "
        "all red clump, all at similar metallicity — the population §1.1 "
        "identifies as the hardest case, where efficient ISM mixing "
        f"({cite('Kreckel:20', bare=True)}: 0.02 - 0.03 dex of scatter in "
        "nearby spiral discs, correlated below 600 pc) "
        "limits how different two clusters can be. Her better precision "
        "(0.03 vs 0.04 dex) is spent on a harder sample, so the two results "
        "are not on the same difficulty scale at all."
    ),
    "what this workbook adds to the pair": (
        "A like-for-like check: §13.7 runs Casamiquela's own clustering step "
        "on our 982 stars and gets RF40 = 0.08 against her 0.29 (their 4x7 "
        "parameter grid never exceeds 0.11 on our stars), while our own arms "
        "reach RF40 = 0.32 for t-SNE and 0.33 for UMAP on the same stars. "
        "Two differences remain that cannot be separated — our abundances "
        "are survey-quality (~0.05 dex against 0.03) and our clusters are "
        "not a single evolutionary stage — which is §1.3's closing point "
        "made quantitative: the scope of a negative result is *these "
        "abundances, these clusters*, never 'abundances cannot tag'."
    ),
    "how to write the answer": (
        "Name the measurement, not the mood. 'Hogg is optimistic and "
        "Casamiquela is pessimistic' is not an answer; 'Hogg measured "
        "whether chemical overdensities are phase-space structures and "
        "Casamiquela measured what fraction of named clusters come back at a "
        "40% two-sided threshold' is. Then state the three axes that differ "
        "— question, success criterion, sample difficulty — and note that "
        "neither paper contradicts the other on any quantity both of them "
        "actually measured."
    ),
    "what you should not claim": (
        "Do not attribute quotes or numbers to either paper beyond what is "
        "reproduced here. This module takes the Hogg description from §1.3 "
        "and §4.5 and the Casamiquela numbers from §13.7 and Table "
        "tab:literature; call sources() for the exact file and section "
        "behind each one. If you need anything else — their per-cluster "
        "table, their element list, their HDBSCAN grid — read the papers "
        "themselves (DOIs are in article/references.bib); the workbook is "
        "not a substitute for them."
    ),
    "references": reference_list(
        "Hogg:16", "Casamiquela:21", "Freeman:02", "BlandHawthorn:16",
        "Kreckel:20",
    ),
}
