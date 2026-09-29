"""Chapter 1, exercise 2: precision and recall at 300:1.

    The field-to-member ratio here is about 300:1. If a search returns 100
    candidates for a cluster with 20 true members, what are its precision and
    recall? Which would you rather sacrifice, and what does that choice cost
    downstream?

Pencil-and-paper, but the arithmetic is the one \\S 1.1's second difficulty
(contamination) turns on, and it is the definition every score in \\S 9 and
\\S 13 is built from. The point is not the two fractions: it is that at a
300:1 base rate, precision 0.20 is already 60x better than chance, and that
the two errors cost completely different things downstream.
"""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import pandas as pd

from exercises.citations import cite, reference_list

#: The exercise's numbers: a search returning 100 candidates for a cluster
#: that has 20 true members.
N_RETURNED = 100
N_TRUE = 20

#: The workbook's own field:member ratio (1 002 members, 357 056 field).
FIELD_RATIO = 300


def scores(n_returned: int, n_true: int, n_recovered: int) -> dict[str, float]:
    """Precision, recall and F1 for a returned candidate list.

    ``n_recovered`` is the number of true members inside the returned list
    (the true positives). Precision is TP / returned, recall is TP / true.
    """
    precision = n_recovered / n_returned if n_returned else 0.0
    recall = n_recovered / n_true if n_true else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom else 0.0
    return {
        "true_positives": float(n_recovered),
        "false_positives": float(n_returned - n_recovered),
        "false_negatives": float(n_true - n_recovered),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def chance_precision(field_ratio: int = FIELD_RATIO) -> float:
    """Precision of a blind draw when there are ``field_ratio`` field stars
    per member: one member in every ``field_ratio + 1`` stars.
    """
    return 1.0 / (field_ratio + 1)


def tradeoff_table(
    n_true: int = N_TRUE, recalls: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),
) -> pd.DataFrame:
    """Two ways to spend the budget, side by side.

    ``fixed_list`` holds the candidate list at 100 and lets recall fall,
    precision falls with it. ``fixed_recall`` insists on finding every member
    and lets the list grow: the number of wasted follow-ups is what grows.
    """
    rows: list[dict[str, object]] = []
    for recall in recalls:
        found = round(n_true * recall)
        fixed_list = scores(N_RETURNED, n_true, found)
        needed = int(round(n_true / (found / N_RETURNED))) if found else 0
        rows.append({
            "recall": recall,
            "members_found": found,
            "members_missed": n_true - found,
            "precision_at_100_candidates": round(fixed_list["precision"], 4),
            "candidates_for_full_recall": needed,
            "wasted_followups": needed - n_true if needed else 0,
        })
    return pd.DataFrame(rows)


def solve() -> dict[str, object]:
    """The arithmetic, exactly, plus the base rate it has to beat."""
    best_case = scores(N_RETURNED, N_TRUE, N_TRUE)
    chance = chance_precision()

    return {
        "n_returned": N_RETURNED,
        "n_true_members": N_TRUE,
        "assumption": (
            "the returned list contains all 20 true members (the best case "
            "the question allows; it does not state the overlap)"
        ),
        "precision": best_case["precision"],
        "precision_exact": str(Fraction(N_TRUE, N_RETURNED)),
        "recall": best_case["recall"],
        "recall_exact": str(Fraction(N_TRUE, N_TRUE)),
        "f1": round(best_case["f1"], 4),
        "f1_exact": str(Fraction(1, 3)),
        "false_positives": int(best_case["false_positives"]),
        "false_negatives": int(best_case["false_negatives"]),
        "chance_precision": round(chance, 5),
        "enrichment_over_chance": round(best_case["precision"] / chance, 1),
        "tradeoff": tradeoff_table(),
        "worst_case_precision": 0.0,
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover (figure)
    """Precision against recall for a fixed 100-candidate list."""
    import matplotlib.pyplot as plt
    import numpy as np

    result = result if result is not None else solve()
    recall = np.linspace(0.05, 1.0, 40)
    precision = recall * N_TRUE / N_RETURNED

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(recall, precision, color="#4c72b0", lw=2,
            label=f"list fixed at {N_RETURNED} candidates")
    ax.axhline(
        cast(float, result["chance_precision"]), color="crimson", ls="--", lw=1,
        label=f"chance precision 1/{FIELD_RATIO + 1} = "
              f"{result['chance_precision']}",
    )
    ax.scatter([1.0], [0.2], s=60, color="k", zorder=5,
               label="the exercise: P=0.20, R=1.00")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_yscale("log")
    ax.set_title("At 300:1, precision 0.20 is 60x chance")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the arithmetic": (
        "The question under-specifies the overlap, so state the assumption: "
        "if all 20 true members are inside the 100 returned candidates, then "
        "recall = 20/20 = 1.00 and precision = 20/100 = 1/5 = 0.20, giving "
        "F1 = 2PR/(P+R) = 1/3 = 0.333. There are 80 false positives and 0 "
        "false negatives. If instead only, say, 10 of the 20 are in the "
        "list, recall = 0.50 and precision = 0.10: precision cannot exceed "
        "recall x 20/100 here, so the two move together as long as the list "
        "length is fixed. The worst case the question permits is 0 overlap: "
        "precision 0.00, recall 0.00, and a search that returned 100 stars "
        "with nothing in them."
    ),
    "why 0.20 is not a bad number": (
        "The base rate is the context the two fractions need. At 300 field "
        "stars per member, a blind draw of 100 stars contains 100/301 = 0.33 "
        "members, i.e. chance precision is 1/301 = 0.0033. Precision 0.20 is "
        "therefore 60x the chance rate. In this workbook's own sample the "
        "ratio is 357 056 field to 1 002 members (356:1) and the measured "
        "field-retrieval precisions in section 13 are 0.240 for abundances "
        "with t-SNE and 0.631 for the PCA-64 latent, so 0.20 is squarely in "
        "the range a real method reaches. Quoting precision without the base "
        "rate is how a genuinely informative search gets called useless."
    ),
    "which one to sacrifice": (
        "Sacrifice precision, and say so explicitly, because the two errors "
        "are not symmetric in this problem. A false negative is a lost "
        "member: chemistry is the *only* surviving record of the birth "
        "cluster "
        f"{cite('Freeman:02', 'BlandHawthorn:16')}, so a star you did not "
        "flag will not be "
        "looked at again, and the information is gone. A false positive is "
        "an extra star on a follow-up list, and follow-up: a Gaia proper "
        f"motion {cite('Gaia:23')}, a radial velocity, a second spectrum. "
        "is cheap and "
        f"available for every APOGEE target {cite('Majewski:17')}. The "
        "asymmetry only holds while "
        "you have a referee, which here you do: section 2.4's kinematic "
        "membership exists precisely to clean a candidate list afterwards."
    ),
    "what that choice costs downstream": (
        "Follow-up budget, and it scales badly. Holding recall at 1.00, the "
        "candidate list needed to contain all 20 members is 20/precision: "
        "100 stars at P=0.20, 200 at P=0.10, 400 at P=0.05, so the wasted "
        "follow-ups go 80, 180, 380. Each halving of precision doubles the "
        "spectroscopy bill. It also costs *statistics*: a group that is 80% "
        "contaminated has an abundance dispersion and a mean metallicity set "
        "mostly by the field, so any physical parameter you fit to the group "
        "(age, distance, [Fe/H] spread: section 15's questions) is biased "
        "toward the field, not merely noisy. Low precision is survivable "
        "when a referee follows; it is fatal when the group itself is the "
        "measurement."
    ),
    "the reporting rule": (
        "Never quote one of the pair. A recall of 1.00 is trivially "
        "achievable by returning the whole catalogue, and section 13 shows "
        "exactly that failure in the wild: UMAP on abundances scores recall "
        "1.000 with precision 0.001: a single degenerate blob containing "
        "every star, reported as perfect recall. Section 9 adds the "
        "companion rule: quote the chance level too, because at 300:1 a "
        "precision that sounds terrible can be 60x better than random, and "
        "a homogeneity that sounds respectable can be at the floor."
    ),
    "references": reference_list(
        "Freeman:02", "BlandHawthorn:16", "Gaia:23", "Majewski:17",
    ),
}
