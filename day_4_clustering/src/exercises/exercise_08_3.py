"""Chapter 8, exercise 3: persistence measured in stars, for M 67 and NGC 2158.

    The persistence of a cluster is a range of minimum cluster sizes, i.e. a
    number of stars (their Eq. 7). Using the member counts of Table 1, work
    out what a persistence of a given value implies for M 67 and for
    NGC 2158, and argue whether the same cut can serve both.

\\S 8.2 states the workbook's requirement directly: "M 67 (230 members) and
NGC 2158 (6 members) must be judged by the same rule". This exercise does the
arithmetic on the real member counts and then tests it: sweeping
min_cluster_size over the actual 1 002-row member matrix and watching what
each value does to the largest and smallest cluster at the same time. No
single value serves both, and the sweep shows exactly how the failure is
distributed across all 25 clusters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import members

#: The two clusters the exercise names.
LARGEST = "M 67"
SMALLEST = "NGC 2158"

#: Size floors swept by :func:`sweep`. 230 is M 67's own population. The
#: largest floor that could in principle still return one cluster.
FLOORS: tuple[int, ...] = (2, 3, 5, 6, 7, 10, 15, 20, 23, 50, 100, 230)


def population() -> pd.DataFrame:
    """Member counts per cluster, largest first: Table 1's own numbers."""
    counts = members().df["cluster"].value_counts()
    frame = pd.DataFrame({
        "cluster": [str(c) for c in counts.index],
        "n_members": [int(v) for v in counts.to_numpy()],
    })
    frame["share"] = (frame["n_members"] / frame["n_members"].sum()).round(4)
    frame["floor_as_fraction_of_cluster_at_mcs10"] = (
        10.0 / frame["n_members"]
    ).round(3)
    return frame


def floor_arithmetic(floors: tuple[int, ...] = (5, 10, 15, 23, 50)) -> pd.DataFrame:
    """What each candidate size floor means, per cluster, in plain counts."""
    counts = population()
    sizes = counts["n_members"].to_numpy()
    rows = []
    for floor in floors:
        below = counts.loc[sizes < floor, "cluster"].tolist()
        rows.append({
            "floor_stars": floor,
            "n_clusters_below_floor": int((sizes < floor).sum()),
            "n_clusters_kept_in_principle": int((sizes >= floor).sum()),
            f"fraction_of_{LARGEST}": round(floor / 230, 4),
            f"fraction_of_{SMALLEST}": round(floor / 6, 4),
            "excluded": ", ".join(below[:6]) + ("…" if len(below) > 6 else ""),
        })
    return pd.DataFrame(rows)


def sweep(floors: tuple[int, ...] = FLOORS) -> pd.DataFrame:
    """Run HDBSCAN* on the real member matrix at each size floor.

    The cluster-only 1 002-row matrix, not member+field: with the field
    present nothing is recovered at any floor (exercise 7.3), and the point
    here is the size floor in isolation.
    """
    import hdbscan

    from cluster.benchmark import _score_one

    data = members()
    rows = []
    for floor in floors:
        predicted = hdbscan.HDBSCAN(min_cluster_size=floor).fit_predict(data.X)
        scores = _score_one(data.labels, predicted).set_index("cluster")
        row: dict[str, object] = {
            "min_cluster_size": floor,
            "n_clusters": int(len(set(predicted.tolist()) - {-1})),
            "n_noise": int((predicted == -1).sum()),
            "macro_recall": round(float(scores["recall"].mean()), 4),
            "macro_precision": round(float(scores["precision"].mean()), 4),
        }
        for name in (LARGEST, SMALLEST):
            row[f"recall_{name}"] = round(float(scores.loc[name, "recall"]), 4)
            row[f"precision_{name}"] = round(float(scores.loc[name, "precision"]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _argmax_scalar(frame: pd.DataFrame, column: str) -> tuple[int, float]:
    """``(min_cluster_size, column value)`` at the row where ``column`` peaks.

    Returns plain Python scalars: indexing a DataFrame row yields a pandas
    ``Series``, which the type checker refuses to narrow to a number.
    """
    values = frame[column].to_numpy(dtype=float)
    floors = frame["min_cluster_size"].to_numpy(dtype=int)
    index = int(np.argmax(values))
    return int(floors[index]), float(values[index])


def solve() -> dict[str, object]:
    """The arithmetic, then the measurement on the real matrix."""
    counts = population()
    sizes = counts["n_members"].to_numpy()
    table = sweep()

    best_large_floor, best_large_recall = _argmax_scalar(table, f"recall_{LARGEST}")
    best_small_floor, best_small_recall = _argmax_scalar(table, f"recall_{SMALLEST}")
    best_macro_floor, best_macro_recall = _argmax_scalar(table, "macro_recall")
    # Precision at the recall-optimal floor, and the floor that maximises
    # precision: two different rows, and quoting the wrong one would compare
    # a precision from mcs=3 with a recall from mcs=15.
    macro_recall_row = int(np.flatnonzero(
        table["min_cluster_size"].to_numpy(dtype=int) == best_macro_floor,
    )[0])
    precision_at_best_recall = float(
        table["macro_precision"].to_numpy(dtype=float)[macro_recall_row],
    )
    best_precision_floor, best_precision = _argmax_scalar(table, "macro_precision")

    return {
        "n_clusters": int(len(counts)),
        "n_members": int(sizes.sum()),
        f"{LARGEST}_members": int(counts.loc[
            counts["cluster"] == LARGEST, "n_members"].to_numpy()[0]),
        f"{SMALLEST}_members": int(counts.loc[
            counts["cluster"] == SMALLEST, "n_members"].to_numpy()[0]),
        "size_ratio": round(float(sizes.max() / sizes.min()), 1),
        "population": counts,
        "floor_arithmetic": floor_arithmetic(),
        "sweep": table,
        "best_for_largest": {
            "min_cluster_size": best_large_floor,
            "recall": round(best_large_recall, 4),
        },
        "best_for_smallest": {
            "min_cluster_size": best_small_floor,
            "recall": round(best_small_recall, 4),
        },
        "best_for_macro": {
            "min_cluster_size": best_macro_floor,
            "macro_recall": round(best_macro_recall, 4),
            "macro_precision_at_that_floor": round(precision_at_best_recall, 4),
        },
        "best_for_precision": {
            "min_cluster_size": best_precision_floor,
            "macro_precision": round(best_precision, 4),
        },
        "clusters_below_floor_10": int((sizes < 10).sum()),
        "clusters_below_floor_23": int((sizes < 23).sum()),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover (figure)
    """Recall for the largest and smallest cluster against the size floor."""
    import matplotlib.pyplot as plt

    table = sweep()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(table["min_cluster_size"], table[f"recall_{LARGEST}"], "o-",
            color="#4c72b0", label=f"{LARGEST} (230 members)")
    ax.plot(table["min_cluster_size"], table[f"recall_{SMALLEST}"], "s-",
            color="#dd8452", label=f"{SMALLEST} (6 members)")
    ax.plot(table["min_cluster_size"], table["macro_recall"], "^--",
            color="#555555", label="macro over 25 clusters")
    ax.axvline(6, color="crimson", ls=":", lw=1, label=f"{SMALLEST} population")
    ax.set_xscale("log")
    ax.set_xlabel("min_cluster_size (stars)")
    ax.set_ylabel("recall")
    ax.set_title("One size floor, two incompatible requirements")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the arithmetic": (
        "The workbook's 25 clusters span 230 members (M 67) to 6 "
        "(NGC 2158): a ratio of 38.3 to 1, in a single sample of 1 002 "
        "rows. Because PLSCAN's persistence is measured in the same units "
        "as the floor it replaces (a range of minimum cluster sizes, "
        f"{cite('Bot:25', bare=True)} Eq. 7), any value has to be read as a "
        "fraction of "
        "the cluster it is applied to. A floor of 10 stars is 4.35% of "
        "M 67: a rounding error, it could lose 95% of its members and "
        "still qualify, and 167% of NGC 2158, which is to say NGC 2158 "
        "cannot pass it at all, at any density, however chemically perfect "
        "it is. The same number means 'no constraint' and 'automatic "
        "disqualification' depending on which row of Table 1 you are on."
    ),
    "how many clusters each floor deletes": (
        "Counted on the real member list: a floor of 5 excludes none of the "
        "25 (the sample's own min_members=5 selection already guarantees "
        "that). A floor of 10 excludes 3: King 7 (8), Berkeley 17 (7), "
        "NGC 2158 (6). A floor of 15 excludes 7, a floor of 20 excludes 9, "
        "a floor of 23 (the Pleiades' own population) excludes 11, i.e. "
        "44% of the sample. A floor of 50 excludes 21 of 25. These are "
        "hard exclusions before any data is examined: the clusters cannot "
        "be found, not merely found badly."
    ),
    "what the real sweep shows": (
        "HDBSCAN* "
        f"{cite('Campello:13', 'McInnes:17')} on the 1 002-row cluster-only "
        "matrix, min_cluster_size "
        "from 2 to 230 (solve() reproduces it). The two clusters want "
        "different settings and neither gets what it wants. NGC 2158 "
        "reaches recall 0.667 at mcs=2 and again at mcs=10, and collapses "
        "to 0.167 everywhere between (mcs=3 to 7); M 67's recall is flat "
        "and poor throughout, 0.121 to 0.174, best at mcs=2 and at "
        "mcs=20-23 (0.170). Macro recall over all 25 is maximised at mcs=15 "
        "(0.602), but that floor already excludes 7 clusters by "
        "construction, so it is maximising an average over the clusters it "
        "did not delete, and its macro precision is only 0.079. Macro "
        "precision peaks at the other end, at mcs=3 (0.276, i.e. just above "
        "chance), where macro recall is 0.338. At mcs=230, M 67's "
        "own population, every point is noise and every score is zero."
    ),
    "can one cut serve both": (
        "Not as a size floor, no: the arithmetic above is a proof, not an "
        "empirical finding: a constant cannot be both 4% and 167% of "
        "something. What PLSCAN "
        f"{cite('Bot:25')} changes is *who chooses*, not the units. "
        "Persistence is still a number of stars, but instead of being fixed "
        "in advance it is the width of the interval over which a leaf "
        "survives, so M 67 is allowed to be evaluated over a range around "
        "230 and NGC 2158 over a range around 6, and the cut through the "
        "leaf tree is picked by total persistence rather than by a constant. "
        "That is the sense in which S 8.2's claim is true: 'what changes is "
        "not the units but who chooses'."
    ),
    "the caveat this exercise must carry": (
        "PLSCAN itself is not run here, and the numbers above are HDBSCAN's. "
        "The reference implementation of "
        f"{cite('Bot:25', parenthetical=False)} is not a "
        "dependency of this workbook; PLSCAN enters the pipeline only "
        "indirectly, as the model-selection rule inside EVoC "
        f"({cite('EVoC', bare=True)}, S 8.3). So "
        "what is demonstrated is the *problem*: that a single size floor "
        "cannot serve a 38:1 population range. Measured on the real data, "
        "rather than PLSCAN's solution to it. The claim that persistence "
        "solves the problem on this sample is untested here; exercise 12's "
        "EVoC results are the closest the workbook comes to evidence, and "
        "EVoC's advantage there is modest."
    ),
    "the practical rule": (
        "Whenever your targets span more than a factor of a few in size, "
        "quote what your size floor means as a fraction of each target, and "
        "list the targets it excludes outright. Three lines of arithmetic "
        "(floor_arithmetic() does it) will often show that a "
        "hyperparameter chosen on a benchmark has silently removed a third "
        "of your science sample. That check costs nothing and is not "
        "standard practice, which is most of why it is worth doing."
    ),
    "references": reference_list(
        "Bot:25", "Campello:13", "McInnes:17", "EVoC",
    ),
}
