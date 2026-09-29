"""Chapter 6, exercise 2 — the k-distance plot and the missing knee.

    Produce the k-distance plot for the DR19 member-plus-field matrix at k=5
    and locate the knee. Repeat for k=2 and k=20. How does the knee move, and
    what does that say about whether a single epsilon exists for these data?

\\S 6.3 calls the k-distance knee "the standard data-driven recipe" and then
says it is "exactly the step that fails in this project". This exercise is
the evidence. The knee is located numerically by two independent rules, the
epsilon it produces is fed back into DBSCAN, and the result is the "single
threshold problem" of \\S 6.3 as a measured fact rather than an assertion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import member_field

#: The three neighbour counts the exercise asks for.
K_VALUES: tuple[int, ...] = (2, 5, 20)


def k_distance_curve(X: np.ndarray, k: int) -> np.ndarray:
    """Sorted distance from every point to its ``k``-th nearest neighbour."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    return np.sort(np.asarray(distances)[:, k])


def kneedle(curve: np.ndarray) -> tuple[int, float]:
    """Knee by maximum deviation from the chord (the Kneedle rule).

    Both axes are min-max normalised, a straight line is drawn between the
    first and last point of the sorted curve, and the knee is the rank whose
    vertical offset from that line is largest. Returns ``(rank, epsilon)``.
    """
    n = curve.size
    x = np.arange(n, dtype=float) / (n - 1)
    y = (curve - curve.min()) / (curve.max() - curve.min())
    offset = y - (y[0] + x * (y[-1] - y[0]))
    rank = int(np.argmax(np.abs(offset)))
    return rank, float(curve[rank])


def max_curvature(curve: np.ndarray) -> tuple[int, float]:
    """Knee by maximum discrete curvature — the textbook alternative.

    Included because it *disagrees* with Kneedle on these data, which is
    itself part of the answer: a knee that two reasonable rules locate in
    different places is not a knee.
    """
    n = curve.size
    x = np.arange(n, dtype=float) / (n - 1)
    y = (curve - curve.min()) / (curve.max() - curve.min())
    first = np.gradient(y, x)
    second = np.gradient(first, x)
    curvature = np.abs(second) / (1.0 + first ** 2) ** 1.5
    rank = int(np.argmax(curvature))
    return rank, float(curve[rank])


def knee_table(k_values: tuple[int, ...] = K_VALUES) -> pd.DataFrame:
    """Both knee rules at every k, with the member/field split for context."""
    from sklearn.neighbors import NearestNeighbors

    data = member_field()
    rows = []
    for k in k_values:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(data.X)
        distances, _ = nn.kneighbors(data.X)
        kth = np.asarray(distances)[:, k]
        curve = np.sort(kth)

        rank_kneedle, eps_kneedle = kneedle(curve)
        rank_curv, eps_curv = max_curvature(curve)
        rows.append({
            "k": k,
            "median_kdist": round(float(np.median(curve)), 4),
            "p90_kdist": round(float(np.percentile(curve, 90)), 4),
            "kneedle_rank_fraction": round(rank_kneedle / curve.size, 4),
            "eps_kneedle": round(eps_kneedle, 4),
            "curvature_rank_fraction": round(rank_curv / curve.size, 4),
            "eps_curvature": round(eps_curv, 4),
            "member_median": round(float(np.median(kth[data.is_member])), 4),
            "field_median": round(float(np.median(kth[~data.is_member])), 4),
            "member_over_field": round(
                float(np.median(kth[data.is_member])
                      / np.median(kth[~data.is_member])), 4,
            ),
        })
    return pd.DataFrame(rows)


def dbscan_sweep(
    eps_values: tuple[float, ...] | None = None, min_pts: int = 5,
) -> pd.DataFrame:
    """Run DBSCAN at, below and above the k=5 knee and report degeneracy."""
    from sklearn.cluster import DBSCAN

    from cluster.stability import degeneracy

    data = member_field()
    if eps_values is None:
        _, eps_knee = kneedle(k_distance_curve(data.X, 5))
        eps_values = tuple(eps_knee * f for f in (0.5, 0.75, 1.0, 1.25))

    rows = []
    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(data.X)
        row: dict[str, object] = {"eps": round(float(eps), 4), "minPts": min_pts}
        row.update(degeneracy(labels))
        rows.append(row)
    return pd.DataFrame(rows)


def solve(k_values: tuple[int, ...] = K_VALUES) -> dict[str, object]:
    """Locate the knee at every k, then test the epsilon it recommends."""
    knees = knee_table(k_values)
    sweep = dbscan_sweep()
    eps_by_k = dict(zip(knees["k"].tolist(), knees["eps_kneedle"].tolist()))
    return {
        "knees": knees,
        "eps_kneedle_by_k": eps_by_k,
        "eps_ratio_k20_over_k2": round(float(eps_by_k[20] / eps_by_k[2]), 4)
        if {2, 20} <= set(eps_by_k) else None,
        "knee_rank_fraction_range": (
            round(float(knees["kneedle_rank_fraction"].min()), 4),
            round(float(knees["kneedle_rank_fraction"].max()), 4),
        ),
        "dbscan_at_the_knee": sweep,
        "n_points": int(len(member_field().X)),
    }


def plot(k_values: tuple[int, ...] = K_VALUES):  # pragma: no cover — figure
    """The three sorted k-distance curves with their Kneedle knees."""
    import matplotlib.pyplot as plt

    data = member_field()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k, colour in zip(k_values, ("#4c72b0", "#dd8452", "#55a868")):
        curve = k_distance_curve(data.X, k)
        rank, eps = kneedle(curve)
        ax.plot(np.arange(curve.size) / curve.size, curve, color=colour, lw=1.6,
                label=f"k={k}  (knee eps={eps:.3f})")
        ax.plot([rank / curve.size], [eps], "o", color=colour, ms=7)
    ax.set_xlabel("fraction of points, sorted by k-distance")
    ax.set_ylabel("distance to the k-th neighbour")
    ax.set_title("No knee: the curve is smooth all the way up")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "where the knee is": (
        "Located with the Kneedle rule (maximum deviation from the chord) on "
        "the 25 000-star member+field matrix — the sorted k-distance graph is "
        f"the epsilon-selection heuristic of {cite('Ester:96', bare=True)} "
        "itself: k=2 gives eps = 0.582 at rank "
        "fraction 0.901, k=5 gives eps = 0.632 at 0.903, k=20 gives eps = "
        "0.707 at 0.897. The knee moves up with k, as it must — the k-th "
        "neighbour is further away for larger k — but it moves by only 21% "
        "(0.582 to 0.707) across a factor of ten in k. That is the reassuring "
        "half of the result and the one usually quoted."
    ),
    "why it is not a knee": (
        "The rank fraction is the tell: all three 'knees' sit at 0.90 of the "
        "sorted curve, to within 0.006. A real knee marks a change of regime "
        "— bulk data below, outliers above. A feature that lands at the same "
        "90th percentile regardless of k is not a regime change; it is the "
        "Kneedle rule reporting the widest part of a smooth convex curve. The "
        "independent check confirms it: maximum discrete curvature puts the "
        "knee at rank fraction 0.997 (eps = 0.739) for k=2, 0.991 for k=5 and "
        "0.002 (eps = 0.170) for k=20 — the two rules disagree by a factor of "
        "four in epsilon and by the entire width of the dataset in rank. Two "
        "defensible rules that disagree that badly mean the feature they are "
        "both looking for is not there."
    ),
    "what the recommended epsilon does": (
        "Fed back into DBSCAN "
        f"({cite('Ester:96', bare=True)}, as implemented in scikit-learn, "
        f"{cite('Pedregosa:11', bare=True)}) with minPts=5, the k=5 knee "
        "value eps = 0.632 "
        "produces 2 clusters, the largest containing 96.8% of all 25 000 "
        "stars — the 'one enormous cluster' failure of S 6.3, flagged "
        "degenerate by cluster.stability.degeneracy(). Going up to 1.25x the "
        "knee (eps = 0.790) gives a single cluster containing 100.0% of the "
        "data with 2 noise points. Going down: 0.75x (eps = 0.474) still "
        "gives a degenerate 70.7% blob; only at 0.5x (eps = 0.316) does the "
        "partition stop being degenerate, at 29 clusters with 70.6% of the "
        "sample thrown away as noise. The knee's own recommendation is the "
        "worst of the four."
    ),
    "does a single epsilon exist": (
        "No, and the k-distance curves show why in one number: at every k the "
        "member and field median k-distances differ by 4-6% (k=2: 0.382 vs "
        "0.408; k=5: 0.427 vs 0.450; k=20: 0.502 vs 0.522). The two "
        "populations S 6.3 says are 'at very different densities' are, in "
        "this metric, at nearly the same density — and what separation there "
        "is shrinks as k grows (ratio 0.937 -> 0.948 -> 0.962). There is no "
        "epsilon between the member scale and the field scale because those "
        "two scales overlap. This is the same measurement exercise 5.1 makes "
        "with the core distance, reached from the other direction."
    ),
    "what to do instead": (
        "Precisely what the next two chapters do. HDBSCAN "
        f"(S 7, {cite('Campello:13', bare=True)}; "
        f"{cite('McInnes:17', bare=True)}) keeps every "
        "epsilon at once and scores candidates by how long they survive, so "
        "it never has to pick the one value that does not exist. PLSCAN "
        f"(S 8, {cite('Bot:25', bare=True)}) "
        "goes further and drops the size floor too. Note the failure here is "
        "*not* that the knee-finding code is bad — both rules are correctly "
        "implemented and reproduce their textbook behaviour on data that has "
        "a knee. The failure is that the assumption behind the recipe (two "
        "separable density regimes) is false for this dataset, which is "
        "exactly the row of Table 1's assumption audit that S 6.3 points at."
    ),
    "a reading caution": (
        "These numbers are for the row-normalised matrix, the pipeline "
        "default. Without row normalisation the absolute distances change "
        "completely and so does every epsilon quoted above — see exercise 7.3 "
        "for what that lever does to HDBSCAN. A k-distance plot is only "
        "interpretable alongside the preprocessing that produced it, and an "
        "epsilon copied from a paper that normalised differently is a "
        "meaningless number."
    ),
    "references": reference_list(
        "Ester:96", "Pedregosa:11", "Campello:13", "McInnes:17", "Bot:25",
    ),
}
