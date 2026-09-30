"""Chapter 3, exercise 1: the concentration of distances.

    Sample n=1000 points uniformly in R^d, compute for each point the ratio
    of the distance to its nearest and to its furthest neighbour, and plot
    the mean ratio against d for d = 1..50. At what d does the nearest
    neighbour stop being meaningfully nearer than the furthest one?

This is the measurement behind \\S 3.2's first claim: in high dimension the
contrast between "near" and "far" degrades, so a density threshold tuned in
2-D has to be re-tuned, and the kNN graph of 16-D abundances has edges only
slightly shorter than chance.
"""

from __future__ import annotations

import numpy as np

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS

#: Dimensions swept by :func:`solve`.
DIMENSIONS: tuple[int, ...] = tuple(range(1, 51))

#: "Meaningfully nearer" threshold: the nearest neighbour is no longer
#: meaningfully nearer once it sits above this fraction of the furthest.
#: 0.5 is a reading convention, not a theorem. :func:`solve` reports the
#: whole curve so a different convention can be applied to the same numbers.
MEANINGFUL_RATIO = 0.5


def concentration_curve(
    dimensions: tuple[int, ...] = DIMENSIONS,
    n_points: int = 1000,
    seeds: tuple[int, ...] = SEEDS[:3],
) -> dict[str, np.ndarray]:
    """Mean nearest/furthest distance ratio against dimension.

    For each ``d`` the points are drawn uniformly in the unit cube
    :math:`[0,1]^d`; for each point the ratio is
    ``min_j d(i,j) / max_j d(i,j)`` over the other points. Averaged over
    points, then over seeds: the spread across seeds is returned too, so the
    curve can be read against its own noise.
    """
    from scipy.spatial.distance import pdist, squareform

    means = np.zeros(len(dimensions), dtype=float)
    stds = np.zeros(len(dimensions), dtype=float)

    for i, d in enumerate(dimensions):
        per_seed = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            points = rng.random((n_points, d))
            dist = squareform(pdist(points))
            np.fill_diagonal(dist, np.inf)
            nearest = dist.min(axis=1)
            np.fill_diagonal(dist, -np.inf)
            furthest = dist.max(axis=1)
            per_seed.append(float(np.mean(nearest / furthest)))
        means[i] = float(np.mean(per_seed))
        stds[i] = float(np.std(per_seed))

    return {
        "dimension": np.asarray(dimensions, dtype=int),
        "mean_ratio": means,
        "std_ratio": stds,
    }


def first_dimension_above(
    curve: dict[str, np.ndarray], threshold: float = MEANINGFUL_RATIO,
) -> int:
    """Lowest ``d`` whose mean ratio exceeds ``threshold`` (0 if none does)."""
    above = curve["dimension"][curve["mean_ratio"] > threshold]
    return int(above[0]) if above.size else 0


def plot(curve: dict[str, np.ndarray] | None = None):  # pragma: no cover (figure)
    """Plot the curve; returns the matplotlib figure."""
    import matplotlib.pyplot as plt

    curve = curve if curve is not None else concentration_curve()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(curve["dimension"], curve["mean_ratio"], color="#1f77b4", lw=2)
    ax.fill_between(
        curve["dimension"],
        curve["mean_ratio"] - curve["std_ratio"],
        curve["mean_ratio"] + curve["std_ratio"],
        color="#1f77b4", alpha=0.25, lw=0,
    )
    ax.axhline(MEANINGFUL_RATIO, color="crimson", ls="--", lw=1,
               label=f"ratio = {MEANINGFUL_RATIO}")
    ax.axvline(16, color="grey", ls=":", lw=1, label="16-D C-space")
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel("mean  nearest / furthest  distance")
    ax.set_title("Distances concentrate as dimension grows")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def solve() -> dict[str, object]:
    """Run the sweep and answer the question."""
    curve = concentration_curve()
    by_d = dict(zip(curve["dimension"].tolist(), curve["mean_ratio"].tolist()))
    return {
        "curve": curve,
        "ratio_at_d": {d: round(by_d[d], 3) for d in (1, 2, 5, 10, 16, 25, 50)},
        "crosses_half_at_d": first_dimension_above(curve),
        "ratio_at_16d": round(by_d[16], 3),
    }


ANSWER: dict[str, object] = {
    "what the curve does": (
        "The mean nearest/furthest ratio rises monotonically with d. In 1-D "
        "the nearest neighbour is ~1000x closer than the furthest (ratio "
        "0.001); by d=10 it is 0.26, by d=16 it is 0.37, and by d=50 it is "
        "0.60. Nothing is broken: the points really are uniformly spread. "
        "What degrades is the *contrast* the algorithms rely on: the rise is "
        "steepest over d=1-10 and then flattens, so most of the damage is "
        "already done by the time you reach a dozen dimensions."
    ),
    "where it stops being meaningful": (
        "Under the 0.5 convention the crossing is at d = 30: beyond that, the "
        "nearest neighbour is less than twice as close as the furthest point. "
        "But the number to take away is the shape, not the crossing. At the "
        "16 dimensions of this workbook's C-space the ratio is already 0.37: "
        "the nearest neighbour is under 3x closer than the most distant star "
        "in the sample, against ~1000x in one dimension. There is no sharp d "
        "at which geometry breaks, only a steady erosion, which is why the "
        "answer must be read off the curve with a stated convention."
    ),
    "why it matters here": (
        "Three consequences, each cashed out later in the workbook, and all "
        "three sit under the prior question §3.1 takes from the methods "
        f"chapter behind its framing {cite('GarciaDias:20')}: whether the "
        "data are multipeaked at all, since an algorithm will return groups "
        "either way. (1) A "
        "density threshold is not transferable: DBSCAN's single epsilon "
        "(section 6) has to be re-tuned per dimension, and the k-distance knee "
        "flattens as d grows. (2) The kNN graph that t-SNE "
        f"{cite('vanderMaaten:08')}, UMAP {cite('McInnes:18')} and EVoC all "
        "build from (sections 10-12) has edges only slightly shorter than "
        "chance edges, so the graph is noisier than it looks. (3) It is the "
        "argument for L2 row-normalisation: on the unit sphere Euclidean "
        "distance becomes a monotone function of cosine distance, which is a "
        "better-conditioned comparison in high d than raw Euclidean."
    ),
    "the caveat": (
        "Uniform points in a cube are the worst case: there is no cluster "
        "structure at all, so all the contrast that remains is sampling noise. "
        "Real abundance data has structure, and concentration is weaker where "
        "genuine clusters exist. The experiment sets a floor on the contrast, "
        "not the contrast itself. It also depends on n: with more points the "
        "nearest neighbour gets closer, so quote n alongside d."
    ),
    "references": reference_list(
        "GarciaDias:20", "vanderMaaten:08", "McInnes:18",
    ),
}
