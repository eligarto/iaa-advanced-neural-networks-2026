"""Chapter 3, exercise 2: what row-normalisation does to metallicity.

    Standardise the 16-D matrix, then row-normalise it. A star with an
    unusually low [Fe/H] and a star with an unusually high one now have the
    same norm. What did normalisation do to the information about overall
    metallicity, and why is that mostly a feature here rather than a bug?

The exercise is about a transformation the pipeline applies unconditionally
(``NORMALIZE_ROWS``, \\\\S 2.2) and the plausible objection that it throws away
the most obviously physical axis in the data. The answer separates three
questions that are easy to run together:

1. what the row norm *was* before normalisation, and what it encoded;
2. what survives the projection onto the unit sphere;
3. whether the pipeline is better or worse for it: measured, not asserted,
   and with the one metric where it comes out *worse* reported too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from exercises.citations import cite, reference_list
from exercises.utils import members, settings

__all__ = [
    "COMPARISON_CLUSTERS",
    "field_normalisation_effect",
    "hdbscan_effect",
    "metallicity_survival",
    "plot",
    "row_norm_vs_metallicity",
    "solve",
]

#: The element whose information the exercise asks about.
METALLICITY = "FE_H"

#: Clusters that bracket the metallicity range, for the per-cluster table:
#: two metal-poor globulars, two metal-rich open clusters, plus the two
#: smallest clusters in the sample for contrast.
COMPARISON_CLUSTERS: tuple[str, ...] = (
    "M 15", "M 92", "M 3", "M 5",     # metal-poor, [-2.3, -1.2]
    "M 67", "NGC 6819", "NGC 2158",   # near-solar, [+0.0, +0.03]
)

#: Neighbourhood sizes for the field-population check.
K_VALUES: tuple[int, ...] = (5, 15, 30)


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho as a plain float, matching the repo's unpacking idiom."""
    rho, _ = spearmanr(a, b)
    return float(rho)


def _plain_and_normalised() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """The member matrix before and after row normalisation, same rows.

    Returns ``(df, X_standardised, X_normalised, elements)``. Both matrices go
    through :func:`cluster.data.make_matrix`, so the only difference is the
    ``normalize_rows`` flag, not a re-implementation of the recipe.
    """
    from cluster.data import make_matrix

    data = members()
    df = data.df.reset_index(drop=True)
    x_plain = make_matrix(df, settings(normalize_rows=False))
    x_norm = make_matrix(df, settings(normalize_rows=True))
    return df, x_plain, x_norm, list(data.elements)


def row_norm_vs_metallicity() -> dict[str, object]:
    """Characterise the norm that normalisation removes.

    Before normalisation each row's length summarises how far that star sits
    from the median abundance vector, across all 16 elements at once. This
    measures how much of that length is really the metallicity axis.
    """
    df, x_plain, _, elements = _plain_and_normalised()
    norms = np.linalg.norm(x_plain, axis=1)
    z_fe = x_plain[:, elements.index(METALLICITY)]
    raw_fe = df[METALLICITY].to_numpy(dtype=float)

    per_cluster = []
    labels = np.array([str(c) for c in df["cluster"]])
    for name in COMPARISON_CLUSTERS:
        mask = labels == name
        if not mask.any():
            continue
        per_cluster.append({
            "cluster": name,
            "n": int(mask.sum()),
            "median_[Fe/H]": round(float(np.median(raw_fe[mask])), 3),
            "median_row_norm": round(float(np.median(norms[mask])), 3),
        })

    return {
        "row_norm_spread": {
            "min": round(float(norms.min()), 3),
            "median": round(float(np.median(norms)), 3),
            "max": round(float(norms.max()), 4),
            "ratio_max_over_min": round(float(norms.max() / norms.min()), 1),
        },
        "pearson_norm_vs_zFeH": round(float(np.corrcoef(norms, z_fe)[0, 1]), 4),
        "spearman_norm_vs_zFeH": round(_rho(norms, z_fe), 4),
        "per_cluster": pd.DataFrame(per_cluster).set_index("cluster"),
    }


def metallicity_survival() -> dict[str, object]:
    """How much metallicity information survives the projection.

    Two different questions, often conflated: whether the *scale* survives
    (it does not: every row becomes length 1) and whether the *ordering*
    survives (measured here, because it is not obvious).
    """
    from scipy.spatial.distance import pdist

    df, x_plain, x_norm, elements = _plain_and_normalised()
    idx = elements.index(METALLICITY)
    z_plain = x_plain[:, idx]
    z_norm = x_norm[:, idx]
    fe = df[METALLICITY].to_numpy(dtype=float)

    change = pdist(fe.reshape(-1, 1))
    coupling = {}
    for name, x in (("standardised", x_plain), ("row-normalised", x_norm)):
        coupling[name] = round(_rho(pdist(x), change), 4)

    return {
        "std_before": round(float(z_plain.std()), 4),
        "std_after": round(float(z_norm.std()), 4),
        "compression_factor": round(float(z_plain.std() / z_norm.std()), 2),
        "spearman_before_vs_after": round(
            _rho(z_plain, z_norm), 4,
        ),
        "distance_vs_dFeH_spearman": coupling,
    }


def field_normalisation_effect(
    k_values: tuple[int, ...] = K_VALUES, max_stars: int | None = 25_000,
) -> dict[str, dict[int, float]]:
    """Does normalisation help or hurt when 24 000 field stars are present?

    For every member star, the fraction of its ``k`` nearest neighbours that
    are themselves members. Chance is the member share of the population
    (~4\\%), so anything above that is enrichment. Computed for both settings.
    """
    from sklearn.neighbors import NearestNeighbors

    from cluster.data import make_matrix
    from exercises.utils import member_field

    population = member_field(max_stars=max_stars)
    df, is_member = population.df, population.is_member
    out: dict[str, dict[int, float]] = {}
    for flag in (False, True):
        x = make_matrix(df, settings(normalize_rows=flag, max_stars=max_stars))
        n_neighbors = max(k_values) + 1
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(x)
        _, neighbours = nn.kneighbors(x[is_member])
        out["normalised" if flag else "standardised_only"] = {
            k: round(float(is_member[neighbours[:, 1:k + 1]].mean()), 4)
            for k in k_values
        }
    out["chance"] = {k: round(float(is_member.mean()), 4) for k in k_values}
    return out


def hdbscan_effect(max_stars: int | None = 25_000) -> pd.DataFrame:
    """The density-based view: does the field collapse into one blob?

    The kNN metric above is a *local* measure and is not where normalisation
    pays off. HDBSCAN on the same two matrices is the decisive comparison,
    because a density threshold is exactly what a row-length gradient is
    able to destroy.
    """
    import hdbscan

    from cluster.data import make_matrix
    from exercises.utils import member_field

    population = member_field(max_stars=max_stars)
    df = population.df
    rows = []
    for flag in (False, True):
        x = make_matrix(df, settings(normalize_rows=flag, max_stars=max_stars))
        labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(x)
        sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.array([0])
        rows.append({
            "setting": "row-normalised" if flag else "standardised only",
            "n_clusters": int(len(set(labels.tolist()) - {-1})),
            "largest_fraction": round(float(sizes.max() / len(labels)), 4),
            "noise_fraction": round(float((labels == -1).mean()), 4),
        })
    return pd.DataFrame(rows).set_index("setting")


def plot(  # pragma: no cover (figure)
    frame: dict[str, object] | None = None,
):  # pragma: no cover (figure)
    """Row norm against [Fe/H] before normalisation."""
    import matplotlib.pyplot as plt

    df, x_plain, x_norm, elements = _plain_and_normalised()
    idx = elements.index(METALLICITY)
    norm = np.linalg.norm(x_plain, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].scatter(df[METALLICITY], norm, s=6, alpha=0.4, color="#4c72b0", lw=0)
    axes[0].set_xlabel("[Fe/H] (raw)")
    axes[0].set_ylabel("row norm before normalisation")
    axes[0].set_title("the norm is mostly metallicity")
    axes[1].scatter(
        df[METALLICITY], np.abs(x_norm[:, idx]), s=6, alpha=0.4,
        color="#dd8452", lw=0,
    )
    axes[1].set_xlabel("[Fe/H] (raw)")
    axes[1].set_ylabel("|z([Fe/H])| after normalisation")
    axes[1].set_title("after: one row length for everyone")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def solve() -> dict[str, object]:
    """Run all three measurements and answer the question."""
    norms = row_norm_vs_metallicity()
    survival = metallicity_survival()
    field = field_normalisation_effect()
    density = hdbscan_effect()
    return {
        "what the norm encoded": norms,
        "what survives": survival,
        "field population": field,
        "density based view": density,
    }


ANSWER: dict[str, object] = {
    "what normalisation removed": (
        "It removed a quantity that was very nearly a metallicity axis in "
        "disguise. Before normalisation the standardised rows are not all the "
        "same length: they run from 0.51 to 11.38, a factor of 22, and that "
        "length is strongly anti-correlated with [Fe/H] (Pearson -0.87, "
        "Spearman -0.66). Metal-poor stars have *longer* rows, because their "
        "deviations from the median abundance pattern add up across all "
        "sixteen elements at once instead of cancelling. So the premise of "
        "the exercise is exactly right, and stronger than it looks: the two "
        "stars it describes do not merely end up with the same norm: the "
        "norm they are made to share was, to a first approximation, a "
        "rescaled [Fe/H]."
    ),
    "what survived": (
        "The scale went, the ordering stayed. After normalisation every row "
        "has length one by construction, and the [Fe/H] coordinate itself is "
        "compressed by a factor of 4.19 (standard deviation 1.000 -> 0.239). "
        "But the rank order is largely intact: Spearman correlation between "
        "the coordinate before and after is 0.919. Normalisation does not "
        "delete metallicity, it *demotes* it from a distance-dominating "
        "scale to one direction among sixteen. The measurable consequence is "
        "in how much of a pairwise distance metallicity explains: Spearman "
        "between the distance and |d[Fe/H]| falls from 0.708 to 0.548, so "
        "the coupling drops by about a quarter but does not vanish."
    ),
    "why it is a feature": (
        "Because the task is telling siblings apart "
        f"{cite('Freeman:02', 'BlandHawthorn:16')}, and overall metallicity "
        "is the one axis that cannot do it. [Fe/H] is a smooth function of "
        "birth radius and age: two stars from different clusters that formed "
        "at similar Galactocentric radius share it, and two stars from the "
        "same cluster share it with the whole Galactic neighbourhood. Left "
        "in as a scale, it dominates every distance (Spearman 0.708 against "
        "|d[Fe/H]| alone, so about half the distance variance) and pulls "
        "together stars that merely sit at the same metallicity, which is "
        "why the pipeline keeps it as a *direction*. The birth signature is "
        "the pattern, the [X/Fe] ratios relative to each other, which is "
        "also the level at which the abundance differences between similar "
        "clusters are found to be marginal "
        f"{cite('GarciaDias:19', 'Casamiquela:21')}; that is the "
        "part row normalisation preserves. The per-cluster table shows the "
        "effect in the raw numbers: the metal-poor globulars carry row norms "
        "of 8.7-9.2 (M 15, M 92) while the near-solar open clusters sit at "
        "1.1-2.0 (M 67, NGC 6819, NGC 2158)."
    ),
    "the measurement that decides it": (
        "On the 25 000-star field, HDBSCAN "
        f"{cite('Campello:13')}, the reference implementation "
        f"{cite('McInnes:17')}, on the un-normalised matrix "
        "collapses to a degenerate solution: 2 clusters with 88.9% of all "
        "stars in the largest, which is the blob failure the workbook flags "
        "in §7. On the row-normalised matrix the same algorithm returns 30 "
        "clusters with the largest holding 11.0%. A density threshold is "
        "exactly what a row-length gradient destroys, and this is the "
        "comparison that sets the pipeline default, not the local kNN "
        "metric. The caveat that belongs beside it: 'not degenerate' is not "
        "the same as 'good'. The normalised run assigns only 24% of the "
        "field at all (76.0% noise), so what normalisation buys is a usable "
        "density structure, not a solved membership problem. Reporting the "
        "first without the second would be the same overclaim the workbook "
        "retracts for the masked latent in §14."
    ),
    "where the defence is weakest": (
        "Normalisation does not help every metric, and the honest answer "
        "says so. On a purely *local* measure: the fraction of a member "
        "star's k nearest neighbours that are also members: the "
        "un-normalised matrix does slightly *better* at every k tested "
        "(0.363/0.314/0.282 against 0.327/0.280/0.254 at k = 5/15/30). The "
        "reason is that the row norm is itself informative about membership "
        "in this particular sample: the members are mostly metal-poor "
        "globular stars, so a large row norm is a weak membership cue, and "
        "keeping it hands the neighbour search a free hint. That is a "
        "property of this member sample, not of the method, and it is the "
        "kind of accident that makes a benchmark flattering. The density "
        "view is the one that decides the pipeline setting, and it is not "
        "close."
    ),
    "the one-line answer": (
        "Normalisation turned the strongest axis in the data into one "
        "coordinate among sixteen, keeping its direction and discarding its "
        "scale. That is the right trade here, because the scale encodes "
        "where a star formed in the Galaxy (which unrelated stars share) "
        "while the pattern encodes which cluster it formed in (which is the "
        "thing being looked for)."
    ),
    "references": reference_list(
        "Freeman:02", "BlandHawthorn:16", "GarciaDias:19", "Casamiquela:21",
        "Campello:13", "McInnes:17",
    ),
}
