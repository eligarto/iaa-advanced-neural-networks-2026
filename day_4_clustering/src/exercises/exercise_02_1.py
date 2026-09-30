"""Chapter 2, exercise 1: why row normalisation makes the metrics agree.

    Take the 16-element vector for one star in the sample. Standardise it and
    then row-normalise it. Show algebraically that after row normalisation
    the Euclidean distance between two stars is a strictly monotone function
    of the cosine distance between them. Why does EVoC care?

This is the proof behind \\S 2.3's fifth pipeline step. The claim is small
and exact, and the reason it matters is a fairness argument: t-SNE and UMAP
use Euclidean distance, EVoC's native metric is cosine, and without this
step the three methods are not looking at the same geometry, so the
benchmark of \\S 13 would be comparing metrics rather than methods.
:func:`solve` checks the identity numerically on the real 1 002 x 16 matrix
to machine precision, and shows how badly it fails without the step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import members

#: Neighbourhood sizes at which the kNN-graph agreement is checked.
K_VALUES: tuple[int, ...] = (5, 15, 30)


def one_star(index: int = 0) -> dict[str, object]:
    """Walk a single star through standardise -> row-normalise.

    Returns the raw ``[X/Fe]`` vector, the pipeline's output vector and its
    norm, so the algebra can be checked against an actual row.
    """
    data = members()
    raw = data.df[data.elements].to_numpy(dtype=float)[index]
    unit = data.X[index]
    return {
        "cluster": str(data.df["cluster"].iloc[index]),
        "elements": list(data.elements),
        "raw_dex": np.round(raw, 4),
        "standardised_normalised": np.round(unit, 4),
        "norm": float(np.linalg.norm(unit)),
    }


def identity_error(X: np.ndarray) -> dict[str, float]:
    """Residual of ``||x-y||^2 == 2 * d_cos(x,y)`` over every pair in ``X``.

    For unit vectors the identity is exact; the returned maxima are therefore
    a floating-point check, not a statistical one.
    """
    from scipy.spatial.distance import pdist

    euclid = pdist(X, metric="euclidean")
    cosine = pdist(X, metric="cosine")
    return {
        "n_pairs": float(euclid.size),
        "max_abs_sq_error": float(np.abs(euclid ** 2 - 2.0 * cosine).max()),
        "max_abs_error": float(np.abs(euclid - np.sqrt(2.0 * cosine)).max()),
        "row_norm_max_deviation": float(
            np.abs(np.linalg.norm(X, axis=1) - 1.0).max(),
        ),
    }


def knn_agreement(X: np.ndarray, k_values: tuple[int, ...] = K_VALUES) -> pd.DataFrame:
    """Do the Euclidean and cosine kNN graphs agree, neighbour for neighbour?

    This is the operational form of the claim: every method in the workbook
    starts from a kNN graph, so "the same geometry" means "the same graph".
    """
    from sklearn.neighbors import NearestNeighbors

    rows: list[dict[str, object]] = []
    for k in k_values:
        euclid = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
        cosine = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
        i_e = euclid.kneighbors(X, return_distance=False)[:, 1:]
        i_c = cosine.kneighbors(X, return_distance=False)[:, 1:]
        same_set = float(
            np.mean([set(a) == set(b) for a, b in zip(i_e, i_c)]),
        )
        rows.append({
            "k": k,
            "identical_neighbour_set": round(same_set, 4),
            "identical_neighbour_order": round(float((i_e == i_c).mean()), 4),
        })
    return pd.DataFrame(rows)


def solve() -> dict[str, object]:
    """Check the identity on the real matrix, with and without the step."""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    normalised = members()
    unnormalised = members(normalize_rows=False)

    with_norm = identity_error(normalised.X)
    without = identity_error(unnormalised.X)

    e_n = pdist(normalised.X, metric="euclidean")
    c_n = pdist(normalised.X, metric="cosine")
    e_u = pdist(unnormalised.X, metric="euclidean")
    c_u = pdist(unnormalised.X, metric="cosine")

    rho_n, _ = spearmanr(e_n, c_n)
    rho_u, _ = spearmanr(e_u, c_u)

    norms_u = np.linalg.norm(unnormalised.X, axis=1)

    return {
        "example_star": one_star(),
        "identity_with_row_normalisation": with_norm,
        "identity_without": without,
        "spearman_with_row_normalisation": round(float(rho_n), 12),
        "spearman_without": round(float(rho_u), 4),
        "knn_agreement_with": knn_agreement(normalised.X),
        "knn_agreement_without": knn_agreement(unnormalised.X, (15,)),
        "unnormalised_row_norms": {
            "min": round(float(norms_u.min()), 3),
            "median": round(float(np.median(norms_u)), 3),
            "max": round(float(norms_u.max()), 3),
        },
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover (figure)
    """Euclidean against cosine distance, with and without the step."""
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist

    result = result if result is not None else solve()
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, normalise, title in (
        (axes[0], True, "row-normalised: exact"),
        (axes[1], False, "not normalised: no relation"),
    ):
        X = members(normalize_rows=normalise).X
        e = pdist(X, metric="euclidean")
        c = pdist(X, metric="cosine")
        pick = rng.choice(e.size, size=min(20_000, e.size), replace=False)
        ax.scatter(c[pick], e[pick], s=2, alpha=0.15, color="#4c72b0", lw=0)
        grid = np.linspace(0, float(c.max()), 200)
        ax.plot(grid, np.sqrt(2 * grid), color="crimson", lw=1.5,
                label=r"$\sqrt{2 d_{\cos}}$")
        ax.set_xlabel("cosine distance")
        ax.set_ylabel("Euclidean distance")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the algebra": (
        "Start from the definition and expand. For any two vectors, "
        "||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y. After L2 row "
        "normalisation ||x|| = ||y|| = 1, so this collapses to "
        "||x - y||^2 = 2 - 2 x.y = 2(1 - x.y). Cosine distance is defined as "
        "d_cos = 1 - x.y/(||x|| ||y||), which on unit vectors is just "
        "1 - x.y. Substituting: ||x - y||^2 = 2 d_cos, i.e. "
        "d_euclid = sqrt(2 d_cos). Since sqrt is strictly increasing on "
        "[0, inf) and d_cos in [0, 2] here, d_euclid is a strictly monotone "
        "function of d_cos: it preserves not just the ordering of pairs but "
        "the identity of every rank. Note the standardisation step plays no "
        "part in the proof; it is the *row* normalisation that does the "
        "work. Standardisation matters for a different reason (§2.3 step 4): "
        "without it, an element with a wide dex range would dominate the "
        "direction of the unit vector."
    ),
    "the numerical check": (
        "Verified on the real 1 002 x 16 matrix over all 501 501 pairs: "
        "max |d_euclid^2 - 2 d_cos| = 2.7e-15 and "
        "max |d_euclid - sqrt(2 d_cos)| = 1.9e-15: machine precision, as an "
        "identity should be. Row norms deviate from 1 by at most 3.3e-16. "
        "Spearman's rank correlation between the two distance vectors is "
        "1.000000000000: the orderings are identical, which is the "
        "monotonicity claim stated as a measurement."
    ),
    "what it buys, operationally": (
        "Every method in this workbook starts by building a k-nearest-"
        "neighbour graph, so 'the same geometry' cashes out as 'the same "
        "graph'. Measured: with row normalisation, the Euclidean and cosine "
        "kNN graphs are identical for 100.00% of stars at k = 5, 15 and 30: "
        "same neighbour set *and* same neighbour order. Without it, only "
        "6.4% of stars get the same neighbour set at k = 15, and the rank "
        "correlation between the two distance vectors falls to 0.361. So "
        "the step is not a cosmetic rescaling: it is the difference between "
        "t-SNE, UMAP and EVoC reading the same graph and reading two "
        "different ones."
    ),
    "why EVoC cares": (
        f"EVoC's native metric {cite('EVoC')} is cosine while t-SNE "
        f"{cite('vanderMaaten:08')} and UMAP {cite('McInnes:18')} are run on "
        "Euclidean (§2.3, §12). Without row normalisation the three arms "
        "would differ in *metric* as well as in method, so any gap between "
        "them in the §13 benchmark would be unattributable. You could not "
        "say whether EVoC lost because its algorithm is worse or because it "
        "was handed a different geometry. After the step the metric "
        "difference is provably nil, so the comparison isolates the "
        "algorithm. That is the fairness argument; it is what makes the "
        "benchmark table mean anything."
    ),
    "the second reason, which is physical": (
        "Normalising to the unit sphere throws away the *magnitude* of the "
        "abundance vector and keeps only its direction: the "
        "abundance *pattern*. That is the right invariance for chemical "
        "tagging as the programme defines it "
        f"{cite('Freeman:02', 'BlandHawthorn:16')}: two stars from the same "
        "birth cloud share a pattern of "
        "element ratios, and an overall scale offset (a metallicity shift, "
        "a systematic pipeline offset) is exactly what you do not want to "
        "cluster on. The unnormalised matrix has row norms from 0.887 to "
        "17.028 with a median of 3.008, so without the step the distance "
        "between two stars is dominated by how extreme each one is rather "
        "than by whether they point the same way."
    ),
    "the practical consequence": (
        "config.py names NORMALIZE_ROWS as the single biggest precision "
        "lever in the pipeline, and §2.3 states the failure mode: without "
        "it, density-based clustering merges the whole field into one blob "
        "(measured on M 67: recall 1.00 / precision 0.03 without, recall "
        "0.34 / precision 0.12 with). The geometric reason is in the numbers "
        "above: on the unit sphere all pairwise distances live in "
        "[0, sqrt(2)] and the density contrast a clusterer needs survives; "
        "in the unnormalised space a 19x spread in row norm means the "
        "densest region is simply wherever the norms are small."
    ),
    "the caveat": (
        "Monotone is not the same as equal. d_euclid = sqrt(2 d_cos) is a "
        "*nonlinear* map, so anything that depends on distance ratios or "
        "differences rather than rank: a fixed DBSCAN epsilon, a kernel "
        "bandwidth, an absolute silhouette value: still changes between the "
        "two metrics. What is preserved exactly is the neighbour ordering, "
        "which is what kNN-graph methods consume; the guarantee does not "
        "extend to methods that consume distances directly."
    ),
    "references": reference_list(
        "EVoC", "vanderMaaten:08", "McInnes:18", "Freeman:02",
        "BlandHawthorn:16",
    ),
}
