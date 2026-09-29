"""Chapter 11, exercise 1 — UMAP's per-point sigma, and the role of rho.

    Implement the per-point binary search for sigma_i in
    \\textbf{Equation~\\ref{eq:umapw}} that fixes sum_j w_ij = log_2 k, and
    verify numerically on a random dataset. Where does the local-connectivity
    term rho_i matter most: dense regions or sparse ones?

\\S 11.1 sets UMAP's bandwidth by the same device t-SNE uses — a per-point
binary search — but against a different target: a fixed *total edge weight*
rather than a fixed entropy, and with a local-connectivity offset rho_i
subtracted first. This module implements both halves, diffs them against the
installed ``umap.umap_.smooth_knn_dist``, and then answers the rho question by
ablation rather than assertion: run the same search with and without rho and
see whose bandwidths move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list

#: UMAP's default neighbourhood size (``config.UMAP['n_neighbors']``).
K: int = 15

#: Bisection controls, matching the library's own ``SMOOTH_K_TOLERANCE``.
TOLERANCE: float = 1e-5
MAX_ITERATIONS: int = 64

#: The library floors sigma at this fraction of the local mean distance, so
#: a point whose k neighbours are nearly coincident cannot get sigma ~ 0.
MIN_SIGMA_SCALE: float = 1e-3


def target_weight(k: int = K) -> float:
    """The target total edge weight, ``log2(k)``."""
    return float(np.log2(k))


def binary_search_sigma(
    distances: np.ndarray, rho: float, k: int = K,
    tolerance: float = TOLERANCE, max_iterations: int = MAX_ITERATIONS,
) -> float:
    """Solve sum_j exp(-(d_ij - rho_i)/sigma_i) = log2(k) for one point.

    ``distances`` are the point's k nearest-neighbour distances (self
    excluded). The summand is clipped at 1 for ``d <= rho``, which is what
    makes the nearest neighbour's edge weight exactly 1 — the "local
    connectivity" guarantee that every point has at least one full-strength
    edge, so the graph cannot fragment.

    Total weight is monotonically *increasing* in sigma (a larger bandwidth
    means slower decay means more weight), which is the opposite sense from
    t-SNE's entropy-in-beta search of exercise 10.1.
    """
    target = target_weight(k)
    low, high, mid = 0.0, np.inf, 1.0

    for _ in range(max_iterations):
        total = float(np.sum(np.exp(-np.maximum(distances - rho, 0.0) / mid)))
        if abs(total - target) < tolerance:
            break
        if total > target:          # too much weight -> shrink sigma
            high = mid
            mid = (low + high) / 2.0
        else:                       # too little weight -> grow sigma
            low = mid
            mid = mid * 2.0 if not np.isfinite(high) else (low + high) / 2.0
    return mid


def smooth_knn(
    X: np.ndarray, k: int = K, use_rho: bool = True,
    local_connectivity: int = 1, min_sigma_scale: float = MIN_SIGMA_SCALE,
) -> dict[str, np.ndarray]:
    """Per-point rho_i and sigma_i for the whole dataset.

    With ``use_rho=False`` the local-connectivity offset is switched off,
    which is the ablation that answers the exercise's second question. The
    ``min_sigma_scale`` floor follows the library: sigma is clamped to at
    least that fraction of the dataset's mean neighbour distance.
    """
    from sklearn.neighbors import NearestNeighbors

    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = neighbours.kneighbors(X)
    distances = distances[:, 1:]                      # drop self

    n = len(X)
    rho = np.zeros(n)
    sigma = np.zeros(n)
    floor = min_sigma_scale * float(distances.mean())
    for i in range(n):
        positive = distances[i][distances[i] > 0.0]
        if use_rho and positive.size >= local_connectivity:
            rho[i] = float(positive[local_connectivity - 1])
        sigma[i] = max(binary_search_sigma(distances[i], rho[i], k), floor)
    return {
        "rho": rho,
        "sigma": sigma,
        "knn_distances": distances,
        "mean_knn_distance": distances.mean(axis=1),
    }


def edge_weights(
    solved: dict[str, np.ndarray],
) -> np.ndarray:
    """Realised w_ij per point — the check that the search hit its target."""
    d = solved["knn_distances"]
    rho = solved["rho"][:, None]
    sigma = solved["sigma"][:, None]
    return np.exp(-np.maximum(d - rho, 0.0) / sigma)


def cross_check(X: np.ndarray, k: int = K) -> dict[str, float]:
    """Diff our sigma/rho against the installed ``umap.umap_.smooth_knn_dist``.

    The library's convention is that ``distances[:, 0]`` is self (distance 0)
    and its inner loop runs from column 1, i.e. over exactly the k real
    neighbours — so it must be handed the full ``(n, k+1)`` block, not the
    ``(n, k)`` slice our own search uses.
    """
    from sklearn.neighbors import NearestNeighbors
    from umap.umap_ import smooth_knn_dist

    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = neighbours.kneighbors(X)      # includes self at column 0

    reference_sigma, reference_rho = smooth_knn_dist(
        distances.astype(np.float32), float(k), local_connectivity=1.0,
    )
    ours = smooth_knn(X, k)
    return {
        "max_abs_sigma_difference": round(float(np.max(np.abs(
            ours["sigma"] - reference_sigma))), 8),
        "max_abs_rho_difference": round(float(np.max(np.abs(
            ours["rho"] - reference_rho))), 8),
        "max_relative_sigma_difference": round(float(np.max(np.abs(
            (ours["sigma"] - reference_sigma) / reference_sigma))), 8),
    }


def random_dataset(
    n: int = 400, d: int = 8, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A deliberately density-inhomogeneous cloud: one tight core, one halo.

    A uniform dataset cannot answer the rho question, because every point has
    the same local density. Two populations with a 10x density contrast make
    the dense/sparse comparison meaningful.
    """
    rng = np.random.default_rng(seed)
    dense = rng.normal(0.0, 0.2, size=(n // 2, d))
    sparse = rng.normal(0.0, 2.0, size=(n - n // 2, d))
    labels = np.array(["dense"] * (n // 2) + ["sparse"] * (n - n // 2))
    return np.vstack([dense, sparse]), labels


def solve(k: int = K, n: int = 400, seed: int = 0) -> dict[str, object]:
    """Verify the search, then ablate rho on random and on real data."""
    X, population = random_dataset(n=n, seed=seed)

    with_rho = smooth_knn(X, k, use_rho=True)
    without_rho = smooth_knn(X, k, use_rho=False)

    achieved = edge_weights(with_rho).sum(axis=1)
    achieved_no_rho = edge_weights(without_rho).sum(axis=1)

    frame = pd.DataFrame({
        "population": population,
        "rho": with_rho["rho"],
        "sigma_with_rho": with_rho["sigma"],
        "sigma_without_rho": without_rho["sigma"],
        "mean_knn_distance": with_rho["mean_knn_distance"],
    })
    frame["sigma_ratio"] = (
        frame["sigma_without_rho"] / frame["sigma_with_rho"]
    )
    frame["rho_over_sigma"] = frame["rho"] / frame["sigma_with_rho"]
    by_population = frame.drop(columns=["population"]).groupby(
        frame["population"],
    ).median().round(4)

    from exercises.utils import members

    real = members()
    real_solved = smooth_knn(real.X, k)

    return {
        "target_total_weight": round(target_weight(k), 6),
        "n_random": int(len(X)),
        "achieved_total_weight": {
            "min": round(float(achieved.min()), 6),
            "max": round(float(achieved.max()), 6),
            "max_abs_error": round(float(np.max(np.abs(
                achieved - target_weight(k)))), 8),
        },
        "achieved_without_rho": {
            "max_abs_error": round(float(np.max(np.abs(
                achieved_no_rho - target_weight(k)))), 8),
        },
        "umap_library_cross_check": cross_check(X, k),
        "by_population": by_population,
        "nearest_edge_weight": {
            "with_rho_min": round(
                float(edge_weights(with_rho)[:, 0].min()), 6),
            "without_rho_min": round(
                float(edge_weights(without_rho)[:, 0].min()), 6),
            "without_rho_median": round(
                float(np.median(edge_weights(without_rho)[:, 0])), 6),
        },
        "dr19": {
            "n_stars": int(len(real.X)),
            "sigma_median": round(float(np.median(real_solved["sigma"])), 4),
            "sigma_min": round(float(real_solved["sigma"].min()), 4),
            "sigma_max": round(float(real_solved["sigma"].max()), 4),
            "rho_median": round(float(np.median(real_solved["rho"])), 4),
            "rho_over_sigma_median": round(float(np.median(
                real_solved["rho"] / real_solved["sigma"])), 4),
        },
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """sigma with and without rho, split by local density."""
    import matplotlib.pyplot as plt

    X, population = random_dataset()
    with_rho = smooth_knn(X, K, use_rho=True)
    without_rho = smooth_knn(X, K, use_rho=False)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for name, colour in (("dense", "#4c72b0"), ("sparse", "#dd8452")):
        mask = population == name
        left.scatter(with_rho["mean_knn_distance"][mask],
                     with_rho["sigma"][mask], s=10, alpha=0.6,
                     color=colour, label=name)
        right.scatter(with_rho["mean_knn_distance"][mask],
                      (without_rho["sigma"] / with_rho["sigma"])[mask],
                      s=10, alpha=0.6, color=colour, label=name)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("mean kNN distance")
    left.set_ylabel(r"$\sigma_i$")
    left.set_title("Bandwidth tracks local scale")
    left.legend(frameon=False)
    right.set_xscale("log")
    right.axhline(1.0, color="grey", ls=":", lw=1)
    right.set_xlabel("mean kNN distance")
    right.set_ylabel(r"$\sigma_i$ without $\rho$  /  with $\rho$")
    right.set_title(r"Where $\rho$ changes the answer")
    right.legend(frameon=False)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the algorithm, and how it differs from t-SNE's": (
        "Same device, different target. t-SNE (exercise 10.1) bisects on the "
        "precision until the neighbour distribution has a fixed *entropy* "
        f"{cite('vanderMaaten:08')}; "
        "UMAP bisects on sigma_i until the *total edge weight* "
        "sum_j exp(-(d_ij - rho_i)/sigma_i) equals log2(k) "
        f"{cite('McInnes:18')}. Two further "
        "differences matter. The exponent is linear in distance, not "
        "quadratic. And the sense of the search is reversed: total weight "
        "*increases* with sigma (slower decay, more weight), whereas t-SNE's "
        "entropy *decreases* with beta. Only the k nearest neighbours enter "
        "the sum at all — UMAP never sees a long-range distance, which is the "
        "point §11.1 makes about its 'global structure' claims."
    ),
    "the implementation is verified": (
        "solve() checks it twice. (1) The realised total weight per point "
        "matches the target log2(15) = 3.906891 to within 9.98e-06 for every "
        "point — the 1e-5 bisection tolerance — with and without rho. (2) Our "
        "sigma_i and rho_i are diffed element-by-element against the installed "
        "``umap.umap_.smooth_knn_dist`` on the same 400-point dataset: maximum "
        "absolute difference 1.9e-06 in sigma (7.5e-06 relative) and 2.4e-07 "
        "in rho, i.e. the two agree to the float32 precision the library "
        "works in. (Neighbours come from scikit-learn's ``NearestNeighbors`` "
        f"{cite('Pedregosa:11')}.) Getting there required matching two "
        "library conventions "
        "that are easy to miss: its ``distances[:, 0]`` is self and its inner "
        "loop skips it, and it floors sigma at 1e-3 x the mean neighbour "
        "distance. Both are documented here because they are exactly the kind "
        "of detail that makes a reimplementation silently wrong."
    ),
    "what rho_i is for": (
        "rho_i is the distance to the point's nearest neighbour, subtracted "
        "before the exponential and floored at zero. Its immediate effect is "
        "that the nearest neighbour's edge weight is exactly exp(0) = 1 for "
        "*every* point, no matter how isolated. Measured on the two-population "
        "test set: with rho the minimum nearest-neighbour weight is 1.000; "
        "without it the median is 0.358 and the minimum 0.272. That is the "
        "local-connectivity guarantee "
        f"{cite('McInnes:18')}, quantified — the fuzzy graph is "
        "connected by construction, so no point can be orphaned and no region "
        "dropped for having large distances."
    ),
    "where rho matters most — the answer": (
        "In *sparse* regions. The ablation makes it quantitative. On a "
        "two-population test set (a tight core at scale 0.2 and a diffuse halo "
        "at scale 2.0), the median bandwidth is:\n\n"
        "  population   rho     sigma_with  sigma_without  ratio   rho/sigma\n"
        "  dense       0.335     0.071        0.333        4.86     4.94\n"
        "  sparse      3.522     0.631        3.387        5.55     5.82\n\n"
        "Two readings agree. In absolute terms rho changes sigma by 0.26 in "
        "the dense core and by 2.76 in the halo — an order of magnitude more. "
        "In relative terms the ratio sigma_without/sigma_with is 4.9 in the "
        "core and 5.5 in the halo, so the term is also doing *proportionally* "
        "more work where the data are thin. And the rho/sigma column shows "
        "why: the offset is 4.9 bandwidths wide in the core but 5.8 in the "
        "halo, i.e. it eats a larger share of the kernel's range.\n\n"
        "The mechanism is scale-invariance. In a dense region the nearest "
        "neighbour is close, so rho is small in absolute terms even if it is "
        "several bandwidths wide, and subtracting it barely moves the "
        "exponent's reach. In a sparse region all k distances are large and "
        "similar, so without rho the whole exponential sits far out in its "
        "tail and sigma must grow several-fold to recover the target weight; "
        "with rho the distances are re-referenced to the local scale and the "
        "bandwidth tracks the *local* density instead. rho is what stops "
        "UMAP's graph being dominated by its densest part."
    ),
    "on the DR19 matrix": (
        "solve() also runs the search on the real 1 002-row member matrix at "
        "k = 15: sigma_i median 0.0739 (range 0.0143-0.4004, a factor of 28 "
        "across points) with rho median 0.405 and rho/sigma median 5.18. So "
        "the local-connectivity offset is large in bandwidth units here too, "
        "even though the L2-normalised 16-D C-space of APOGEE abundances "
        f"{cite('Majewski:17')} has far less density "
        "contrast than the synthetic two-population set (exercise 3.1's "
        "concentration result works against strong density contrast). The "
        "sigma spread of 28x is the real finding: it says the DR19 member "
        "cloud is *not* uniform in local scale, which is precisely why a "
        "single fixed bandwidth would be wrong."
    ),
    "the parameter that is really doing the work": (
        "Not rho and not sigma, but k. The target log2(k) and the truncation "
        "of the sum to k terms are both set by n_neighbors, and §11.3 records "
        "what that costs on real data: at n_neighbors = 5 the recall on M 67 "
        "fell from 0.415 to 0.071, while at 30 it was indistinguishable from "
        "the default. sigma_i and rho_i are machinery for making the graph "
        "scale-free *given* k; they cannot rescue a k that is wrong."
    ),
    "references": reference_list(
        "McInnes:18", "vanderMaaten:08", "Pedregosa:11", "Majewski:17",
    ),
}
