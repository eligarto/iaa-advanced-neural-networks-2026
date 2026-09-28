"""Chapter 6, exercise 3 — border points, visit order, and reproducibility.

    Border points within epsilon of two core points of different clusters are
    assigned by visit order in the reference implementation. Find two such
    points in a small synthetic example, and construct two visit orders that
    produce different partitions. Why does this matter when comparing
    scikit-learn with a GPU implementation?

\\S 6.2 notes that the DBSCAN result "does not depend on the order in which
points are visited, except for the assignment of border points that are
within epsilon of two different clusters --- an ambiguity the original paper
leaves to the implementation, and a real reproducibility hazard when
comparing libraries." This module builds the smallest dataset that exhibits
it, runs the reference algorithm under two visit orders, and shows the two
partitions differing. Nothing is asserted: the disagreement is constructed
and verified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list

EPS = 1.0
MIN_PTS = 4

#: A five-point rosette: dense enough that every member is core at minPts=4,
#: small enough that the whole blob fits inside one epsilon-ball.
_BLOB = np.array([
    (-0.15, 0.00), (-0.05, 0.12), (0.05, 0.12), (0.15, 0.00), (0.00, -0.12),
])

#: Three such blobs on a line 2.1 apart (so the nearest inter-blob distance
#: is 1.8 > eps: the blobs never merge), plus two bridge points p and q.
#: p sits midway between A and B, q midway between B and C. Each bridge point
#: has only 3 neighbours, so it is *not* core; each is within eps of a core
#: point in the blob on either side, so it is a border point of two clusters.
POINTS: np.ndarray = np.vstack([
    _BLOB + (0.00, 0.0),
    _BLOB + (2.10, 0.0),
    _BLOB + (4.20, 0.0),
    np.array([(1.05, 0.0), (3.15, 0.0)]),
])

NAMES: tuple[str, ...] = (
    "A0", "A1", "A2", "A3", "A4",
    "B0", "B1", "B2", "B3", "B4",
    "C0", "C1", "C2", "C3", "C4",
    "p", "q",
)

#: Visit orders. ``FORWARD`` starts at blob A, ``REVERSE`` at blob C; both
#: reach the bridge points last, so the bridges are claimed by whichever
#: expansion arrived first.
FORWARD: tuple[int, ...] = tuple(range(17))
REVERSE: tuple[int, ...] = tuple(range(10, 15)) + tuple(range(5, 10)) \
    + tuple(range(5)) + (15, 16)


def roles(X: np.ndarray = POINTS, eps: float = EPS,
          min_pts: int = MIN_PTS) -> dict[str, np.ndarray]:
    """Neighbour counts, core flags and the epsilon-adjacency matrix."""
    distance = np.linalg.norm(X[:, None] - X[None], axis=-1)
    within = distance <= eps
    counts = within.sum(axis=1)
    return {
        "distance": distance, "within": within, "counts": counts,
        "core": counts >= min_pts,
    }


def ambiguous_points(X: np.ndarray = POINTS, eps: float = EPS,
                     min_pts: int = MIN_PTS) -> pd.DataFrame:
    """Non-core points reachable from cores in more than one blob."""
    info = roles(X, eps, min_pts)
    blob_of = np.array([0] * 5 + [1] * 5 + [2] * 5 + [-1, -1])
    rows = []
    for i in range(len(X)):
        if info["core"][i]:
            continue
        reach = np.flatnonzero(info["within"][i] & info["core"])
        blobs = sorted({int(blob_of[j]) for j in reach if blob_of[j] >= 0})
        if len(blobs) >= 2:
            rows.append({
                "point": NAMES[i],
                "n_eps": int(info["counts"][i]),
                "core": False,
                "reachable_from": ", ".join(NAMES[j] for j in reach),
                "n_distinct_clusters": len(blobs),
            })
    return pd.DataFrame(rows)


def dbscan_visit(
    order: tuple[int, ...], X: np.ndarray = POINTS, eps: float = EPS,
    min_pts: int = MIN_PTS,
) -> np.ndarray:
    """DBSCAN exactly as Ester et al. (1996) state it, with an explicit order.

    A point first marked noise and later met inside a cluster's expansion is
    reclaimed as a border point of *that* cluster — the "first come, first
    served" rule that makes the output order-dependent.
    """
    info = roles(X, eps, min_pts)
    within, core = info["within"], info["core"]
    unvisited, noise = 0, -1
    labels = np.full(len(X), unvisited, dtype=int)
    cluster_id = 0

    for point in order:
        if labels[point] != unvisited:
            continue
        if not core[point]:
            labels[point] = noise
            continue
        cluster_id += 1
        labels[point] = cluster_id
        queue = [int(j) for j in np.flatnonzero(within[point]) if j != point]
        while queue:
            current = queue.pop(0)
            if labels[current] == noise:
                labels[current] = cluster_id      # border point, reclaimed
                continue
            if labels[current] != unvisited:
                continue
            labels[current] = cluster_id
            if core[current]:
                queue.extend(
                    int(j) for j in np.flatnonzero(within[current]) if j != current
                )
    return labels


def partition(labels: np.ndarray) -> list[list[str]]:
    """Label-invariant view of a clustering: the set of point groups."""
    return sorted(
        sorted(NAMES[i] for i in np.flatnonzero(labels == c))
        for c in sorted({int(v) for v in labels if v > 0})
    )


def _by_count_desc(item: tuple[tuple[tuple[str, ...], ...], int]) -> int:
    """Sort key: most frequent partition first."""
    return -item[1]


def sklearn_partitions(n_permutations: int = 500, seed: int = 42) -> pd.DataFrame:
    """How many distinct partitions scikit-learn gives under row shuffling.

    The algorithm is deterministic given the row order; permuting the rows is
    the only thing that changes, and it stands in for the different traversal
    order a GPU or parallel implementation would use.
    """
    from sklearn.cluster import DBSCAN

    rng = np.random.default_rng(seed)
    tally: dict[tuple[tuple[str, ...], ...], int] = {}
    for _ in range(n_permutations):
        perm = rng.permutation(len(POINTS))
        labels = DBSCAN(eps=EPS, min_samples=MIN_PTS).fit_predict(POINTS[perm])
        inverse = np.empty(len(POINTS), dtype=int)
        inverse[perm] = np.arange(len(POINTS))
        restored = labels[inverse]
        groups = tuple(sorted(
            tuple(sorted(NAMES[i] for i in np.flatnonzero(restored == c)))
            for c in sorted({int(v) for v in restored if v >= 0})
        ))
        tally[groups] = tally.get(groups, 0) + 1
    ordered: list[tuple[tuple[tuple[str, ...], ...], int]] = sorted(
        tally.items(), key=_by_count_desc,
    )
    return pd.DataFrame([
        {"partition": " | ".join(",".join(g) for g in groups), "count": count}
        for groups, count in ordered
    ])


def solve() -> dict[str, object]:
    """Build the example, run both visit orders, prove the partitions differ."""
    from sklearn.metrics import adjusted_rand_score

    forward = dbscan_visit(FORWARD)
    reverse = dbscan_visit(REVERSE)
    permutations = sklearn_partitions()
    return {
        "n_points": int(len(POINTS)),
        "eps": EPS,
        "min_pts": MIN_PTS,
        "neighbour_counts": roles()["counts"].tolist(),
        "ambiguous": ambiguous_points(),
        "partition_forward": partition(forward),
        "partition_reverse": partition(reverse),
        "partitions_differ": partition(forward) != partition(reverse),
        "bridge_labels": {
            "p": {"forward": int(forward[15]), "reverse": int(reverse[15])},
            "q": {"forward": int(forward[16]), "reverse": int(reverse[16])},
        },
        "adjusted_rand_index": round(
            float(adjusted_rand_score(forward, reverse)), 4,
        ),
        "sklearn_distinct_partitions": int(len(permutations)),
        "sklearn_permutation_tally": permutations,
    }


def plot():  # pragma: no cover — figure
    """The three blobs and the two ambiguous bridge points."""
    import matplotlib.pyplot as plt

    info = roles()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.set_aspect("equal")
    core = info["core"]
    ax.scatter(POINTS[core, 0], POINTS[core, 1], c="#4c72b0", s=60, label="core")
    ax.scatter(POINTS[~core, 0], POINTS[~core, 1], c="#dd8452", s=110,
               marker="^", label="border (ambiguous)")
    for i in np.flatnonzero(~core):
        ax.add_patch(plt.Circle(tuple(POINTS[i]), EPS, color="#dd8452",
                                alpha=0.10, fill=True, lw=0.8))
        ax.annotate(NAMES[i], tuple(POINTS[i]), textcoords="offset points",
                    xytext=(0, 12), ha="center")
    ax.set_title(rf"Two border points, each claimed by two clusters "
                 rf"($\varepsilon$={EPS}, minPts={MIN_PTS})")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the construction": (
        "Three five-point rosettes centred at x = 0, 2.1 and 4.2, plus two "
        "bridge points p at (1.05, 0) and q at (3.15, 0), with eps = 1.0 and "
        "minPts = 4. The blobs are 1.8 apart at their nearest, so they never "
        "merge directly. Every blob point has 5-6 neighbours and is core; p "
        "and q have exactly 3 and are not. Verified by solve(): p is within "
        "eps of core points A3 and B0, q of B3 and C0. Each bridge is "
        "therefore a border point of two different clusters simultaneously — "
        "the configuration S 6.2 says the original paper "
        f"{cite('Ester:96')} leaves undefined."
    ),
    "the two visit orders": (
        "Visiting in index order (A, B, C, then p, q), A's expansion reaches "
        "p first and B's reaches q first, giving {A0..A4, p}, {B0..B4, q}, "
        "{C0..C4}. Visiting C first (C, B, A, then p, q), C's expansion takes "
        "q and B's takes p, giving {A0..A4}, {B0..B4, p}, {C0..C4, q}. Same "
        "points, same eps, same minPts, same algorithm — two different "
        "partitions, confirmed by direct comparison in solve(). The adjusted "
        "Rand index between them is 0.646, so this is not a relabelling: two "
        "of the seventeen points genuinely sit in different clusters."
    ),
    "what scikit-learn actually does": (
        "scikit-learn "
        f"{cite('Pedregosa:11')} is deterministic for a fixed row order, "
        "which is what makes this hazard easy to miss — run it twice and you "
        "get the same answer, so it looks reproducible. Shuffle the rows 500 "
        "times and feed the same 17 points back in: solve() finds 4 distinct "
        "partitions. The two 'expected' ones (p with A and q with C, or p and "
        "q each with B) account for 175 and 167 of the 500; the mixed cases "
        "take the other 158. So a third of the time you get a partition that "
        "neither visit order above would have predicted, purely from the "
        "order the rows happened to arrive in."
    ),
    "why it matters for GPU and library comparisons": (
        "A GPU implementation does not do breadth-first search from one seed "
        "point at a time — it processes many points in parallel and resolves "
        "cluster identity by label propagation or union-find, so the "
        "'whichever expansion arrives first' rule is replaced by whichever "
        "thread block finished first, which is not even deterministic run to "
        "run. cuML's DBSCAN, Spark's, and scikit-learn's can therefore all "
        "return different border-point assignments on identical input, "
        "correctly. If you benchmark them against each other and see an ARI "
        "of 0.95 instead of 1.00, the difference may be entirely border "
        "points and say nothing about implementation quality. Equally, a "
        "reproducibility claim of the form 'we reran the pipeline and got the "
        "same clusters' is only meaningful if the row order was also fixed."
    ),
    "how much of the real result is at risk": (
        "In this toy, 2 of 17 points (12%) are ambiguous — deliberately "
        "engineered. On real data the fraction depends on how much of the "
        "sample sits at cluster edges, and for the DR19 matrix at the "
        "epsilon values of exercise 6.2 the clusters are mostly one giant "
        "blob, so border ambiguity is the least of the problems. But the "
        "general rule for the workbook stands (S 9.4): report ARI or "
        "precision/recall between implementations rather than exact label "
        "agreement, and never treat a sub-1.0 agreement as a bug without "
        "first checking whether the disagreeing points are border points. "
        "HDBSCAN avoids the whole issue "
        f"({cite('Campello:13', bare=True)}): it has no border-point "
        "concept, because it never commits to a single epsilon."
    ),
    "references": reference_list("Ester:96", "Pedregosa:11", "Campello:13"),
}
