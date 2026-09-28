"""Chapter 6, exercise 1 — the epsilon-ball classification, by hand.

    Draw the epsilon-ball classification by hand for the 2-D dataset in
    Figure 11 with epsilon = 0.35 and minPts = 4, and mark every point core,
    border or noise. Then change minPts to 6 and say precisely which points
    change status.

\\S 6.1 defines core, border and noise in three sentences; this exercise is
the check that the definitions were read rather than skimmed. The second half
is the real lesson: minPts moves from 4 to 6 — a change of two — and the
partition goes from two clusters and four noise points to one cluster and
thirteen. That fragility is the argument \\S 6.3 makes for the k-distance plot
and the reason chapters 7 and 8 exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list

#: The 19 points of Figure 11 (``scripts/teaching_assets/steps.py``,
#: ``DB_POINTS``), rescaled by 0.35 so that the exercise's epsilon = 0.35
#: reproduces the figure's own epsilon = 1.0 geometry exactly. Without the
#: rescale, epsilon = 0.35 on the figure's coordinates leaves every point
#: isolated and the exercise has no content.
FIGURE_SCALE = 0.35

POINTS: np.ndarray = FIGURE_SCALE * np.array([
    (0.80, 2.35), (1.25, 2.62), (1.70, 2.30), (2.15, 2.60), (2.60, 2.28),
    (3.05, 2.58), (3.50, 2.28), (3.95, 2.55), (4.40, 2.32),   # the chain
    (2.85, 3.35),                                             # above the chain
    (6.05, 1.38), (5.59, 1.05), (5.77, 0.51), (6.33, 0.51), (6.51, 1.05),
    (0.55, 0.95), (1.85, 0.30), (3.95, 0.20), (6.60, 2.90),   # isolated
])

#: Readable names: C = the zigzag chain, R = the tight ring, I = isolated.
NAMES: tuple[str, ...] = (
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "Cabove",
    "R0", "R1", "R2", "R3", "R4",
    "I0", "I1", "I2", "I3",
)

EPS = 0.35


def classify(
    X: np.ndarray = POINTS, eps: float = EPS, min_pts: int = 4,
) -> dict[str, np.ndarray]:
    """Core / border / noise for every point, plus the DBSCAN partition.

    ``min_pts`` counts the point itself, as \\S 6.1's margin note specifies.
    Border points are attached to the first core that reaches them in index
    order — an arbitrary choice, and the subject of exercise 3.
    """
    distance = np.linalg.norm(X[:, None] - X[None], axis=-1)
    within = distance <= eps
    counts = within.sum(axis=1)
    core = counts >= min_pts
    border = (~core) & within[:, core].any(axis=1)
    noise = (~core) & (~border)

    labels = np.full(len(X), -1, dtype=int)
    cluster_id = 0
    for seed in np.flatnonzero(core):
        if labels[seed] >= 0:
            continue
        cluster_id += 1
        stack = [int(seed)]
        while stack:
            current = stack.pop()
            if labels[current] >= 0:
                continue
            labels[current] = cluster_id
            stack.extend(
                int(j) for j in np.flatnonzero(within[current] & core)
                if labels[j] < 0
            )
    for point in np.flatnonzero(border):
        reachable = np.flatnonzero(within[point] & core)
        labels[point] = int(labels[reachable[0]])

    role = np.where(core, "core", np.where(border, "border", "noise"))
    return {
        "counts": counts, "core": core, "border": border, "noise": noise,
        "role": role, "labels": labels,
    }


def table(min_pts_values: tuple[int, ...] = (4, 6)) -> pd.DataFrame:
    """One row per point, one role column per minPts value."""
    frame = pd.DataFrame({
        "point": list(NAMES),
        "x": POINTS[:, 0].round(4),
        "y": POINTS[:, 1].round(4),
    })
    for min_pts in min_pts_values:
        result = classify(min_pts=min_pts)
        frame[f"n_eps(minPts={min_pts})"] = result["counts"]
        frame[f"role(minPts={min_pts})"] = result["role"]
        frame[f"cluster(minPts={min_pts})"] = result["labels"]
    return frame


def changes(low: int = 4, high: int = 6) -> pd.DataFrame:
    """Exactly which points change role between two minPts values."""
    a, b = classify(min_pts=low), classify(min_pts=high)
    moved = np.flatnonzero(a["role"] != b["role"])
    return pd.DataFrame({
        "point": [NAMES[i] for i in moved],
        "n_eps": a["counts"][moved],
        f"role@{low}": a["role"][moved],
        f"role@{high}": b["role"][moved],
    })


def solve(eps: float = EPS) -> dict[str, object]:
    """Classify at minPts=4 and minPts=6 and diff the two."""
    out: dict[str, object] = {"eps": eps, "n_points": int(len(POINTS))}
    for min_pts in (4, 6):
        result = classify(eps=eps, min_pts=min_pts)
        labels = result["labels"]
        sizes = [int((labels == c).sum()) for c in sorted(set(labels[labels > 0].tolist()))]
        out[f"minPts={min_pts}"] = {
            "n_core": int(result["core"].sum()),
            "n_border": int(result["border"].sum()),
            "n_noise": int(result["noise"].sum()),
            "n_clusters": len(sizes),
            "cluster_sizes": sizes,
        }
    out["table"] = table()
    out["changes"] = changes()
    out["n_changed"] = int(len(changes()))
    out["neighbour_counts"] = classify()["counts"].tolist()
    return out


def plot(min_pts: int = 4):  # pragma: no cover — figure
    """Redraw Figure 11b: every point coloured by its role, with the balls."""
    import matplotlib.pyplot as plt

    result = classify(min_pts=min_pts)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.set_aspect("equal")
    for i in np.flatnonzero(result["core"]):
        ax.add_patch(plt.Circle(tuple(POINTS[i]), EPS, color="#4c72b0",
                                alpha=0.07, lw=0.8, fill=True))
    colours = {"core": "#4c72b0", "border": "#dd8452", "noise": "#999999"}
    for role, colour in colours.items():
        mask = result["role"] == role
        ax.scatter(POINTS[mask, 0], POINTS[mask, 1], c=colour, s=70,
                   label=f"{role} ({int(mask.sum())})", zorder=3)
    ax.set_title(rf"$\varepsilon$ = {EPS}, minPts = {min_pts}")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the setup": (
        "The core/border/noise vocabulary and the epsilon-minPts test are "
        f"those of the original DBSCAN paper {cite('Ester:96')}. The "
        "figure's 19 points are used at their own geometry: the teaching "
        "asset draws them with epsilon = 1.0, so the coordinates are scaled "
        "by 0.35 here and the exercise's epsilon = 0.35 reproduces the "
        "picture exactly. The neighbour counts (including the point itself, "
        "as S 6.1 specifies) are, in figure order: 3, 4, 5, 5, 5, 6, 5, 4, 3, "
        "2, 5, 5, 5, 5, 5, 1, 1, 1, 1. Every classification below follows "
        "from that one list of nineteen integers and a threshold."
    ),
    "minPts = 4": (
        "12 core, 3 border, 4 noise, and two clusters. The zigzag chain C1-C7 "
        "are all core (counts 4-6); its two endpoints C0 and C8 (count 3) are "
        "border, as is Cabove (count 2), which sits inside C5's ball but "
        "cannot recruit anyone. The five ring points R0-R4 all have count 5 "
        "and are core, forming a second cluster. The four isolated points "
        "I0-I3 have count 1 — themselves only — and are noise. Cluster 1 has "
        "10 members (the chain plus Cabove), cluster 2 has 5."
    ),
    "minPts = 6, and exactly what changes": (
        "13 of the 19 points change status — this is the answer the exercise "
        "is fishing for. Only C5 (count 6) is still core. C3, C4, C6, C7 fall "
        "core -> border; C1, C2 fall core -> noise (counts 4 and 5, and "
        "neither is inside C5's ball); C0 and C8 fall border -> noise. Cabove "
        "is the one point that does *not* change — it was border at minPts=4 "
        "and is border at minPts=6, because its single core neighbour C5 is "
        "the one point that survives. The entire ring R0-R4 collapses "
        "core -> noise in one step: all five have count 5, one short of the "
        "new threshold, and with no core left among them there is nothing for "
        "them to be a border of. Result: one cluster of 6 (C3, C4, C5, C6, "
        "C7, Cabove) and 13 noise points."
    ),
    "the lesson in the ring": (
        "The ring is the instructive case. It is a perfectly good, visually "
        "obvious cluster — five points in a tight pentagon — and it "
        "disappears completely because of a threshold change of two. Nothing "
        "about the data moved. DBSCAN has no notion of 'this group was nearly "
        "dense enough'; a point is core or it is not, and a group with no "
        "core point is not a cluster at any confidence. That discontinuity is "
        "the whole motivation for HDBSCAN's stability score "
        f"(S 7.2, {cite('Campello:13', bare=True)}), which "
        "asks how *long* a group survives rather than whether it passes one "
        "test."
    ),
    "how this connects to the real data": (
        "The same arithmetic runs on the DR19 matrix, where the pipeline uses "
        "minPts = 5 precisely because it is permissive enough to keep small "
        "clusters like the Pleiades candidates "
        f"(S 6.3; the Pleiades membership is that of "
        f"{cite('Kos:17', bare=True)}). The rule of thumb "
        "minPts ~ d+1 would give 17 here, and by the logic above that would "
        "delete every cluster with fewer than 17 chemically tight members — "
        "which, from the member counts of S 2, is 7 of the 25. The knob is "
        "not a detail; it is a decision about which clusters are allowed to "
        "exist before any data is looked at."
    ),
    "references": reference_list("Ester:96", "Campello:13", "Kos:17"),
}
