"""Chapter 4, exercise 4: cosine assignment with a Euclidean mean update.

    Replace the Euclidean distance in the assignment step with cosine
    distance, keeping the mean update. What happens to the objective's
    guarantee of monotone decrease, and does the algorithm still terminate?

This is \\S 4.2's proof read backwards. The two halves of that proof are not
symmetric: the assignment step is optimal for *any* distance, but the update
step's optimality is a property of the squared Euclidean distance alone --
the arithmetic mean minimises sum of squared distances, and nothing says it
minimises sum of cosine distances. Break the pairing and the guarantee goes
with it. :func:`solve` exhibits a four-point counterexample where the mean
update *raises* the cosine objective, and shows the one-line fix (normalise
the centroid: spherical k-means).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, members

#: A hand-checkable case where the mean update raises the cosine objective.
COUNTEREXAMPLE: tuple[tuple[float, float], ...] = (
    (-0.6, 0.0), (2.7, -1.1), (-0.4, 0.0), (-1.5, 0.7),
)

#: Initial centres for :data:`COUNTEREXAMPLE` (two of its own points).
COUNTEREXAMPLE_CENTRES: tuple[tuple[float, float], ...] = (
    (-1.5, 0.7), (-0.4, 0.0),
)

#: Synthetic datasets swept when counting how often the guarantee breaks.
N_SYNTHETIC = 200


def cosine_distance(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """``1 - cos(angle)`` between every point and every centre."""
    xs = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    cs = centres / np.maximum(
        np.linalg.norm(centres, axis=1, keepdims=True), 1e-12,
    )
    return 1.0 - xs @ cs.T


def cosine_objective(
    X: np.ndarray, centres: np.ndarray, assign: np.ndarray,
) -> float:
    """The objective the hybrid loop is implicitly trying to minimise."""
    d = cosine_distance(X, centres)
    return float(d[np.arange(len(X)), assign].sum())


def hybrid_lloyd(
    X: np.ndarray, centres: np.ndarray, max_iter: int = 500,
) -> dict[str, object]:
    """Cosine assignment, arithmetic-mean update: the exercise's hybrid.

    Logs the objective after each half-step, so an increase can be attributed
    to the step that caused it. Also detects a non-consecutive repeat of an
    assignment, which is what a genuine cycle would look like.
    """
    current = centres.copy().astype(float)
    trace: list[dict[str, object]] = []
    values: list[float] = []
    seen: dict[tuple[int, ...], int] = {}
    cycle: tuple[int, int] | None = None
    assign = np.zeros(len(X), dtype=int)

    for iteration in range(max_iter):
        assign = cosine_distance(X, current).argmin(1)
        j_assign = cosine_objective(X, current, assign)
        values.append(j_assign)
        trace.append({
            "iteration": iteration, "step": "assign", "J_cos": j_assign,
        })
        key = tuple(assign.tolist())
        if key in seen and iteration - seen[key] > 1 and cycle is None:
            cycle = (seen[key], iteration)
        seen.setdefault(key, iteration)

        updated = current.copy()
        for k in range(len(current)):
            mask = assign == k
            if mask.any():
                updated[k] = X[mask].mean(0)
        j_update = cosine_objective(X, updated, assign)
        values.append(j_update)
        trace.append({
            "iteration": iteration, "step": "update", "J_cos": j_update,
        })
        if np.allclose(updated, current, atol=1e-13):
            current = updated
            break
        current = updated

    increases = [
        {
            "index": i, "step": str(trace[i]["step"]),
            "J_before": round(values[i - 1], 6),
            "J_after": round(values[i], 6),
            "delta": round(values[i] - values[i - 1], 6),
        }
        for i in range(1, len(values))
        if values[i] > values[i - 1] + 1e-12
    ]
    return {
        "centres": current,
        "assignment": assign,
        "trace": pd.DataFrame(trace),
        "increases": increases,
        "monotone": not increases,
        "n_iterations": len(trace) // 2,
        "J_first": values[0],
        "J_last": values[-1],
        "cycle": cycle,
        "hit_cap": len(trace) // 2 >= max_iter,
    }


def spherical_lloyd(
    X: np.ndarray, centres: np.ndarray, max_iter: int = 500,
) -> dict[str, object]:
    """The fix: normalise the centroid, i.e. spherical k-means.

    Under cosine distance the minimiser of the within-group cost is the
    *normalised* mean direction, not the arithmetic mean, so restoring the
    pairing between distance and update restores the monotonicity proof.
    """
    xs = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    current = centres / np.maximum(
        np.linalg.norm(centres, axis=1, keepdims=True), 1e-12,
    )
    values: list[float] = []
    assign = np.zeros(len(xs), dtype=int)

    for _ in range(max_iter):
        d = 1.0 - xs @ current.T
        assign = d.argmin(1)
        values.append(float(d[np.arange(len(xs)), assign].sum()))

        updated = current.copy()
        for k in range(len(current)):
            mask = assign == k
            if mask.any():
                direction = xs[mask].sum(0)
                norm = float(np.linalg.norm(direction))
                if norm > 0:
                    updated[k] = direction / norm
        d = 1.0 - xs @ updated.T
        values.append(float(d[np.arange(len(xs)), assign].sum()))
        if np.allclose(updated, current, atol=1e-13):
            current = updated
            break
        current = updated

    increases = sum(
        1 for a, b in zip(values, values[1:]) if b > a + 1e-12
    )
    return {
        "n_iterations": len(values) // 2,
        "increases": increases,
        "monotone": increases == 0,
        "J_first": values[0],
        "J_last": values[-1],
    }


def synthetic_sweep(n_datasets: int = N_SYNTHETIC) -> dict[str, object]:
    """How often does the hybrid break monotonicity, and at which step?"""
    runs_with_increase = 0
    total_increases = 0
    by_step = {"assign": 0, "update": 0}
    worst = 0.0
    cycles = 0
    capped = 0

    for seed in range(n_datasets):
        rng = np.random.default_rng(seed)
        blobs = rng.normal(0.0, 4.0, size=(5, 3))
        X = blobs[rng.integers(0, 5, 120)] + rng.normal(size=(120, 3))
        centres = X[rng.choice(len(X), 5, replace=False)]
        result = hybrid_lloyd(X, centres, max_iter=200)

        increases = result["increases"]
        assert isinstance(increases, list)
        if increases:
            runs_with_increase += 1
            total_increases += len(increases)
            for entry in increases:
                by_step[str(entry["step"])] += 1
                worst = max(worst, float(entry["delta"]))
        if result["cycle"] is not None:
            cycles += 1
        if result["hit_cap"]:
            capped += 1

    return {
        "n_datasets": n_datasets,
        "runs_with_an_increase": runs_with_increase,
        "total_increasing_moves": total_increases,
        "increases_by_step": by_step,
        "largest_single_increase": round(worst, 6),
        "cycles_detected": cycles,
        "runs_that_hit_the_iteration_cap": capped,
    }


def cycle_hunt(n_datasets: int = 3_000, max_iter: int = 500) -> dict[str, object]:
    """Does the hybrid ever fail to terminate? Brute-force search.

    Small random problems, where a cycle would be easiest to hit: a repeated
    non-consecutive assignment, or a run that exhausts ``max_iter``.
    """
    cycles = 0
    capped = 0
    with_increase = 0

    for seed in range(n_datasets):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(6, 25))
        k = int(rng.integers(2, 5))
        dim = int(rng.integers(2, 5))
        X = rng.normal(size=(n, dim))
        centres = X[rng.choice(n, k, replace=False)]
        result = hybrid_lloyd(X, centres, max_iter=max_iter)
        if result["cycle"] is not None:
            cycles += 1
        if result["hit_cap"]:
            capped += 1
        increases = result["increases"]
        assert isinstance(increases, list)
        if increases:
            with_increase += 1

    return {
        "n_datasets": n_datasets,
        "cycles_detected": cycles,
        "runs_that_hit_the_cap": capped,
        "runs_with_an_increase": with_increase,
    }


def solve() -> dict[str, object]:
    """The counterexample, the sweep, the real matrix, and the fix."""
    X = np.asarray(COUNTEREXAMPLE, dtype=float)
    centres = np.asarray(COUNTEREXAMPLE_CENTRES, dtype=float)
    minimal = hybrid_lloyd(X, centres, max_iter=100)

    data = members()
    real: list[dict[str, object]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        start = data.X[rng.choice(len(data.X), 25, replace=False)]
        result = hybrid_lloyd(data.X, start, max_iter=500)
        increases = result["increases"]
        assert isinstance(increases, list)
        real.append({
            "seed": seed,
            "n_iterations": result["n_iterations"],
            "increasing_moves": len(increases),
            "J_first": round(cast(float, result["J_first"]), 4),
            "J_last": round(cast(float, result["J_last"]), 4),
            "cycle": result["cycle"],
        })

    rng = np.random.default_rng(42)
    start = data.X[rng.choice(len(data.X), 25, replace=False)]
    fixed = spherical_lloyd(data.X, start)

    increases = minimal["increases"]
    assert isinstance(increases, list)
    return {
        "counterexample_points": [list(p) for p in COUNTEREXAMPLE],
        "counterexample_centres": [list(c) for c in COUNTEREXAMPLE_CENTRES],
        "counterexample_increases": increases,
        "counterexample_trace": minimal["trace"],
        "synthetic_sweep": synthetic_sweep(),
        "cycle_hunt": cycle_hunt(),
        "real_matrix": pd.DataFrame(real),
        "spherical_kmeans_fix": fixed,
        "rows_already_unit_norm": bool(
            np.allclose(np.linalg.norm(data.X, axis=1), 1.0),
        ),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover (figure)
    """The cosine objective along the hybrid loop on the counterexample."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    trace = result["counterexample_trace"]
    assert isinstance(trace, pd.DataFrame)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colours = ["#4c72b0" if s == "assign" else "#c44e52"
               for s in trace["step"]]
    ax.plot(range(len(trace)), trace["J_cos"], "-", color="grey", lw=1)
    ax.scatter(range(len(trace)), trace["J_cos"], c=colours, s=60, zorder=5)
    ax.set_xlabel("half-step (blue = assign, red = mean update)")
    ax.set_ylabel(r"$J_{\cos}$")
    ax.set_title("The mean update can raise the cosine objective")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "which half of the proof survives": (
        "The assignment step does, unchanged. Its optimality argument never "
        "used any property of the Euclidean distance: J decomposes into one "
        "independent term per point, and assigning each point to its nearest "
        "centre minimises every term separately. That holds for cosine "
        "distance, Manhattan distance, or any dissimilarity at all. So with "
        "J redefined as sum of cosine distances to the assigned centre, the "
        "assignment step still cannot increase J: measured over 200 "
        "synthetic runs, not one increase was ever attributable to it. Only "
        "the second half of the two-move loop "
        f"{cite('Lloyd:82')} is at risk."
    ),
    "which half breaks": (
        "The update step. Its optimality is a property of the *squared "
        "Euclidean* distance specifically: d/dmu sum ||x - mu||^2 = 0 gives "
        "mu = mean(x), because that objective is a convex quadratic in mu. "
        "That mean-of-the-members rule is the one "
        f"{cite('MacQueen:67', parenthetical=False)} wrote down with the "
        "sum-of-squares objective, and the two belong together. "
        "The cosine objective sum (1 - x.mu/(||x|| ||mu||)) is a different "
        "function of mu, and the arithmetic mean is not its minimiser: it "
        "is not even a stationary point in general, since cosine distance is "
        "scale-invariant in mu while the mean is a statement about "
        "magnitude. So the update step can move the centre to a place where "
        "J_cos is *larger*, and the monotone-decrease guarantee is gone."
    ),
    "a counterexample you can check by hand": (
        "Four points in the plane: (-0.6, 0.0), (2.7, -1.1), (-0.4, 0.0), "
        "(-1.5, 0.7), with the two centres started at (-1.5, 0.7) and "
        "(-0.4, 0.0). The first cosine assignment gives J_cos = 1.926092; "
        "replacing each centre by the arithmetic mean of its members raises "
        "it to 3.696654, an increase of +1.770561 in a single step. The "
        "mechanism is visible in the numbers: point (2.7, -1.1) points "
        "almost opposite to the others, and averaging it with a point that "
        "points the other way produces a mean vector whose *direction* is a "
        "poor summary of either, which is all cosine distance can see. "
        "solve() replays the whole trace."
    ),
    "how often it happens": (
        "Not rare. Over 200 synthetic datasets (120 points, 5 blobs in 3-D, "
        "K=5, seeded from data points), 32 runs (16%) contained at least "
        "one step where J_cos rose, with 57 increasing moves in total and a "
        "largest single increase of 1.431896. All 57 were attributed to the "
        "update step and none to the assignment step, exactly as the theory "
        "predicts. So this is not a pathological construction: it is what "
        "the hybrid does about one run in six."
    ),
    "does it still terminate": (
        "Empirically yes, and the honest answer is 'yes in practice, but the "
        "proof no longer applies'. The finiteness argument of §4.2 runs: J "
        "is non-increasing, bounded below, and takes finitely many values, "
        "so it must become constant. Drop the monotonicity and the first "
        "premise is gone, and with it the whole argument: nothing forbids "
        "the loop from cycling between two assignments forever. The same "
        "dependence shows up in the EM algorithm "
        f"{cite('Dempster:77')}, whose convergence rests entirely on the "
        "likelihood never decreasing: monotonicity is the premise these "
        "proofs are built on, not a bonus property. Measured "
        "(cycle_hunt()): over 3 000 small random datasets (6-24 points, "
        "K=2-4, d=2-4) the hybrid produced zero cycles and zero runs that "
        "hit a 500-iteration cap, while 227 runs contained an increasing "
        "move. So the algorithm does stop, but you are relying on luck "
        "rather than on a theorem, and code that depends on it needs an "
        "explicit iteration cap. Note also what the counterexample's own "
        "trace shows: J_cos goes 1.926 -> 3.697 -> 0.205 -> 0.087, so a "
        "single bad update can be repaired by the next assignment: the "
        "loop is not monotone but it is not obviously divergent either, "
        "which is exactly why the empirical answer and the provable answer "
        "come apart here."
    ),
    "on this workbook's own matrix it never triggers": (
        "A useful special case. §2.3 step 5 L2-normalises every row, so the "
        "member matrix already lives on the unit sphere: verified: all "
        "1 002 row norms are 1 to machine precision. On unit vectors "
        "exercise 2.1's identity applies, ||x-y||^2 = 2 d_cos, so cosine and "
        "Euclidean rank the same pairs and the two assignment steps produce "
        "*identical* assignments. Run the hybrid with K=25 from seven random "
        "seeds on the real matrix and there are zero increasing moves in "
        "every run (16-24 iterations, J_cos falling from ~258-291 to "
        "~176-180). The guarantee is restored not because the hybrid is "
        "fixed but because the data was preprocessed into the one case where "
        "the two metrics agree, which is another way of saying what §2.3 "
        "says about why the row normalisation is there."
    ),
    "the fix": (
        "Pair the update with the distance: spherical k-means. Under cosine "
        "distance the minimiser of sum(1 - x.mu/(||x|| ||mu||)) over a group "
        "of unit vectors is the *normalised* mean direction, "
        "mu = sum(x)/||sum(x)||. Normalising the centroid after the mean "
        "update restores the update step's optimality, hence the monotone "
        "decrease, hence the finiteness argument, hence termination: all "
        "three come back together because they were always one argument. "
        "Verified on the real matrix: spherical k-means at K=25 runs 20 "
        "iterations with zero increasing moves, J_cos 265.996 -> 179.720. "
        "The general lesson is the one worth keeping: in a coordinate-"
        "descent algorithm the distance and the centroid update are a "
        "matched pair, and swapping one without the other silently deletes "
        "the convergence proof. K-medians pairs L1 with the median for the "
        "same reason (§4.1), and K-medoids pairs an arbitrary metric with an "
        "actual data point."
    ),
    "references": reference_list("Lloyd:82", "MacQueen:67", "Dempster:77"),
}
