"""Chapter 4, exercise 2: k-means++ seeding versus restarts.

    Implement k-means++ seeding in ten lines of NumPy and compare the final J
    over 50 random datasets against a uniform-random initialisation. How much
    of the improvement comes from the seeding rule, and how much from the
    multiple restarts?

\\S 4.3 states that k-means++ "typically reaches a better local optimum in
fewer restarts" \\citep{Arthur:07}. This exercise turns "typically" into a
number and, more usefully, separates the two things that are usually bundled
together in ``KMeans(init='k-means++', n_init=10)``. The measured answer is
that the two mechanisms are worth about the same on their own and do not
add up: seeding buys 72% of the baseline J, restarts buy 76%, and both
together buy 81%.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list

#: Number of synthetic datasets in the sweep.
N_DATASETS = 50

#: Restarts per configuration. Sklearn's ``n_init`` default of 10.
N_RESTARTS = 10

#: Ground-truth blob count of the synthetic data, handed to every method.
K = 6


def kmeanspp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: D^2 sampling \\citep{Arthur:07}, in ten lines.

    The first centre is uniform over the data; each subsequent centre is
    sampled with probability proportional to the squared distance to the
    nearest centre already chosen, which spreads the seeds out.
    """
    chosen = [int(rng.integers(X.shape[0]))]
    d2 = ((X - X[chosen[0]]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = float(d2.sum())
        weights = d2 / total if total > 0 else np.full(X.shape[0], 1.0 / X.shape[0])
        nxt = int(rng.choice(X.shape[0], p=weights))
        chosen.append(nxt)
        d2 = np.minimum(d2, ((X - X[nxt]) ** 2).sum(axis=1))
    return X[chosen].copy()


def uniform_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform-random centres inside the data's bounding box.

    The naive baseline, and the one the exercise asks for: centres are not
    required to be data points, so some can start in empty space.
    """
    lo, hi = X.min(axis=0), X.max(axis=0)
    return lo + rng.random((k, X.shape[1])) * (hi - lo)


def sample_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Forgy initialisation: ``k`` distinct data points, chosen uniformly.

    The honest middle term. Comparing k-means++ only against centres in empty
    space overstates the seeding rule's contribution, because half of what it
    fixes is simply "start on a data point".
    """
    return X[rng.choice(X.shape[0], size=k, replace=False)].copy()


def lloyd(
    X: np.ndarray, centres: np.ndarray, max_iter: int = 300, tol: float = 1e-10,
) -> float:
    """Run Lloyd's algorithm to a fixed point and return the final J."""
    current = centres.copy()
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - current[None, :, :]) ** 2).sum(-1)
        assign = d2.argmin(1)
        updated = current.copy()
        for k in range(len(current)):
            mask = assign == k
            if mask.any():
                updated[k] = X[mask].mean(0)
        moved = float(np.abs(updated - current).max())
        current = updated
        if moved < tol:
            break
    d2 = ((X[:, None, :] - current[None, :, :]) ** 2).sum(-1)
    return float(d2.min(1).sum())


def make_dataset(
    rng: np.random.Generator, n: int = 300, k: int = K,
    dim: int = 2, spread: float = 0.6,
) -> np.ndarray:
    """``k`` isotropic Gaussian blobs: the case K-means is designed for.

    Deliberately easy: if the seeding rule does not help here, it will not
    help on the abundance matrix, where §4.4's assumption list is violated.
    """
    centres = rng.normal(0.0, 6.0, size=(k, dim))
    labels = rng.integers(0, k, size=n)
    return centres[labels] + rng.normal(0.0, spread, size=(n, dim))


def sweep(
    n_datasets: int = N_DATASETS, n_restarts: int = N_RESTARTS, k: int = K,
) -> pd.DataFrame:
    """Final J per dataset for each (init rule, restart budget) pair."""
    methods = {
        "kmeans++": kmeanspp, "uniform": uniform_init, "sample": sample_init,
    }
    rows: list[dict[str, object]] = []
    for dataset in range(n_datasets):
        X = make_dataset(np.random.default_rng(1000 + dataset), k=k)
        row: dict[str, object] = {"dataset": dataset}
        for name, init in methods.items():
            values = [
                lloyd(X, init(X, k, np.random.default_rng(
                    100_000 + dataset * 100 + restart,
                )))
                for restart in range(n_restarts)
            ]
            row[f"{name}_1"] = values[0]
            row[f"{name}_{n_restarts}"] = float(np.min(values))
            row[f"{name}_std"] = float(np.std(values))
        rows.append(row)
    return pd.DataFrame(rows)


def solve(
    n_datasets: int = N_DATASETS, n_restarts: int = N_RESTARTS, k: int = K,
) -> dict[str, object]:
    """Decompose the improvement into seeding and restarts."""
    table = sweep(n_datasets, n_restarts, k)
    mean = table.drop(columns="dataset").mean()

    baseline = float(mean["uniform_1"])
    seeding_only = float(mean["kmeans++_1"])
    restarts_only = float(mean[f"uniform_{n_restarts}"])
    both = float(mean[f"kmeans++_{n_restarts}"])

    def gain(value: float) -> float:
        return round(100.0 * (baseline - value) / baseline, 1)

    candidates = pd.DataFrame(table[[
        f"kmeans++_{n_restarts}", f"uniform_{n_restarts}",
        f"sample_{n_restarts}", "kmeans++_1", "uniform_1", "sample_1",
    ]])
    best = candidates.min(axis=1)
    hit_rate = {
        column: round(float((table[column] <= best * 1.001).mean()), 2)
        for column in (
            "kmeans++_1", "uniform_1", "sample_1",
            f"kmeans++_{n_restarts}", f"uniform_{n_restarts}",
            f"sample_{n_restarts}",
        )
    }

    return {
        "per_dataset": table,
        "mean_J": mean.round(3),
        "baseline_uniform_1_restart": round(baseline, 3),
        "seeding_only": round(seeding_only, 3),
        "restarts_only": round(restarts_only, 3),
        "both": round(both, 3),
        "gain_seeding_only_pct": gain(seeding_only),
        "gain_restarts_only_pct": gain(restarts_only),
        "gain_both_pct": gain(both),
        "sample_init_1_restart": round(float(mean["sample_1"]), 3),
        "sample_init_n_restarts": round(float(mean[f"sample_{n_restarts}"]), 3),
        "reaches_best_fraction": hit_rate,
        "restart_spread": {
            name: round(float(mean[f"{name}_std"]), 3)
            for name in ("kmeans++", "uniform", "sample")
        },
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover (figure)
    """Mean final J for each (init rule, restart budget)."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    labels = ["uniform\n1 restart", "k-means++\n1 restart",
              "uniform\n10 restarts", "k-means++\n10 restarts"]
    values: list[float] = [
        cast(float, result["baseline_uniform_1_restart"]),
        cast(float, result["seeding_only"]),
        cast(float, result["restarts_only"]),
        cast(float, result["both"]),
    ]
    colours = ["#c44e52", "#dd8452", "#55a868", "#4c72b0"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(labels, values, color=colours)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 12,
                f"{value:.0f}", ha="center", fontsize=9)
    ax.set_ylabel(f"mean final $J$ over {N_DATASETS} datasets")
    ax.set_title("Seeding and restarts buy about the same thing")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the seeding rule in ten lines": (
        "Pick the first centre uniformly from the data. Keep an array D2 of "
        "each point's squared distance to the nearest centre chosen so far. "
        "For each remaining centre, sample a data point with probability "
        "D2/sum(D2) and update D2 with an elementwise minimum against the "
        "new centre's distances. That is the whole of k-means++ "
        f"{cite('Arthur:07')}. See kmeanspp() in this module. The "
        "D^2 weighting is the point: a point far from every existing centre "
        "is likely to be picked next, so the seeds spread out instead of "
        "clumping in the densest blob. What follows the seeding is the "
        f"ordinary two-move loop {cite('Lloyd:82')}, unchanged: the rule "
        "decides only where that loop starts."
    ),
    "the measured decomposition": (
        "Over 50 synthetic datasets (300 points, 6 isotropic Gaussian blobs "
        "in 2-D, K=6), mean final J: uniform init with 1 restart 1098.8 "
        "(the baseline); k-means++ with 1 restart 306.2, a 72.1% reduction "
        "from seeding alone; uniform init with 10 restarts 261.4, a 76.2% "
        "reduction from restarts alone; k-means++ with 10 restarts 204.7, "
        "81.4% from both. So the honest answer to 'how much comes from "
        "which' is: roughly the same from each, and they do not add. 72 + "
        "76 would be 148, and the combination delivers 81. The two "
        "mechanisms fix the same failure (a badly placed initial centre), so "
        "most of what one repairs the other would have repaired too."
    ),
    "the fairer comparison": (
        "Uniform-random centres in the bounding box is a weak baseline "
        "because it can place a centre in empty space, where it captures no "
        "points and effectively reduces K. Forgy initialisation: k distinct "
        "*data points*, chosen uniformly: is the honest middle term, and it "
        "gets most of the way there: mean J 693.9 at 1 restart and 230.8 at "
        "10, against k-means++'s 306.2 and 204.7. Measured this way, "
        "k-means++'s advantage over plain data-point sampling is real but "
        "much smaller than its advantage over the bounding-box baseline. "
        "Roughly half of what the rule appears to buy is simply 'start on a "
        "data point'; the D^2 weighting buys the other half."
    ),
    "the number that matters more than the mean": (
        "Reliability, not average J. Fraction of the 50 datasets on which "
        "each configuration reaches the best J found by any configuration "
        "for that dataset: k-means++ with 10 restarts 0.98, Forgy with 10 "
        "restarts 0.86, uniform with 10 restarts 0.54; and with a single "
        "restart, k-means++ 0.52, Forgy 0.26, uniform 0.06. A single "
        "uniform-random start finds the best answer 6% of the time. The "
        "spread across the 10 restarts tells the same story: mean standard "
        "deviation of J is 136.9 for k-means++ against 667.5 for uniform and "
        "651.1 for Forgy: k-means++'s real contribution is that its runs "
        "agree with each other."
    ),
    "the practical reading": (
        "If you can afford only one run, the seeding rule is what saves you "
        "(0.52 versus 0.06 hit rate). If you can afford ten, restarts alone "
        "get you most of the way and the seeding rule mainly buys "
        "consistency. scikit-learn's default: init='k-means++', n_init=10 "
        f"{cite('Pedregosa:11')}. "
        "takes both, and the 81.4% figure above is what that default is "
        "worth against the naive alternative on data K-means is designed "
        "for. Note the scope: these are isotropic blobs with K set to the "
        "truth. On data that violates §4.4's assumption list, a better local "
        "optimum of a misspecified objective is not obviously a better "
        "answer: §4.6's 'the objective can be lowered without the answer "
        "getting better'."
    ),
    "caveats on the experiment": (
        "Three. (1) The datasets are the easy case by construction; "
        "well-separated isotropic blobs are exactly where the D^2 rule "
        "works, and the gap narrows on overlapping or elongated structure. "
        "(2) 'Improvement' here is measured in J, which is the quantity "
        "K-means optimises and not the quantity anyone cares about: §4.6 "
        "shows homogeneity and J disagree on the member matrix. (3) The "
        "restart budget and K are fixed at 10 and 6; the balance between the "
        "two mechanisms shifts with both, so quote them with the result. For "
        "what initialisation sensitivity looks like on real spectra rather "
        f"than on blobs, {cite('GarciaDias:18', parenthetical=False)} report "
        "that their K-means solution on 153 847 APOGEE spectra moves with the "
        "initialisation: the seeding rule mitigates that, it does not remove "
        "it."
    ),
    "references": reference_list(
        "Arthur:07", "Lloyd:82", "Pedregosa:11", "GarciaDias:18",
    ),
}
