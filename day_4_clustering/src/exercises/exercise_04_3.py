"""Chapter 4, exercise 3 — K=2 asks a different question from K=25.

    K-means with K=2 on our member matrix asks a different question from
    K=25: not "which cluster is this star in" but "which family". Run both
    and compare the homogeneity. Which of the two numbers would you quote to
    support strong chemical tagging, and why is that the wrong number?

\\S 4.6's summary box quotes homogeneity 0.228 at K=2 with completeness
0.954, and 0.449 at K=25 — the trap being that a reader who wanted an
optimistic headline would reach for the completeness. This exercise
reproduces both, adds the seed spread and the chance floor the workbook's
own \\S 9 demands, and shows what the K=2 split is actually cutting on: it
is a 91%-accurate globular/open separation, which is weak tagging by any
name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, members, settings

#: The K values swept, spanning both readings of the question.
K_VALUES: tuple[int, ...] = (2, 3, 5, 10, 25, 40)

#: The number of true clusters in the member matrix.
K_TRUE = 25

#: Null draws per K for the chance-level estimate.
N_NULL = 200


def _matrix(row_normalised: bool) -> tuple[np.ndarray, np.ndarray]:
    """The member matrix in either of the workbook's two conventions.

    ``row_normalised`` True is the L2-normalised C-space every embedding arm
    consumes (\\S 2.3 step 5); False is ``baseline_matrix``'s standardise-only
    matrix, which is what \\S 4.6's summary box and
    ``article/scripts/kmeans_baseline.py`` actually ran on. They give
    different numbers, so the convention has to be stated.
    """
    from cluster.baseline import baseline_matrix

    data = members()
    if row_normalised:
        return data.X, data.labels
    matrix = baseline_matrix(
        data.df, settings(), False, elements=list(data.elements),
    )
    return matrix, data.labels


def kmeans_scores(
    X: np.ndarray, labels: np.ndarray, k: int,
    seeds: tuple[int, ...] = SEEDS, n_init: int = 10,
) -> pd.DataFrame:
    """One row per seed: the paper's metric quartet plus J and blob size."""
    from sklearn.cluster import KMeans

    from cluster.baseline import recovery_fraction, separation_scores

    rows: list[dict[str, object]] = []
    for seed in seeds:
        fit = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
        scores = separation_scores(labels, fit.labels_)
        rows.append({
            "seed": seed,
            **scores,
            "sse": float(fit.inertia_),
            "largest_fraction": float(
                np.bincount(fit.labels_).max() / len(labels),
            ),
            "rf40": float(recovery_fraction(labels, fit.labels_, 0.4)["rf"]),
        })
    return pd.DataFrame(rows)


def chance_homogeneity(
    labels: np.ndarray, sizes: np.ndarray, n_draws: int = N_NULL,
    seed: int = 42,
) -> dict[str, float]:
    """Homogeneity from shuffling labels into groups of the observed sizes.

    \\S 9.4's fifth rule: a mutual-information score's chance level moves with
    the partition shape, so K=2 and K=25 have different floors and cannot be
    compared without them.
    """
    from cluster.baseline import separation_scores

    rng = np.random.default_rng(seed)
    template = np.repeat(np.arange(len(sizes)), sizes)
    draws = []
    for _ in range(n_draws):
        shuffled = template.copy()
        rng.shuffle(shuffled)
        draws.append(separation_scores(labels, shuffled)["homogeneity"])
    return {
        "mean": round(float(np.mean(draws)), 4),
        "p95": round(float(np.percentile(draws, 95)), 4),
    }


def family_split(seed: int = 42) -> dict[str, object]:
    """What is the K=2 partition actually cutting on?"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import completeness_score, homogeneity_score

    from cluster.clusters import CLUSTER_BY_NAME

    X, labels = _matrix(row_normalised=True)
    fit = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X)
    kind = np.array([CLUSTER_BY_NAME[str(c)].kind for c in labels])
    feh = members().df["FE_H"].to_numpy(dtype=float)

    groups = []
    for g in (0, 1):
        mask = fit.labels_ == g
        groups.append({
            "group": g,
            "n_stars": int(mask.sum()),
            "globular_fraction": round(float((kind[mask] == "globular").mean()), 3),
            "median_feh": round(float(np.nanmedian(feh[mask])), 3),
            "n_clusters_touched": int(len({str(c) for c in labels[mask]})),
        })

    guess = np.where(fit.labels_ == 0, "globular", "open")
    accuracy = max(
        float((kind == guess).mean()), float((kind != guess).mean()),
    )
    return {
        "groups": pd.DataFrame(groups),
        "crosstab": pd.crosstab(kind, fit.labels_),
        "homogeneity_vs_kind": round(float(homogeneity_score(kind, fit.labels_)), 4),
        "completeness_vs_kind": round(float(completeness_score(kind, fit.labels_)), 4),
        "accuracy_vs_kind": round(accuracy, 4),
    }


def solve(
    k_values: tuple[int, ...] = K_VALUES, seeds: tuple[int, ...] = SEEDS,
) -> dict[str, object]:
    """Both K values, both matrix conventions, with floors and seed spread."""
    from sklearn.cluster import KMeans

    out: dict[str, object] = {}
    for tag, normalised in (("row_normalised", True), ("standardised", False)):
        X, labels = _matrix(normalised)
        rows: list[dict[str, object]] = []
        for k in k_values:
            per_seed = kmeans_scores(X, labels, k, seeds)
            rows.append({
                "K": k,
                "homogeneity": round(float(per_seed["homogeneity"].mean()), 4),
                "homogeneity_std": round(
                    float(per_seed["homogeneity"].std(ddof=0)), 4,
                ),
                "completeness": round(float(per_seed["completeness"].mean()), 4),
                "v_measure": round(float(per_seed["v_measure"].mean()), 4),
                "accuracy": round(float(per_seed["accuracy"].mean()), 4),
                "sse": round(float(per_seed["sse"].mean()), 3),
                "largest_fraction": round(
                    float(per_seed["largest_fraction"].mean()), 4,
                ),
                "rf40": round(float(per_seed["rf40"].mean()), 4),
            })
        out[tag] = pd.DataFrame(rows)

    X, labels = _matrix(row_normalised=True)
    floors: dict[str, object] = {}
    for k in (2, K_TRUE):
        fit = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        floors[f"K={k}"] = chance_homogeneity(labels, np.bincount(fit.labels_))

    table = out["row_normalised"]
    assert isinstance(table, pd.DataFrame)
    at = {int(k): row for k, row in zip(table["K"], table.to_dict("records"))}

    out["chance_homogeneity"] = floors
    out["family_split"] = family_split()
    out["homogeneity_k2"] = at[2]["homogeneity"]
    out["completeness_k2"] = at[2]["completeness"]
    out["homogeneity_k25"] = at[K_TRUE]["homogeneity"]
    out["completeness_k25"] = at[K_TRUE]["completeness"]
    out["n_stars"] = int(len(labels))
    out["n_clusters"] = int(len(set(labels)))
    return out


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Homogeneity and completeness against K, with the chance floor."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    table = result["row_normalised"]
    assert isinstance(table, pd.DataFrame)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(table["K"], table["homogeneity"], yerr=table["homogeneity_std"],
                marker="o", color="#4c72b0", label="homogeneity", capsize=3)
    ax.plot(table["K"], table["completeness"], "s-", color="#c44e52",
            label="completeness")
    ax.plot(table["K"], table["v_measure"], "^--", color="#55a868",
            label="v-measure")
    floors = result["chance_homogeneity"]
    assert isinstance(floors, dict)
    for key, colour in (("K=2", "#999999"), ("K=25", "#555555")):
        k = int(key.split("=")[1])
        ax.scatter([k], [floors[key]["mean"]], marker="_", s=300,
                   color=colour, label=f"chance homogeneity at {key}")
    ax.axvline(K_TRUE, color="grey", ls=":", lw=1)
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_title("Homogeneity always rises with K; completeness always falls")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the two runs": (
        "On the L2-normalised 1 002 x 16 member matrix, mean over the seven "
        f"workbook seeds with scikit-learn's KMeans {cite('Pedregosa:11')} at "
        "n_init=10: K=2 gives homogeneity 0.188 +- 0.001 "
        "and completeness 0.750; K=25 gives homogeneity 0.560 +- 0.009 and "
        "completeness 0.500. On the standardise-only matrix that §4.6's "
        "summary box actually used (cluster.baseline.baseline_matrix, no row "
        "normalisation) the same protocol gives K=2 homogeneity 0.229 / "
        "completeness 0.954 and K=25 homogeneity 0.459 +- 0.009 / "
        "completeness 0.492 — and at the box's own single seed 42 it "
        "reproduces exactly: 0.2285/0.9545 at K=2 and 0.4490/0.4933 at K=25. "
        "Both conventions are reported here because they differ by far more "
        "than the seed noise, and a number quoted without its matrix "
        "convention is not reproducible."
    ),
    "which number an optimist would quote": (
        "Completeness at K=2 — 0.954 on the standardised matrix. It is the "
        "highest number on the whole table and it sounds like 'we recovered "
        "95% of the structure'. It is the wrong number for two independent "
        "reasons. First, completeness measures whether each true cluster's "
        "stars end up together, and with K=2 almost everything lands in one "
        "group: the largest group holds 63% of the stars on the "
        "standardised matrix (55% normalised). Putting all the stars in a "
        "single group gives completeness exactly 1.0 while telling you "
        "nothing, so completeness at small K is a near-degenerate statistic. "
        "Second, it answers the wrong question: strong tagging is the claim "
        "that you can name the birth cluster, and a two-way split cannot "
        "name anything — the metric that speaks to that claim is homogeneity "
        "(is each found group chemically one cluster?), which at K=2 is "
        "0.188-0.228, the worst on the table."
    ),
    "why homogeneity at K=25 is also not a clean win": (
        "It is the right metric but it is bought with information you would "
        "not have. §4.3 says the workbook *cheats deliberately*: the "
        "catalogue tells us there are 25 clusters "
        f"({cite('Dias:02', 'Harris:96', bare=True)}), so K=25 hands the "
        "algorithm the answer to the hardest part of the question. And "
        "homogeneity rises monotonically with K by construction — measured "
        "here: 0.188 at K=2, 0.246 at K=3, 0.338 at K=5, 0.449 at K=10, "
        "0.560 at K=25, 0.614 at K=40, while completeness falls 0.750, "
        "0.630, 0.589, 0.548, 0.500, 0.475 over the same range. You can "
        "manufacture any homogeneity you like by raising K. That is why §4.6 "
        "insists the pair is quoted together, and why the v-measure (0.301, "
        "0.354, 0.430, 0.493, 0.528, 0.536) is nearly flat from K=25 to "
        "K=40: the extra homogeneity is paid for exactly."
    ),
    "against the chance floor": (
        "§9.4's fifth rule: the chance level of a mutual-information score "
        "moves with the partition shape, so the two K values have different "
        "floors. Shuffling the true labels into groups of the observed sizes "
        "(200 draws, seed 42) gives chance homogeneity 0.0045 (p95 0.0069) "
        "at K=2 and 0.1072 (p95 0.1148) at K=25. So K=2's 0.188 sits 0.18 "
        "above its floor and K=25's 0.560 sits 0.45 above its — K=25 wins on "
        "the excess too, but note that a naive reading of 0.560 against the "
        "'1/25 = 0.04' intuition overstates it by more than a factor of two. "
        "Always subtract the floor for the partition shape you actually "
        "produced, not the one you intended."
    ),
    "what the K=2 split is really cutting on": (
        "Not families of birth sites — chemistry's oldest and coarsest axis. "
        "The K=2 partition on the normalised matrix separates 362 of 365 "
        "globular-cluster stars into one group and 547 of 637 open-cluster "
        "stars into the other: accuracy 0.907 against the open/globular "
        "label, homogeneity 0.628 and completeness 0.599 with respect to "
        "cluster *kind*. The two groups have median [Fe/H] of -1.31 and "
        "-0.00. So K-means at K=2 is a 91%-accurate metallicity cut, and it "
        "touches 22 and 20 of the 25 clusters respectively — no cluster is "
        "cleanly isolated. That is precisely §1.3's *weak* tagging: grouping "
        "stars into chemical families that share a formation epoch, not "
        "recovering birth sites. The distinction belongs to the chemical "
        f"tagging programme itself {cite('Freeman:02', 'BlandHawthorn:16')}, "
        "and K=2 lands on the weak side of it."
    ),
    "the answer to the question as asked": (
        "Neither number supports strong chemical tagging, and quoting either "
        "one alone is the error. K=2's completeness of 0.95 is high because "
        "the partition is nearly degenerate; K=25's homogeneity of 0.56 is "
        "respectable but is produced by telling the algorithm how many "
        "clusters to find and rises further to 0.61 if you simply ask for "
        "40. The number that does speak to strong tagging is the recovery "
        "fraction, which asks per cluster whether a named object came back: "
        "measured here it is 0.04 at K=2 (one cluster of 25) and 0.223 at "
        "K=25. Report the metric pair, the chance floor for that partition "
        "shape, and the recovery fraction; anything less can be made to say "
        "whatever the author wants."
    ),
    "the general lesson": (
        "Choosing K is choosing the question. K=2 asks 'which chemical "
        "family', K=25 asks 'which birth cluster', and no single score "
        "compares answers to two different questions. §4.3's three "
        "heuristics pick K without labels — the elbow in J, the silhouette "
        f"score ({cite('Rousseeuw:87', bare=True)}), and model selection "
        "under a Gaussian mixture fitted by EM "
        f"({cite('Dempster:77', bare=True)}), of which K-means is the "
        "hard-assignment limit — but none of them can pick the *question*; "
        "that is a scientific choice "
        "that has to be made and stated before the clustering runs. The "
        "monotone homogeneity/completeness trade-off measured above is the "
        "mechanism that lets an unstated choice of K turn into an unearned "
        "headline."
    ),
    "references": reference_list(
        "Dias:02", "Harris:96", "Freeman:02", "BlandHawthorn:16",
        "Rousseeuw:87", "Dempster:77", "Pedregosa:11",
    ),
}
