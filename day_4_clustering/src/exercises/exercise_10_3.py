"""Chapter 10, exercise 3: HDBSCAN* min_cluster_size on a fixed t-SNE map.

    Take the t-SNE embedding of the abundances and run HDBSCAN* with
    min_cluster_size = 5, 15 and 50. Report the number of groups and the
    largest group fraction each time. Identify which run is degenerate and
    say what the degeneracy looks like in the map.

\\S 10.3's point is that a "t-SNE row" in the workbook's tables is really
"t-SNE followed by HDBSCAN*", and both steps contribute to the score. This
exercise holds the first step fixed (one embedding, computed once) and moves
only the second, so the whole spread is attributable to the clusterer. It is
also the cheapest demonstration of \\S 9.4's rule 5: the number of groups and
the largest-group fraction move far more than the headline score does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, settings

#: The three floors the exercise asks for, plus two intermediates so the
#: transition between them is visible rather than inferred.
MIN_CLUSTER_SIZES: tuple[int, ...] = (5, 10, 15, 25, 50)


def embedding(seed: int = SEEDS[0], perplexity: int = 30) -> tuple[
    np.ndarray, np.ndarray,
]:
    """One t-SNE map of the member matrix, computed once and reused."""
    from cluster.benchmark import fit_tsne
    from exercises.utils import members

    data = members()
    params = {k: v for k, v in settings().tsne.items() if k != "method"}
    params["perplexity"] = perplexity
    return fit_tsne(data.X, params, seed), data.labels


def sweep(
    Z: np.ndarray, labels: np.ndarray,
    sizes: tuple[int, ...] = MIN_CLUSTER_SIZES,
) -> pd.DataFrame:
    """Cluster one fixed embedding at several size floors."""
    from cluster.baseline import separation_scores
    from cluster.benchmark import cluster_embedding
    from cluster.stability import degeneracy

    rows = []
    for size in sizes:
        pred = cluster_embedding(Z, {
            "min_cluster_size": size,
            "min_samples": None,
            "cluster_selection_epsilon": 0.0,
        })
        scores = separation_scores(labels, pred)
        collapse = degeneracy(pred)
        n_noise = int(str(collapse["n_noise"]))
        rows.append({
            "min_cluster_size": size,
            "homogeneity": round(scores["homogeneity"], 4),
            "completeness": round(scores["completeness"], 4),
            "v_measure": round(scores["v_measure"], 4),
            "accuracy": round(scores["accuracy"], 4),
            "n_groups": collapse["n_clusters"],
            "largest_group": collapse["largest_cluster"],
            "largest_fraction": collapse["largest_fraction"],
            "n_noise": n_noise,
            "noise_fraction": round(n_noise / len(labels), 3),
            "flagged_degenerate": collapse["degenerate"],
        })
    return pd.DataFrame(rows)


def largest_group_composition(
    Z: np.ndarray, labels: np.ndarray, min_cluster_size: int,
) -> pd.DataFrame:
    """Which true clusters end up inside the biggest predicted group.

    This is the "what does the degeneracy look like in the map" half of the
    question: a size floor that is too high does not shrink the answer
    uniformly, it fuses whole clusters into one region.
    """
    from cluster.benchmark import cluster_embedding

    pred = cluster_embedding(Z, {
        "min_cluster_size": min_cluster_size,
        "min_samples": None,
        "cluster_selection_epsilon": 0.0,
    })
    real = pred[pred != -1]
    if real.size == 0:
        return pd.DataFrame({"cluster": [], "n_in_group": [],
                             "share_of_cluster": []})
    values, counts = np.unique(real, return_counts=True)
    biggest = values[counts.argmax()]
    inside = labels[pred == biggest]

    rows = []
    for name in np.unique(inside):
        n_inside = int((inside == name).sum())
        rows.append({
            "cluster": str(name),
            "n_in_group": n_inside,
            "share_of_cluster": round(n_inside / int((labels == name).sum()), 3),
        })
    frame = pd.DataFrame(rows).sort_values("n_in_group", ascending=False)
    return frame.reset_index(drop=True)


def solve(
    seed: int = SEEDS[0], sizes: tuple[int, ...] = MIN_CLUSTER_SIZES,
) -> dict[str, object]:
    """Sweep the size floor over one fixed t-SNE map."""
    Z, labels = embedding(seed)
    table = sweep(Z, labels, sizes)

    worst = int(table.loc[table["homogeneity"].idxmin(), "min_cluster_size"])
    return {
        "seed": seed,
        "n_stars": int(len(labels)),
        "n_true_clusters": int(np.unique(labels).size),
        "embedding": "one t-SNE map, perplexity 30, init='pca'",
        "sweep": table,
        "worst_min_cluster_size": worst,
        "largest_group_at_50": largest_group_composition(Z, labels, 50),
        "largest_group_at_5": largest_group_composition(Z, labels, 5),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover (figure)
    """Score, group count and largest-group fraction against the size floor."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    table = result["sweep"]
    assert isinstance(table, pd.DataFrame)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(table["min_cluster_size"], table["homogeneity"], "o-",
            color="#4c72b0", label="homogeneity")
    ax.plot(table["min_cluster_size"], table["completeness"], "s-",
            color="#55a868", label="completeness")
    ax.plot(table["min_cluster_size"], table["largest_fraction"], "^--",
            color="crimson", label="largest-group fraction")
    ax.set_xlabel("HDBSCAN* min_cluster_size")
    ax.set_ylabel("score")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    twin = ax.twinx()
    twin.bar(table["min_cluster_size"], table["n_groups"], width=
             table["min_cluster_size"] * 0.3, alpha=0.15, color="grey")
    twin.set_ylabel("number of predicted groups")
    ax.set_title("One embedding, five size floors")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the design matters": (
        "The t-SNE map is computed *once* and reused for every size floor. "
        "That is the only way to attribute the spread to HDBSCAN* rather than "
        "to the embedding, and given §9.3's row-order result, re-embedding "
        "per setting would have confounded the two sources of variation "
        "completely. Note also that t-SNE with init='pca' is seed-deterministic "
        "here, so quoting a seed spread for this sweep would be quoting zero. "
        "The embedding is scikit-learn's TSNE "
        f"{cite('Pedregosa:11')} and the clusterer is HDBSCAN* "
        f"{cite('Campello:13')} through its reference implementation "
        f"{cite('McInnes:17')}."
    ),
    "what I measured": (
        "One t-SNE map (perplexity 30, seed 42) of the 1 002-row member "
        "matrix, clustered at five floors:\n\n"
        "  mcs   h      c      groups  largest  largest_frac  noise_frac\n"
        "    5   0.520  0.507     31      146      0.146       0.230\n"
        "   10   0.403  0.516     12      274      0.273       0.160\n"
        "   15   0.345  0.536      6      325      0.324       0.171\n"
        "   25   0.344  0.633      4      334      0.333       0.146\n"
        "   50   0.295  0.556      4      280      0.279       0.319\n\n"
        "Homogeneity falls by 43% from mcs = 5 to mcs = 50 while completeness "
        "mostly *rises*, which is the h/c trade-off of exercise 9.1 playing "
        "out along a hyperparameter: bigger groups are more complete and less "
        "pure. The group count collapses from 31: more than the 25 true "
        "clusters: to 4. Most of the damage is done between 5 and 15."
    ),
    "which run is degenerate": (
        "None of the five is flagged by cluster.stability.degeneracy, and that "
        "is the interesting part of the answer. The flag fires when the "
        "largest group holds ≥ 60% of the stars, and the worst run here peaks "
        "at 33.3% (mcs = 25). Yet the mcs = 25 and mcs = 50 runs return 4 "
        "groups for 25 true clusters, and mcs = 50 sends 32% of the sample to "
        "noise. Those runs are degenerate in the sense the exercise means: "
        "they have stopped resolving the structure, while passing the "
        "automated test, because HDBSCAN*'s noise label absorbs the stars that "
        "would otherwise have inflated the largest group. Lesson: a degeneracy "
        "check on largest_fraction alone is defeated by any method with a "
        "noise class. Pair it with the group count and the noise fraction, "
        "both of which are unambiguous here."
    ),
    "what the degeneracy looks like in the map": (
        "At mcs = 50 the largest predicted group holds 280 stars and its "
        "composition is the whole story: it contains *sixteen* different true "
        "clusters, and they are all open clusters: NGC 7789 entire (47/47), "
        "NGC 1245 (90%), NGC 6819 (90%), NGC 188 (84%), Collinder 261 (77%), "
        "NGC 1798 (69%), plus fractions of M 67, IC 166, NGC 6791, NGC 2420 "
        "and five more. That is not one cluster with stragglers; it is the "
        "metal-rich open-cluster continent of the map read as a single density "
        "peak. Contrast mcs = 5, where the largest group is 146 stars and is "
        "recognisably globular (M 3 (58% of it), M 5, M 15, M 13, M 92) a "
        "real if impure structure. (The open/globular split is the one in the "
        f"standard catalogues, {cite('Dias:02', 'Harris:96', bare=True)}.) "
        "Raising the floor did not merge "
        "well-separated islands; it withdrew the algorithm's licence to split "
        "a continent t-SNE had already failed to break up. The signature to "
        "look for is exactly this: one group drawing members from many true "
        "clusters, sitting in the densest region, with the periphery "
        "relabelled as noise."
    ),
    "the parameter is a claim, not a default": (
        "min_cluster_size says 'I am not interested in groups smaller than "
        "this'. On this sample that is a strong statement: 14 of the 25 "
        "clusters have 25 members or fewer and NGC 2158 has 6, so mcs = 50 "
        "makes 21 of the 25 clusters unfindable *by construction* before any "
        "data is seen. It is also the one genuinely free parameter of the "
        f"HDBSCAN* formulation {cite('Campello:13')}. Everything "
        "else is read off the condensed hierarchy. The workbook's DR17 sweep "
        "found the same lever moving t-SNE recall from 0.087 to 0.404 between "
        "5 and 10 (§12), which is why §12 treats EVoC's lack of a size floor "
        f"{cite('EVoC')} as a genuine advantage: the setting that most "
        "changes the answer is also the one hardest to justify from first "
        "principles."
    ),
    "the reporting rule": (
        "Never print a homogeneity without the group count and the "
        "largest-group fraction next to it (§9.4 rule 5): here the score "
        "moves by 0.23 while the group count moves by a factor of eight, and "
        "the group count is the more informative number. And state "
        "min_cluster_size in the caption: 'clustered with HDBSCAN*' describes "
        "three quite different experiments in this table."
    ),
    "references": reference_list(
        "Campello:13", "McInnes:17", "Pedregosa:11", "Dias:02", "Harris:96",
        "EVoC",
    ),
}
