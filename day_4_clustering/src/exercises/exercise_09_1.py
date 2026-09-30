"""Chapter 9, exercise 1: the two limits of homogeneity and completeness.

    Show that a clustering that puts every star in a single group has
    h = 0 and c = 1, and that a clustering that gives every star its own
    group has h = 1 and c ≈ 0. Where does accuracy sit in each limit?

\\S 9.2 pairs \\textbf{Equation~\\ref{eq:merits}}'s $h$ and $c$ deliberately:
each one is trivially maximised by a degenerate partition, so quoting either
alone lets a collapse read as a discovery. This exercise proves the two limits
from the definitions and then checks them numerically: on toy labels and on
the real 1 002-row member matrix, so the floor and the ceiling of every score
in \\textbf{Table~\\ref{tab:honest}} are known before any of them is read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list

#: Toy truth used for the analytic check: three true clusters of three stars.
TOY_TRUTH: tuple[int, ...] = (0, 0, 0, 1, 1, 1, 2, 2, 2)


def _scores(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """The workbook's four merit functions, via ``cluster.baseline``."""
    from cluster.baseline import separation_scores

    return separation_scores(true, pred)


def limits(true_labels: np.ndarray) -> pd.DataFrame:
    """Score the two degenerate partitions plus the perfect one.

    ``single`` puts every star in one group, ``singleton`` gives every star
    its own, ``perfect`` reproduces the truth. The three rows bracket every
    real result: nothing a method returns can fall outside them.
    """
    n = len(true_labels)
    partitions = {
        "single": np.zeros(n, dtype=int),
        "singleton": np.arange(n, dtype=int),
        "perfect": np.unique(true_labels, return_inverse=True)[1],
    }
    rows = []
    for name, pred in partitions.items():
        scores = _scores(np.asarray(true_labels), pred)
        rows.append({
            "partition": name,
            "n_groups": int(np.unique(pred).size),
            **{k: round(v, 6) for k, v in scores.items()},
        })
    return pd.DataFrame(rows)


def accuracy_limits(true_labels: np.ndarray) -> dict[str, float]:
    """Closed forms for accuracy in the two degenerate limits.

    Under the Hungarian matching of ``cluster.baseline._accuracy``:

    * ``single``: the one predicted group can only be matched to one true
      cluster, so accuracy is the largest cluster's share of the sample;
    * ``singleton``: each singleton group matches at most one star, and only
      one singleton per true cluster can be used, so accuracy is
      (number of true clusters) / n.
    """
    labels = np.asarray(true_labels)
    n = len(labels)
    counts = np.bincount(np.unique(labels, return_inverse=True)[1])
    return {
        "single_predicted": float(counts.max() / n),
        "singleton_predicted": float(counts.size / n),
        "n": float(n),
        "n_clusters": float(counts.size),
    }


def solve() -> dict[str, object]:
    """Verify both limits on toy labels and on the real member matrix."""
    from exercises.utils import members

    toy = np.asarray(TOY_TRUTH)
    toy_table = limits(toy)

    data = members()
    real = data.labels
    real_table = limits(real)

    return {
        "toy_truth": TOY_TRUTH,
        "toy": toy_table,
        "toy_accuracy_closed_form": {
            k: round(v, 6) for k, v in accuracy_limits(toy).items()
        },
        "n_stars": int(len(real)),
        "n_clusters": int(np.unique(real).size),
        "members": real_table,
        "member_accuracy_closed_form": {
            k: round(v, 6) for k, v in accuracy_limits(real).items()
        },
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover (figure)
    """Bar chart of h, c, V and accuracy in the three reference partitions."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    table = result["members"]
    assert isinstance(table, pd.DataFrame)

    metrics = ["homogeneity", "completeness", "v_measure", "accuracy"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, (_, row) in enumerate(table.iterrows()):
        ax.bar(x + (i - 1) * 0.27, [row[m] for m in metrics], 0.27,
               label=str(row["partition"]))
    ax.set_xticks(x, ["h", "c", "V", "accuracy"])
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("The degenerate limits on the 1 002-row member matrix")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the single-group limit": (
        "Put every star in one group K. Then K is a constant, so knowing it "
        "tells you nothing about the true label: H(C|K) = H(C), and "
        "h = 1 - H(C|K)/H(C) = 0. In the other direction, knowing C tells you "
        "K exactly (it is always the same group), so H(K|C) = 0 and c = 1. "
        "Verified: on the 1 002-row member matrix the single-group partition "
        "scores exactly h = 0.000000, c = 1.000000, V = 0.000000; the toy "
        "3x3 example gives the same. Note scikit-learn "
        f"{cite('Pedregosa:11')} returns c = 1 here by "
        "convention: H(K) = 0 too, so the ratio is 0/0 and the library "
        "defines it as the perfect value."
    ),
    "the singleton limit": (
        "Give every star its own group. Knowing the group identifies the star "
        "and therefore its true cluster, so H(C|K) = 0 and h = 1 exactly. "
        "Completeness is 1 - H(K|C)/H(K), and here H(K) = log n while "
        "H(K|C) = sum_c p_c log n_c, so c = 1 - (sum_c p_c log n_c)/log n. It "
        "is *not* zero: it only tends to zero as the clusters get large "
        "relative to log n. Measured: on the member matrix (n = 1 002, 25 "
        "clusters of 6-230 stars) c = 0.397, so V = 0.569. On the 3x3 toy "
        "c = 0.500. The 'c ≈ 0' of the exercise statement is the asymptotic "
        "reading; on a sample this small the singleton partition still scores "
        "a respectable-looking V-measure, which is exactly the trap."
    ),
    "where accuracy sits": (
        "Accuracy is the Hungarian-matched diagonal of the confusion matrix, "
        "so it is bounded by how many predicted groups there are to match. "
        "Single group: one group can be matched to one true cluster, so "
        "accuracy = (largest cluster)/n = 230/1002 = 0.2295 on the member "
        "matrix (0.333 on the toy). Singleton: each true cluster can claim at "
        "most one singleton, so accuracy = (number of clusters)/n = "
        "25/1002 = 0.0250 (0.333 on the toy). Both closed forms are confirmed "
        "by solve(). Accuracy is the one metric of the four that punishes both "
        "degeneracies, which is why it belongs in the table, but it is also "
        "the one most sensitive to the number of groups returned, so it cannot "
        "referee on its own either."
    ),
    "why the pair is the point": (
        "Each degenerate partition maxes out exactly one of the pair and "
        "bottoms out the other: h = 0/c = 1 for the collapse, h = 1/c = 0.40 "
        "for the shatter. A method reporting h = 0.95 has told you nothing "
        "until you see its c, and vice versa. This is §9.3's failure mode 1 "
        "made arithmetic: the two PCA rows that sat at homogeneity 0.269 were "
        "not measuring a method, they were measuring a collapse, and the draft "
        "that called the spectral latent 'three times better than PCA' was "
        "comparing against a crash. The V-measure hides it too: the singleton "
        "partition above scores V = 0.569, indistinguishable from the real "
        "t-SNE result on abundances (V = 0.513, solve() of exercise 9.2). "
        "The cluster-only protocol these scores come from: take the known "
        "members, cluster them, compare the partition with the true one: "
        f"follows {cite('GarciaDias:19')}, and §9.4's rule 6 supplies the "
        "matching warning that the chance level moves with the sample: "
        "shuffling the 175 stars and 31 mostly tiny clusters of "
        f"{cite('Casamiquela:21')} into random groups of the observed sizes "
        "already scores V = 0.51 before any method is applied."
    ),
    "the operational rule": (
        "Print h, c, the number of predicted groups and the largest-group "
        "fraction together, always: rule 5 of §9.4. The two degenerate rows "
        "computed here are the floor and ceiling every real row has to be read "
        "against, and they cost one line of code: cluster.stability.degeneracy "
        "flags a partition whose largest group holds ≥ 60% of the stars, which "
        "is the practical version of the h = 0 limit."
    ),
    "references": reference_list(
        "Pedregosa:11", "GarciaDias:19", "Casamiquela:21",
    ),
}
