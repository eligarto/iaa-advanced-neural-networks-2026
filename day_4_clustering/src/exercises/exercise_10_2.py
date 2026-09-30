"""Chapter 10, exercise 2: the perplexity sweep.

    Embed the member matrix with perplexities 5, 30 and 100, and for each
    compute the kNN purity of \\S 5 plus the cluster-only homogeneity. Which
    perplexity would you choose, on which evidence, and does the choice
    change the paper's conclusion?

\\S 10.1 calls perplexity "the parameter you must justify", and \\S 9.4 rule 4
asks for a parameter-free score beside every tuned one. This exercise puts the
two together: the sweep is scored with homogeneity, which depends on the
downstream clusterer and can be tuned, and with kNN purity, which cannot. The
two disagree, and the disagreement is the answer. Because t-SNE with
``init='pca'`` is seed-deterministic, the error bars here are over *row order*
(\\S 9.3), which is the variation that actually exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, settings

#: The three perplexities the exercise names.
PERPLEXITIES: tuple[int, ...] = (5, 30, 100)

#: Row orders per perplexity: the sorted one plus four permutations. Seeds
#: would give zero spread here (init='pca' is deterministic), so the honest
#: error bar is over row order. See exercise 9.2.
N_ORDERS: int = 5

#: Neighbourhood size for kNN purity, matching the workbook's usage.
K: int = 15


def _orders(n: int, n_orders: int = N_ORDERS) -> list[np.ndarray]:
    orders = [np.arange(n)]
    orders += [
        np.random.default_rng(1000 + i).permutation(n)
        for i in range(1, n_orders)
    ]
    return orders


def sweep(
    X: np.ndarray, labels: np.ndarray,
    perplexities: tuple[int, ...] = PERPLEXITIES,
    seed: int = SEEDS[0], n_orders: int = N_ORDERS, k: int = K,
) -> pd.DataFrame:
    """One row per (perplexity, row order): both scores plus degeneracy."""
    from cluster.baseline import separation_scores
    from cluster.benchmark import cluster_embedding, fit_tsne, knn_purity
    from cluster.stability import degeneracy

    cfg = settings()
    rows = []
    for perplexity in perplexities:
        params = {k_: v for k_, v in cfg.tsne.items() if k_ != "method"}
        params["perplexity"] = perplexity
        for i, index in enumerate(_orders(len(X), n_orders)):
            Z = fit_tsne(X[index], params, seed)
            y = labels[index]
            pred = cluster_embedding(Z, cfg.hdbscan)
            purity = knn_purity(Z, y, k=k)
            scores = separation_scores(y, pred)
            collapse = degeneracy(pred)
            rows.append({
                "perplexity": perplexity,
                "row_order": "sorted" if i == 0 else f"random_{i}",
                "knn_purity": round(
                    float(np.mean(list(purity.values()))), 4),
                "homogeneity": round(scores["homogeneity"], 4),
                "completeness": round(scores["completeness"], 4),
                "v_measure": round(scores["v_measure"], 4),
                "n_groups": collapse["n_clusters"],
                "largest_fraction": collapse["largest_fraction"],
            })
    return pd.DataFrame(rows)


def purity_by_cluster_size(
    X: np.ndarray, labels: np.ndarray,
    perplexities: tuple[int, ...] = PERPLEXITIES,
    seed: int = SEEDS[0], k: int = K, small: int = 25, large: int = 45,
) -> pd.DataFrame:
    """Does perplexity help the small clusters or the big ones?

    Perplexity is the number of neighbours each point is *forced* to have, so
    a cluster smaller than the perplexity cannot be represented without
    spilling probability onto non-members. If that argument is right, small
    clusters should suffer as perplexity grows.
    """
    from cluster.benchmark import fit_tsne, knn_purity

    cfg = settings()
    counts = pd.Series(labels).value_counts()
    rows = []
    for perplexity in perplexities:
        params = {k_: v for k_, v in cfg.tsne.items() if k_ != "method"}
        params["perplexity"] = perplexity
        Z = fit_tsne(X, params, seed)
        purity = knn_purity(Z, labels, k=k)
        small_names = [c for c in purity if int(counts[c]) <= small]
        large_names = [c for c in purity if int(counts[c]) >= large]
        rows.append({
            "perplexity": perplexity,
            "n_small_clusters": len(small_names),
            f"purity_le_{small}_members": round(
                float(np.mean([purity[c] for c in small_names])), 4),
            "n_large_clusters": len(large_names),
            f"purity_ge_{large}_members": round(
                float(np.mean([purity[c] for c in large_names])), 4),
        })
    return pd.DataFrame(rows)


def solve(
    perplexities: tuple[int, ...] = PERPLEXITIES, n_orders: int = N_ORDERS,
) -> dict[str, object]:
    """Sweep perplexity; score with a tuned metric and a parameter-free one."""
    from exercises.utils import knn_purity_raw, members

    data = members()
    table = sweep(data.X, data.labels, perplexities, n_orders=n_orders)
    summary = table.drop(columns=["row_order"]).groupby("perplexity").agg(
        ["mean", "std", "min", "max"],
    ).round(4)

    raw = knn_purity_raw(data.X, data.labels, k=K)
    baseline = float(np.mean(list(raw.values())))

    return {
        "n_stars": int(len(data.X)),
        "k": K,
        "n_row_orders": n_orders,
        "note": (
            "error bars are over ROW ORDER, not seed: t-SNE with init='pca' "
            "is seed-deterministic, so a seed loop reports zero spread (§9.3)"
        ),
        "raw_16d_knn_purity": round(baseline, 4),
        "per_run": table,
        "summary": summary,
        "by_cluster_size": purity_by_cluster_size(data.X, data.labels,
                                                  perplexities),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover (figure)
    """The tuned score and the parameter-free one, against perplexity."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    table = result["per_run"]
    assert isinstance(table, pd.DataFrame)
    grouped = table.groupby("perplexity")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for column, colour, label in (
        ("homogeneity", "#4c72b0", "homogeneity (tuned)"),
        ("knn_purity", "#dd8452", f"kNN purity k={K} (parameter-free)"),
    ):
        mean = grouped[column].mean()
        std = grouped[column].std()
        ax.errorbar(np.asarray(mean.index), np.asarray(mean),
                    yerr=np.asarray(std), fmt="o-", color=colour,
                    capsize=4, label=label)
    ax.axhline(float(str(result["raw_16d_knn_purity"])), color="grey", ls=":",
               lw=1, label="kNN purity in raw 16-D C-space")
    ax.set_xscale("log")
    ax.set_xlabel("perplexity")
    ax.set_ylabel("score")
    ax.set_ylim(0, 0.75)
    ax.set_title("A tuned score and a parameter-free one disagree")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "what I measured": (
        "1 002 member rows, seed 42, HDBSCAN* min_cluster_size 5, five row "
        "orders per perplexity (seeds would give zero spread. See §9.3):\n\n"
        "  perp   homogeneity      kNN purity (k=15)   groups\n"
        "     5   0.587 ± 0.007    0.378 ± 0.002       41.4\n"
        "    30   0.473 ± 0.049    0.387 ± 0.003       27.2\n"
        "   100   0.338 ± 0.088    0.373 ± 0.003       15.8\n\n"
        "Homogeneity falls by 42% across the sweep and its row-order spread "
        "grows twelvefold; kNN purity moves by 0.014 in total, less than 4% "
        "of its own value, and is flat within noise. The embeddings are "
        f"scikit-learn's TSNE {cite('Pedregosa:11')} with init='pca', and "
        "the downstream clusterer is HDBSCAN* "
        f"{cite('Campello:13')}, so both halves of a 't-SNE row' are "
        "named before any number is read."
    ),
    "the two scores disagree, and that is the finding": (
        "On homogeneity, perplexity 5 wins outright. On kNN purity: the "
        "parameter-free score of §9.4 rule 4: perplexity 30 is marginally "
        "best and the three are effectively tied. They disagree because they "
        "measure different things: kNN purity asks whether members stay near "
        "members in the map, and t-SNE preserves neighbourhoods about equally "
        "well at all three settings; homogeneity asks how HDBSCAN* carves that "
        "map, and a low perplexity produces many small tight blobs which a "
        "density clusterer splits into many pure groups. The homogeneity "
        "advantage of perplexity 5 is largely an artefact of returning 41 "
        "groups for 25 clusters, not of a better embedding."
    ),
    "which perplexity I would choose": (
        "30, the pipeline default, and the evidence is kNN purity plus "
        "stability rather than homogeneity. Three reasons. (1) The "
        "parameter-free score prefers it, marginally, and cannot be gamed. "
        "(2) Perplexity 5's homogeneity win comes with 41 predicted groups "
        "against 25 true clusters: buying purity with fragmentation, which "
        "is exactly the 'best overlap is a best case' caveat of §9.2. "
        "(3) Perplexity 100 is actively unstable: its row-order spread is "
        "±0.088 and one of the five orders collapsed to two groups with 56% "
        "of the stars in one of them (h = 0.203, c = 0.797): a partition that "
        "would trip the degeneracy flag. Choosing 5 on homogeneity alone would "
        "be §9.3's failure mode 4, tuning on the score you then report. All "
        "three values sit inside the 5-50 range the original paper "
        f"{cite('vanderMaaten:08')} names as typical, and that paper offers "
        "no rule for choosing within it, which is why the choice has to be "
        "made on measured evidence rather than inherited."
    ),
    "the most uncomfortable number": (
        "kNN purity in the raw 16-D C-space is 0.365. In the t-SNE map it is "
        "0.378, 0.387 and 0.373 at perplexity 5, 30 and 100. So the entire "
        "embedding buys at most 0.02 of neighbourhood purity over doing "
        "nothing at all. t-SNE is not finding structure here; it is "
        "rearranging structure that was already weak, and every homogeneity "
        "difference in this sweep is a statement about HDBSCAN*'s behaviour on "
        "a 2-D scatter rather than about chemical tagging. That is the same "
        "conclusion §10.5 reaches from a different direction: 'the "
        "neighbourhoods are preserved, and the neighbourhoods were not "
        "separated to begin with'."
    ),
    "small clusters versus large": (
        "Splitting kNN purity by cluster size makes the mechanism visible. "
        "Clusters with ≤ 25 members score 0.266, 0.290, 0.272 at perplexity "
        "5, 30, 100; clusters with ≥ 45 members score 0.619, 0.606, 0.590. "
        "Large clusters do monotonically *worse* as perplexity grows: their "
        "neighbourhoods get diluted by the field of other clusters, while "
        "small clusters peak at 30. The naive prediction, that a perplexity "
        "above a cluster's size should destroy it, is only half right: the "
        "small clusters are already so impure (0.27-0.29) that there is "
        "little left to destroy. Perplexity is not the binding constraint on "
        "this data; chemical separability is."
    ),
    "does it change the paper's conclusion": (
        "No, and it strengthens it. The workbook's claim is that abundances "
        "alone separate these clusters poorly: Table 'honest' quotes t-SNE "
        "homogeneity 0.560 against 0.942 for kinematics-only. Every "
        "perplexity in this sweep lands between 0.34 and 0.59, all far below "
        "the kinematic ceiling, and the parameter-free score is flat at "
        "0.37-0.39 against a raw-space 0.365 throughout. There is no setting "
        "of this knob that rescues the abundance arm, which is the same "
        "conclusion the strong-chemical-tagging literature reaches from "
        f"cluster-recovery experiments {cite('Casamiquela:21')}; the "
        "t-SNE-on-abundances approach itself follows "
        f"{cite('Kos:17', parenthetical=False)}. What the sweep *does* "
        "change is the confidence interval you should attach to any single "
        "t-SNE number: the row-order spread alone is ±0.05 at the default and "
        "±0.09 at perplexity 100, which is larger than several of the "
        "method-to-method gaps the workbook compares."
    ),
    "references": reference_list(
        "vanderMaaten:08", "Pedregosa:11", "Campello:13", "Kos:17",
        "Casamiquela:21",
    ),
}
