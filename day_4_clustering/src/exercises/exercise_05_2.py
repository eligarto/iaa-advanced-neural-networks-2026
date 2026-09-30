"""Chapter 5, exercise 2: kNN purity in raw C-space and after embedding.

    Build the 15-NN graph of the member matrix and compute the purity of the
    known members in the *raw* C-space. Compare with the purity after t-SNE,
    UMAP and EVoC. Which method moves the number most, and does it move it by
    separating members or by compressing the field?

\\S 5.4 introduces kNN purity as the workbook's one hyperparameter-free
cohesion score, and \\S 13 reports it for every method. This exercise
establishes the baseline the table is read against: the raw-space number.
Without it, an embedding's purity has nothing to be compared to, and the
question "did the embedding help?" cannot be answered at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, knn_purity_raw, member_field, members, settings

#: Graph k for this exercise. \S 5.3's UMAP/EVoC graph size, not \S 5.4's
#: scoring k of 10. Both are reported by :func:`solve`.
K = 15

#: Field stars kept in the with-field half of the experiment. The full field
#: is 23 998 rows; t-SNE on that is minutes per seed for no extra insight, so
#: the comparison is run on a 4 000-star random draw and the ratio is stated.
N_FIELD_SUBSAMPLE = 4_000


def macro_purity(Z: np.ndarray, labels: np.ndarray, k: int = K) -> float:
    """Macro-averaged kNN purity of ``labels`` in the space ``Z``."""
    per_cluster = knn_purity_raw(Z, labels, k=k)
    return float(np.mean(list(per_cluster.values()))) if per_cluster else float("nan")


def chance_floor(labels: np.ndarray) -> float:
    """Purity a random neighbour list would score, macro-averaged."""
    counts = pd.Series(labels).value_counts()
    n = int(len(labels))
    return float(np.mean([(int(c) - 1) / (n - 1) for c in counts.to_numpy()]))


def _field_share(Z: np.ndarray, labels: np.ndarray, is_member: np.ndarray,
                 k: int = K) -> float:
    """Mean fraction of a member's ``k`` neighbours that are field stars."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    return float((labels[idx[is_member]] == "field").mean())


def cluster_only(k: int = K, seeds: tuple[int, ...] = SEEDS[:3]) -> pd.DataFrame:
    """Purity on the 1 002-row cluster-only matrix, raw vs embedded."""
    from cluster.benchmark import fit_tsne, fit_umap

    data = members()
    cfg = settings()
    tsne_params = {key: value for key, value in cfg.tsne.items() if key != "method"}

    rows: list[dict[str, object]] = [{
        "space": "raw C-space (16-D)",
        "mean": round(macro_purity(data.X, data.labels, k), 4),
        "std": 0.0,
        "n_seeds": 0,
    }]
    for name, fit, params in (
        ("t-SNE (2-D)", fit_tsne, tsne_params),
        ("UMAP (2-D)", fit_umap, dict(cfg.umap)),
    ):
        scores = [
            macro_purity(fit(data.X, params, seed), data.labels, k)
            for seed in seeds
        ]
        arr = np.asarray(scores, dtype=float)
        rows.append({
            "space": name,
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "n_seeds": len(seeds),
        })
    rows.append({
        "space": "EVoC", "mean": float("nan"), "std": float("nan"), "n_seeds": 0,
    })
    frame = pd.DataFrame(rows)
    frame["chance"] = round(chance_floor(data.labels), 4)
    return frame


def with_field(k: int = K, seed: int = SEEDS[0]) -> dict[str, object]:
    """The same comparison with field stars present: the retrieval task.

    Subsampled to ``N_FIELD_SUBSAMPLE`` field rows (a 4:1 field:member ratio
    rather than the population's 24:1) so three embeddings fit in a notebook
    cell; the purity numbers are therefore optimistic, and the point is the
    *ordering*, not the absolute value.
    """
    from cluster.benchmark import fit_evoc, fit_tsne, fit_umap

    population = member_field()
    rng = np.random.default_rng(seed)
    field_rows = np.flatnonzero(~population.is_member)
    keep = np.concatenate([
        np.flatnonzero(population.is_member),
        rng.choice(field_rows, size=min(N_FIELD_SUBSAMPLE, field_rows.size),
                   replace=False),
    ])
    keep.sort()

    X = np.ascontiguousarray(population.X[keep])
    labels = population.labels[keep]
    is_member = population.is_member[keep]

    cfg = settings()
    tsne_params = {key: value for key, value in cfg.tsne.items() if key != "method"}
    spaces: dict[str, np.ndarray] = {
        "raw C-space (16-D)": X,
        "t-SNE (2-D)": fit_tsne(X, tsne_params, seed),
        "UMAP (2-D)": fit_umap(X, dict(cfg.umap), seed),
    }
    rows = [{
        "space": name,
        "purity": round(macro_purity(Z, labels, k), 4),
        "field_share_of_member_neighbourhood": round(
            _field_share(Z, labels, is_member, k), 4,
        ),
    } for name, Z in spaces.items()]

    evoc_labels = fit_evoc(X, dict(cfg.evoc), seed)
    return {
        "table": pd.DataFrame(rows),
        "n_members": int(is_member.sum()),
        "n_field": int((~is_member).sum()),
        "field_to_member_subsample": round(
            float((~is_member).sum() / is_member.sum()), 1,
        ),
        "field_to_member_population": round(
            float((~population.is_member).sum() / population.is_member.sum()), 1,
        ),
        "member_fraction": round(float(is_member.mean()), 4),
        "evoc_n_labels": int(len(np.unique(evoc_labels))),
        "evoc_noise_fraction": round(float((evoc_labels == -1).mean()), 4),
        "evoc_purity": "undefined. EVoC returns labels, not a space",
    }


def solve(k: int = K) -> dict[str, object]:
    """Raw-space purity, the embedded comparison, and the field check."""
    data = members()
    return {
        "k_graph": k,
        "k_score": 10,
        "raw_purity_k15": round(macro_purity(data.X, data.labels, 15), 4),
        "raw_purity_k10": round(macro_purity(data.X, data.labels, 10), 4),
        "chance_floor": round(chance_floor(data.labels), 4),
        "cluster_only": cluster_only(k),
        "with_field": with_field(k),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover (figure)
    """Bar chart of macro purity per space, against the chance floor."""
    import matplotlib.pyplot as plt

    table = cluster_only()
    finite = table[np.isfinite(table["mean"].to_numpy(dtype=float))]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(finite["space"], finite["mean"], yerr=finite["std"],
           color=["#999999", "#4c72b0", "#dd8452"], capsize=3)
    ax.axhline(float(table["chance"].iloc[0]), color="crimson", ls="--", lw=1,
               label=f"chance = {float(table['chance'].iloc[0]):.3f}")
    ax.set_ylabel(f"macro kNN purity (k={K})")
    ax.set_title("Embedding barely moves purity on the cluster-only matrix")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the raw-space baseline": (
        "On the 1 002-row / 25-cluster member matrix the macro kNN purity in "
        "raw 16-D C-space is 0.365 at k=15 and 0.398 at k=10 (the pipeline's "
        "scoring k of S 5.4). The score is the automated form of the manual "
        "membership test used in the GALAH chemical-tagging analysis of "
        f"{cite('Kos:17', parenthetical=False)}. Draw a polygon around the "
        "visual group and ask whether the members are inside it: with the "
        "polygon replaced by a neighbour list "
        f"({cite('Cover:67', bare=True)}). The chance floor. The purity a "
        "random neighbour list would score, macro-averaged over the 25 "
        "clusters: is "
        "0.039, essentially the 1/25 = 0.04 quoted in S 9. So raw abundances "
        "already carry about nine times chance: a member's 15 nearest "
        "chemical neighbours include roughly five other members of its own "
        "cluster. That is real signal and it is the number every embedding "
        "has to beat."
    ),
    "what the embeddings do to it": (
        f"Measured over three seeds: t-SNE {cite('vanderMaaten:08')} 0.389 "
        f"+/- 0.000, UMAP {cite('McInnes:18')} 0.373 +/- "
        "0.005, against raw 0.365. Both moves are small: t-SNE gains 0.024, "
        "UMAP 0.008, and UMAP's gain does not clear its own seed spread. "
        "t-SNE's zero std is an artifact of init='pca' making it deterministic "
        "given the data, not evidence of stability (S 9.3 shows the same "
        "matrix moving between 0.218 and 0.560 under row permutation). The "
        "honest summary: on the cluster-only matrix, no embedding changes the "
        "neighbourhood structure much, because the 15-NN graph they are built "
        "from is the same graph the raw score reads."
    ),
    "EVoC has no purity": (
        f"EVoC {cite('EVoC')} returns cluster labels, not coordinates: "
        "fit_evoc() is a "
        "fit_predict, and there is no EVoC space in which to count "
        "neighbours. kNN purity is therefore undefined for it, which is why "
        "the workbook's own benchmark table (S 13) prints '---' in the EVoC "
        "purity cell rather than a number. This is not a gap to be filled by "
        "scoring EVoC in raw space: that would just report the raw baseline "
        "and attribute it to EVoC. If you want a comparable number, score "
        "EVoC on a clustering metric (precision/recall, homogeneity) where "
        "all three methods produce the same kind of output."
    ),
    "separating members or compressing the field": (
        "This is the sharp half of the question, and it needs the field to "
        "answer. With 4 000 field stars added to the 1 002 members (a 4:1 "
        "ratio; the real population is 24:1), purity *falls* in every space "
        "and falls furthest after embedding: raw 0.225, t-SNE 0.201, UMAP "
        "0.160, against a 0.200 member fraction. The diagnostic is the share "
        "of a member's neighbourhood that is field: raw 0.482, t-SNE 0.535, "
        "UMAP 0.563. So the embeddings do not separate members from the "
        "field: they pull *more* field stars into member neighbourhoods. "
        "Whatever gain t-SNE showed on the cluster-only matrix was tidying "
        "already-separated groups, not retrieval."
    ),
    "the answer to 'which moves it most'": (
        "t-SNE, by 0.024 on the cluster-only matrix, and the direction "
        "reverses once the field is present, where UMAP moves it most, "
        "downwards, by 0.065. Both effects are small compared with what the "
        "question implies. The methodological reading is that kNN purity is "
        "deliberately insensitive to the embedding: it measures local "
        "neighbourhood composition, and a neighbourhood-preserving embedding "
        "is *designed* not to change that. A large purity gain from t-SNE or "
        "UMAP would be evidence that the embedding had invented structure, "
        "not that it had found it."
    ),
    "what to quote": (
        "Always quote purity with (i) its k, (ii) whether the field is in the "
        "pool, and (iii) the chance floor for that pool. The same data gives "
        "0.398 (k=10, members only), 0.365 (k=15, members only) and 0.225 "
        "(k=15, with a 4:1 field). A factor of nearly two between the most "
        "and least flattering framing, with no change to the method."
    ),
    "references": reference_list(
        "Cover:67", "Kos:17", "vanderMaaten:08", "McInnes:18", "EVoC",
    ),
}
