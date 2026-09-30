"""Chapter 7, exercise 3: row normalisation and the "one blob" failure.

    Run HDBSCAN* on the DR19 member-plus-field matrix with row normalisation
    on and off. Record the number of clusters and the largest cluster
    fraction in each case, and identify which setting produces the "one blob"
    failure of \\S 6.3.

\\S 7.4 says the residual failure of HDBSCAN* on these data "is not the
hierarchy but the space", and \\S 2 calls L2 row normalisation "the single
biggest precision lever" in the pipeline. This is the measurement behind both
claims, and it is the one experiment in this chapter that touches the real
catalogue. The result is unambiguous in one direction and much less flattering
in the other than the headline suggests: both halves are reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import member_field, settings

#: The two settings compared. True is the pipeline default (\S 2).
NORMALIZE_ROWS: tuple[bool, ...] = (True, False)


def run_once(normalize_rows: bool) -> dict[str, object]:
    """Fit HDBSCAN* on the member+field matrix under one normalisation."""
    import hdbscan

    from cluster.baseline import separation_scores
    from cluster.benchmark import _score_one
    from cluster.stability import degeneracy

    data = member_field(normalize_rows=normalize_rows)
    cfg = settings()
    predicted = hdbscan.HDBSCAN(**cfg.hdbscan).fit_predict(data.X)

    scores = _score_one(data.labels, predicted)
    diagnosis = degeneracy(predicted)
    separation = separation_scores(data.labels, predicted)

    values, counts = np.unique(predicted[predicted != -1], return_counts=True)
    field_share = float("nan")
    if counts.size:
        biggest = values[int(np.argmax(counts))]
        in_biggest = predicted == biggest
        field_share = float((~data.is_member[in_biggest]).mean())

    n_clusters = int(values.size)
    largest = int(counts.max()) if counts.size else 0
    n_noise = int((predicted == -1).sum())
    return {
        "normalize_rows": normalize_rows,
        "n_stars": int(len(data.X)),
        "n_clusters": n_clusters,
        "largest_cluster": largest,
        "largest_fraction": round(largest / len(data.X), 3),
        "n_noise": n_noise,
        "noise_fraction": round(n_noise / len(data.X), 4),
        "degenerate": bool(diagnosis["degenerate"]),
        "field_fraction_of_largest": round(field_share, 4),
        "macro_recall": round(float(scores["recall"].mean()), 4),
        "macro_precision": round(float(scores["precision"].mean()), 4),
        "homogeneity": round(float(separation["homogeneity"]), 4),
        "completeness": round(float(separation["completeness"]), 4),
        "v_measure": round(float(separation["v_measure"]), 4),
        "per_cluster": scores.sort_values("n_true", ascending=False).round(4),
    }


def comparison(flags: tuple[bool, ...] = NORMALIZE_ROWS) -> pd.DataFrame:
    """The headline table: one row per normalisation setting."""
    rows = []
    for flag in flags:
        result = run_once(flag)
        rows.append({
            key: value for key, value in result.items() if key != "per_cluster"
        })
    return pd.DataFrame(rows)


def solve() -> dict[str, object]:
    """Fit both ways and diagnose which one collapses."""
    results = {flag: run_once(flag) for flag in NORMALIZE_ROWS}
    collapsed = [flag for flag, r in results.items() if bool(r["degenerate"])]
    normalised = results[True]
    plain = results[False]
    recall_on = float(str(normalised["macro_recall"]))
    recall_off = float(str(plain["macro_recall"]))
    precision_on = float(str(normalised["macro_precision"]))
    precision_off = float(str(plain["macro_precision"]))
    return {
        "table": comparison(),
        "one_blob_setting": (
            "normalize_rows=False" if False in collapsed else "neither"
        ),
        "degenerate_settings": [f"normalize_rows={flag}" for flag in collapsed],
        "with_normalisation": {
            key: value for key, value in normalised.items() if key != "per_cluster"
        },
        "without_normalisation": {
            key: value for key, value in plain.items() if key != "per_cluster"
        },
        "per_cluster_normalised": normalised["per_cluster"],
        "per_cluster_unnormalised": plain["per_cluster"],
        "recall_cost_of_normalising": round(recall_on - recall_off, 4),
        "precision_gain_of_normalising": round(precision_on - precision_off, 4),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover (figure)
    """Largest-cluster fraction and cluster count, on and off."""
    import matplotlib.pyplot as plt

    table = comparison()
    labels = [f"normalize_rows={flag}" for flag in table["normalize_rows"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    axes[0].bar(labels, table["largest_fraction"], color=["#4c72b0", "#c44e52"])
    axes[0].axhline(0.6, color="crimson", ls="--", lw=1,
                    label="degeneracy threshold 0.6")
    axes[0].set_ylabel("largest cluster / n stars")
    axes[0].set_title("The 'one blob' test")
    axes[0].legend(frameon=False)
    axes[1].bar(labels, table["n_clusters"], color=["#4c72b0", "#c44e52"])
    axes[1].set_ylabel("clusters found")
    axes[1].set_title("Clusters recovered (25 are real)")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the measurement": (
        "HDBSCAN* "
        f"{cite('Campello:13')}, as implemented by the hdbscan library "
        f"{cite('McInnes:17')}, with the "
        "pipeline's own settings (min_cluster_size=5, "
        "min_samples=None, cluster_selection_epsilon=0.0) on the 25 000-star "
        "member+field matrix. With row normalisation: 30 clusters, largest "
        "2 745 stars = 11.0% of the sample, 76.0% of stars called noise, not "
        "degenerate. Without it: 2 clusters, largest 22 213 stars = 88.9% of "
        "the sample, 11.0% noise, flagged degenerate by "
        "cluster.stability.degeneracy()."
    ),
    "which setting is the 'one blob' failure": (
        "normalize_rows=False, unambiguously. 88.9% of all 25 000 stars land "
        "in a single cluster, and 96.96% of that cluster is field stars: it "
        "is the background, relabelled as a discovery. This is exactly the "
        "signature S 6.3 describes for a density threshold set too high for "
        "the background, reproduced here by HDBSCAN* rather than DBSCAN "
        f"({cite('Ester:96', bare=True)}), "
        "which makes the point that the hierarchy does not save you from a "
        "bad geometry. Every recall number computed against that blob is "
        "meaningless: NGC 6819 and NGC 6791 score recall 1.00 and M 67 0.93 "
        "purely because the blob swallowed them."
    ),
    "the precision arithmetic": (
        "Macro precision is 0.0399 without normalisation and 0.0410 with it: "
        "essentially identical, and both terrible. Macro recall is 0.771 "
        "without and 0.229 with. So on this sample, at these settings, row "
        "normalisation *costs* 0.54 in recall and buys 0.001 in precision. "
        "That is not the 0.03 -> 0.12 precision gain quoted for M 67 in S 2, "
        "and the difference is worth understanding: the S 2 number is a "
        "single cluster's precision under the full pipeline, this is a macro "
        "average over 25 clusters with HDBSCAN run directly on raw C-space "
        "and no embedding step at all. Both are true; they measure different "
        "things, and only the degeneracy flag distinguishes them honestly."
    ),
    "why the degeneracy flag is the number that matters": (
        "Precision and recall both lie here, in opposite directions. The "
        "unnormalised run looks *better* on recall (0.771 vs 0.229) and on "
        "homogeneity (0.098 vs 0.087) and v-measure (0.086 vs 0.041): every "
        "conventional score prefers the collapsed partition. Only "
        "largest_fraction exposes it: 0.889 against a 0.6 degeneracy "
        "threshold. This is S 9's central warning made concrete. A clusterer "
        "that gives up and returns one blob still produces finite, "
        "publishable-looking scores, and three of the four standard metrics "
        "reward it for doing so."
    ),
    "what row normalisation actually buys": (
        "It breaks the blob. That is the whole of it, and it is enough. "
        "Projecting each star onto the unit sphere makes Euclidean distance "
        "a monotone function of cosine distance, so the comparison becomes "
        "one of abundance *patterns* rather than overall metallicity scale, "
        "and the thin-disc locus stops being one connected overdensity. What "
        "it does not do is find the clusters: with normalisation on, the "
        "largest cluster is still 2 745 stars and still 99.02% field, and "
        "76% of the sample is noise. The failure mode changed from 'one "
        "blob' to 'thirty fragments of the field', which is better only in "
        "the sense that a non-degenerate wrong answer can be improved."
    ),
    "the chapter's own conclusion, confirmed": (
        "S 7.4 says the residual failure 'is not the hierarchy but the "
        "space: the field is a single dense blob that contains the clusters, "
        "and no choice of min_cluster_size separates an object from its own "
        "background'. Measured: with the best available normalisation, "
        "HDBSCAN* on raw 16-D C-space recovers 23% macro recall at 4% "
        "precision, against a 4% chance floor. That a direct clustering of "
        "abundance space fails to recover known clusters is not peculiar to "
        "this workbook: it is the same conclusion reached from APOGEE "
        "abundances and from open-cluster samples "
        f"{cite('GarciaDias:19', 'Casamiquela:21')}. Row normalisation is "
        "necessary (without it nothing works at all) and nowhere near "
        "sufficient. That is the motivation for S 10-12: change the geometry "
        "first, then let HDBSCAN* work as designed."
    ),
    "reproducibility note": (
        "HDBSCAN* is deterministic given the matrix, so these numbers need "
        "no seed averaging, but they *are* specific to the 25 000-star FAST "
        "sample (CLUSTER_MAX_STARS=25000). A different field cap changes the "
        "density of the background and therefore every number above. Quote "
        "the sample size with the result, always."
    ),
    "references": reference_list(
        "Campello:13", "McInnes:17", "Ester:96", "GarciaDias:19",
        "Casamiquela:21",
    ),
}
