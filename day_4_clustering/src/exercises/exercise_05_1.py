"""Chapter 5, exercise 1 — the core distance as a density estimate.

    Derive the density estimate of \\S 5.2 from the volume of a ball
    containing k points. Then compute, for the DR19 abundance matrix, the
    spread of kappa_15 across members and field stars. Is the cluster signal
    visible in the core distance alone, before any clustering?

The derivation is three lines; the measurement is the point. \\S 5.2 claims
that in 16 dimensions a factor of two in density is a 4% change in kappa, and
this exercise cashes that claim out on the real matrix: the member and field
core-distance distributions overlap almost completely, so kappa alone is not a
membership test. That is the quantitative reason the workbook needs the
embeddings of \\S\\S 10-12 rather than a threshold on local density.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import member_field

#: Neighbour count used for the core distance — the pipeline's graph k
#: (\S 5.3: UMAP and EVoC both build 15-NN graphs).
K = 15

#: The workbook's C-space dimension.
DIMENSION = 16


def density_sensitivity(dimension: int = DIMENSION, factor: float = 2.0) -> float:
    """Fractional change in kappa produced by a ``factor`` change in density.

    From :math:`\\hat\\rho \\propto k / (n\\,\\kappa^d)`, holding :math:`k`
    and :math:`n` fixed gives :math:`\\kappa \\propto \\rho^{-1/d}`, so
    halving the density multiplies kappa by :math:`2^{1/d}`. Returns that
    multiplier minus one.
    """
    return float(factor ** (1.0 / dimension) - 1.0)


def core_distances(X: np.ndarray, k: int = K) -> np.ndarray:
    """Distance from every row of ``X`` to its ``k``-th nearest neighbour."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    return np.asarray(distances)[:, k]


def _spread(values: np.ndarray) -> dict[str, float]:
    """Median, mean, std and the 5-95 percentile range of one population."""
    return {
        "n": float(values.size),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


def solve(k: int = K) -> dict[str, object]:
    """Measure kappa_k for members and field on the real DR19 matrix."""
    from sklearn.metrics import roc_auc_score

    data = member_field()
    kappa = core_distances(data.X, k=k)
    is_member = data.is_member

    member_kappa = kappa[is_member]
    field_kappa = kappa[~is_member]

    # Can kappa alone rank members ahead of field stars? A small kappa is the
    # membership hypothesis, hence the sign.
    auc = float(roc_auc_score(is_member.astype(int), -kappa))

    per_cluster = (
        pd.DataFrame({"cluster": data.labels, "kappa": kappa})
        .loc[is_member]
        .groupby("cluster")["kappa"]
        .agg(["size", "median"])
        .sort_values("size", ascending=False)
        .round(4)
    )
    field_median = float(np.median(field_kappa))
    denser_than_field = int((per_cluster["median"] < field_median).sum())

    # Precision of the obvious naive rule: call the N smallest kappa members.
    order = np.argsort(kappa)
    top = order[: int(is_member.sum())]

    return {
        "k": k,
        "dimension": int(data.X.shape[1]),
        "member": _spread(member_kappa),
        "field": _spread(field_kappa),
        "median_ratio_member_over_field": round(
            float(np.median(member_kappa) / field_median), 4,
        ),
        "implied_density_ratio": round(
            float((field_median / np.median(member_kappa)) ** data.X.shape[1]), 3,
        ),
        "auc_member_from_small_kappa": round(auc, 4),
        "per_cluster_median": per_cluster,
        "clusters_denser_than_field": denser_than_field,
        "n_clusters": int(len(per_cluster)),
        "precision_of_smallest_kappa_rule": round(float(is_member[top].mean()), 4),
        "member_base_rate": round(float(is_member.mean()), 4),
        "kappa_change_for_2x_density": round(density_sensitivity(), 4),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Overlaid kappa histograms for members and field."""
    import matplotlib.pyplot as plt

    data = member_field()
    kappa = core_distances(data.X)
    is_member = data.is_member

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bins = np.linspace(0.0, float(np.percentile(kappa, 99.5)), 60).tolist()
    ax.hist(kappa[~is_member], bins=bins, density=True, color="#999999",
            alpha=0.7, label=f"field (n={int((~is_member).sum())})")
    ax.hist(kappa[is_member], bins=bins, density=True, histtype="step",
            color="crimson", lw=2, label=f"members (n={int(is_member.sum())})")
    ax.axvline(float(np.median(kappa[~is_member])), color="#555555", ls=":", lw=1)
    ax.axvline(float(np.median(kappa[is_member])), color="crimson", ls=":", lw=1)
    ax.set_xlabel(rf"core distance $\kappa_{{{K}}}$")
    ax.set_ylabel("density")
    ax.set_title("Cluster members are not measurably denser in 16-D C-space")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the derivation": (
        "A ball of radius kappa_k around x contains, by construction, exactly "
        "k of the n points. Its volume in d dimensions is V = c_d kappa_k^d "
        "with c_d = pi^(d/2)/Gamma(d/2+1) a constant that does not depend on "
        "the data. The fraction of the sample inside it is k/n, so the mean "
        "density over the ball is (k/n)/V, giving rho-hat(x) = k / (n c_d "
        "kappa_k(x)^d), i.e. rho-hat proportional to k/(n kappa_k^d) — "
        "equation 5.1. Inverting it, kappa is proportional to rho^(-1/d): the "
        "distance is a density estimate, and the exponent is the dimension."
    ),
    "why d ruins the leverage": (
        "kappa ~ rho^(-1/d) means the *sensitivity* of the measured distance "
        "to the underlying density falls as 1/d. In d=16, doubling the density "
        "shrinks kappa by only 2^(1/16) - 1 = 4.4% — the figure quoted in "
        "S 5.2, confirmed here by density_sensitivity(). In 2-D the same "
        "density contrast would move kappa by 41%. So a density difference "
        "that would be glaringly obvious on a 2-D scatter plot is, in C-space, "
        "a few-percent shift buried inside a distribution whose own spread is "
        "tens of percent."
    ),
    "what the real data show": (
        "Measured on the 25 000-star APOGEE DR19 member+field matrix "
        f"({cite('Majewski:17', 'Almeida:23', bare=True)}) with k=15 "
        "(solve() reproduces it): members have median kappa_15 = 0.484, field "
        "stars 0.506 — members are denser, but by 4.4%, which is exactly one "
        "'factor of two in density' worth of signal and nothing more. The "
        "distributions overlap almost entirely: the member spread (p05-p95 = "
        "0.302-0.746) sits inside the field spread (0.227-0.738), and 44.8% of "
        "field stars are below the member median. Only 0.4% of members fall "
        "below the field's 5th percentile."
    ),
    "is the signal visible in kappa alone": (
        "No. The ROC AUC for separating members from field using kappa_15 "
        "alone is 0.509 — chance is 0.500. Take the 1 002 stars with the "
        "smallest core distance, exactly as many as there are members, and "
        "0.2% of them are members, *below* the 4.0% base rate: the rule is "
        "worse than picking at random, because the very densest parts of the "
        "16-D matrix are field stars in the crowded thin-disc locus, not "
        "clusters. Per cluster the picture is only slightly better — 16 of the "
        "25 have a median kappa below the field median, so nine are literally "
        "sparser than their own background. M 15 (0.670), M 92 (0.638) and "
        "Berkeley 17 (0.652) are the sparsest; M 71 (0.313) and M 107 (0.320) "
        "the densest."
    ),
    "why this is the chapter's real lesson": (
        "The core distance is the substrate of everything that follows — "
        f"DBSCAN's epsilon test (S 6, {cite('Ester:96', bare=True)}), "
        "HDBSCAN's mutual reachability "
        f"(S 7, {cite('Campello:13', bare=True)}), "
        f"the kNN graphs inside UMAP {cite('McInnes:18')} and EVoC "
        "(S 11-12). This measurement says "
        "that in raw C-space that substrate carries almost no membership "
        "information on its own. Density-based clustering can still work, but "
        "only because it uses *connectivity* between neighbours rather than "
        "the density value at a point, and because the embeddings first change "
        "the geometry. Anyone proposing to find clusters by thresholding local "
        "density in 16-D abundances should run this two-line check first."
    ),
    "the caveat": (
        "kappa is measured here on the row-normalised, standardised matrix "
        "(the pipeline default of S 2), so it is an angular density on the "
        "unit sphere rather than a density in raw abundance units. The "
        "conclusion is not sensitive to that choice — the overlap is far too "
        "large — but the absolute numbers are, and a kappa quoted without the "
        "normalisation recipe is meaningless. Note also that the field here is "
        "the 25 000-star sample, not all 357 056 field stars; a larger field "
        "would shrink every kappa (more points, same volume) without changing "
        "the member/field ratio much."
    ),
    "references": reference_list(
        "Majewski:17", "Almeida:23", "Ester:96", "Campello:13", "McInnes:18",
    ),
}
