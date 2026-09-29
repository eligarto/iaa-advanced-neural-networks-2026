"""Chapter 1, exercise 1 — why old metal-poor clusters should be easier.

    A cluster of 10^4 stars dissolves over a few hundred Myr; a globular
    cluster of 10^6 stars survives for more than 10 Gyr. Explain from that
    why chemical tagging should be *easier* for old metal-poor clusters than
    for young solar-metallicity ones, then check your answer against the
    per-cluster results in Section 13 (the benchmark).

This is the workbook's first lesson in checking a physical argument against
the measurement. The argument from \\S 1.1 is sound and the data only half
agrees with it: globulars do score above open clusters on average, but the
ordering inside the sample is driven by *sample size*, not by metallicity,
and the single best-tagged object is the youngest, most solar-metallicity
cluster in the list. :func:`solve` recomputes the per-cluster table so the
student can see the argument survive in the mean and fail in the detail.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import knn_purity_raw, members

#: Metallicity below which a cluster counts as "metal-poor" for the split.
METAL_POOR_FEH = -1.0

#: Neighbourhood size for the kNN-purity measure (the workbook's default).
K_NEIGHBOURS = 15


def per_cluster_table(k: int = K_NEIGHBOURS) -> pd.DataFrame:
    """Per-cluster kNN purity in the raw 16-D C-space, with kind and [Fe/H].

    kNN purity is the parameter-free half of the workbook's metric pair: for
    each member, the fraction of its ``k`` nearest neighbours that belong to
    the same cluster, averaged over the cluster. It needs no embedding and no
    seed, so it isolates the *chemistry* from the projector.
    """
    from cluster.clusters import CLUSTER_BY_NAME

    data = members()
    purity = knn_purity_raw(data.X, data.labels, k=k)
    frame = data.df

    rows: list[dict[str, object]] = []
    for name, value in purity.items():
        sub = frame[frame["cluster"] == name]
        rows.append({
            "cluster": str(name),
            "kind": CLUSTER_BY_NAME[str(name)].kind,
            "n_members": int(len(sub)),
            "feh": round(
                float(np.nanmedian(sub["FE_H"].to_numpy(dtype=float))), 3,
            ),
            "knn_purity": round(float(value), 4),
        })
    table = pd.DataFrame(rows).sort_values("knn_purity", ascending=False)
    return table.reset_index(drop=True)


def solve(k: int = K_NEIGHBOURS) -> dict[str, object]:
    """Score the physical argument against the per-cluster measurement."""
    from scipy.stats import spearmanr

    table = per_cluster_table(k=k)

    by_kind = table.groupby("kind")["knn_purity"].agg(
        ["mean", "std", "count", "min", "max"],
    ).round(4)

    poor = table[table["feh"] < METAL_POOR_FEH]
    rich = table[table["feh"] >= METAL_POOR_FEH]

    rho_feh, p_feh = spearmanr(table["feh"], table["knn_purity"])
    rho_n, p_n = spearmanr(table["n_members"], table["knn_purity"])

    return {
        "per_cluster": table,
        "by_kind": by_kind,
        "globular_mean": round(
            float(table.loc[table["kind"] == "globular", "knn_purity"].mean()),
            4,
        ),
        "open_mean": round(
            float(table.loc[table["kind"] == "open", "knn_purity"].mean()), 4,
        ),
        "metal_poor": sorted(str(c) for c in poor["cluster"]),
        "metal_poor_mean": round(float(poor["knn_purity"].mean()), 4),
        "metal_rich_mean": round(float(rich["knn_purity"].mean()), 4),
        "spearman_feh": (round(float(rho_feh), 3), round(float(p_feh), 4)),
        "spearman_n_members": (round(float(rho_n), 3), round(float(p_n), 4)),
        "best": str(table["cluster"].iloc[0]),
        "best_purity": float(table["knn_purity"].iloc[0]),
        "chance_floor": round(1.0 / len(table), 4),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """kNN purity against metallicity, coloured by cluster kind."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    table = result["per_cluster"]
    assert isinstance(table, pd.DataFrame)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for kind, colour in (("globular", "#c44e52"), ("open", "#4c72b0")):
        sub = table[table["kind"] == kind]
        ax.scatter(
            sub["feh"], sub["knn_purity"],
            s=20 + 1.4 * sub["n_members"], color=colour, alpha=0.75,
            edgecolor="k", linewidth=0.4, label=f"{kind} (size = n members)",
        )
    ax.axhline(
        cast(float, result["chance_floor"]), color="grey", ls="--", lw=1,
        label="chance floor 1/25",
    )
    ax.axvline(METAL_POOR_FEH, color="grey", ls=":", lw=1)
    for _, row in table.iterrows():
        if row["knn_purity"] > 0.6 or row["n_members"] > 100:
            ax.annotate(
                str(row["cluster"]), (row["feh"], row["knn_purity"]),
                fontsize=7, xytext=(4, 3), textcoords="offset points",
            )
    ax.set_xlabel("median [Fe/H]")
    ax.set_ylabel(f"kNN purity (k={K_NEIGHBOURS}, raw 16-D C-space)")
    ax.set_title("Is metal-poor really easier?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the physical argument": (
        "Chemical tagging rests on chemistry being the only surviving "
        "record of a dispersed birth cluster "
        f"{cite('Freeman:02', 'BlandHawthorn:16')}; from there three "
        "separate effects all point the same way. (1) Mass and "
        "survival: a 10^6-solar-mass globular is bound tightly enough to "
        "survive >10 Gyr, so its members are still together and still "
        "labelled, whereas a 10^4-star open cluster is gone in a few hundred "
        "Myr and its siblings are scattered over kiloparsecs. (2) Chemical "
        "contrast: metal-poor stars formed from gas that had been enriched "
        "by few generations, so the abundance pattern is further from the "
        "crowded solar-metallicity locus where most field stars live — a "
        "cluster at [Fe/H] = -2 has almost no field background to hide in, "
        "while a cluster at [Fe/H] = 0 sits in the middle of the disc "
        "distribution. (3) Internal spread: section 1.1 notes that efficient "
        "ISM mixing limits how *different* two clusters can be "
        f"{cite('Kreckel:20')} — 0.02-0.03 dex of scatter in nearby spiral "
        "discs, correlated below 600 pc — and the "
        "clusters most alike are the coeval solar-metallicity open ones. So "
        "the prediction is: globular and metal-poor should tag better."
    ),
    "what the data actually says": (
        "Half right, and the half that is wrong is more interesting. "
        "Measured as kNN purity (k=15) in the raw 16-D C-space on the 1 002 "
        "member rows, the seven globulars average 0.451 and the eighteen "
        "open clusters average 0.331 — the predicted direction, and both far "
        "above the 0.04 chance floor of 25 equal clusters. Splitting on "
        "metallicity instead of kind gives 0.433 for the five clusters below "
        "[Fe/H] = -1 (M 3, M 5, M 13, M 15, M 92) against 0.348 for the "
        "other twenty. But the effect does not survive as a trend: "
        "Spearman's rho between median [Fe/H] and purity is +0.167 "
        "(p = 0.43) — not significant, and *positive*, i.e. pointing the "
        "wrong way. The metallicity story survives as a group mean and "
        "dissolves as a correlation."
    ),
    "the confound that actually drives the table": (
        "Sample size. Spearman's rho between the number of member rows and "
        "kNN purity is +0.758 (p < 0.0001) — far stronger than anything "
        "metallicity does. A cluster with 230 rows has 230 chances to be "
        "somebody's neighbour; a cluster with 6 rows (NGC 2158) scores "
        "exactly 0.000 because five neighbours cannot outvote fifteen. "
        "The globulars in this sample are also the populous ones (M 3 has "
        "154 rows, M 5 has 67), so 'globular' and 'well-sampled' are "
        "confounded and the kind-split cannot separate them. This is the "
        "same lesson as chapter 2's M 67 ablation: a macro average over a "
        "skewed population reports the population as much as the method."
    ),
    "the counterexample that breaks the story": (
        "The single best-tagged cluster in the sample is the Pleiades, at "
        "kNN purity 0.864 — the *youngest* (~100 Myr) and most "
        "solar-metallicity ([Fe/H] = +0.02) object on the list, and only 23 "
        "rows, so it is not a size artefact either. Meanwhile the two most "
        "metal-poor clusters, M 15 ([Fe/H] = -2.26, purity 0.271) and M 92 "
        "(-2.19, 0.293), sit in the bottom half. Whatever makes the Pleiades "
        "easy is not age or metallicity: it is a distinctive abundance "
        "pattern (young, unevolved, chemically homogeneous, and nearby so "
        "the stars are bright and the abundances precise). It is also the "
        "cluster "
        f"{cite('Kos:17', parenthetical=False)} recovered with t-SNE from "
        "GALAH abundances, so its distinctiveness is not peculiar to this "
        "catalogue. This is exactly "
        "the caveat section 1.3 attaches to every negative result in the "
        "field — the scope is *these abundances, these clusters*."
    ),
    "how to state the answer honestly": (
        "The argument is physically correct and the data is consistent with "
        "it in the mean (globular 0.451 > open 0.331), but the mean is not "
        "evidence for the mechanism, because the strongest predictor in the "
        "table is member count, not metallicity or age. The defensible claim "
        "is: 'old metal-poor globulars score higher on average in our "
        "sample, but the dominant term is sampling and the trend with "
        "[Fe/H] is not significant (rho = 0.17, p = 0.43)'. To test the "
        "mechanism properly you would need to hold member count fixed and "
        "vary metallicity — which this sample of 25 objects cannot do."
    ),
    "the other half of the story": (
        "kNN purity measures separation among *members only*. The field "
        "retrieval task in section 13 is harder and orders the clusters "
        "differently: from results/dr19_field_retrieval.txt, EVoC's "
        "per-cluster recall over 358 058 stars puts NGC 1245 at 0.76 and the "
        "Pleiades at 0.74 at the top, with M 15 at 0.16 and M 13 at 0.15 "
        "near the bottom — so against a real field background the "
        "metal-poor globulars do *worse*, not better. Whether metal-poor is "
        "easier therefore depends on which question you asked, which is the "
        "whole reason section 13 reports the two tasks separately."
    ),
    "references": reference_list(
        "Freeman:02", "BlandHawthorn:16", "Kreckel:20", "Kos:17",
    ),
}
