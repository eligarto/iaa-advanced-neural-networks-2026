"""Chapter 15, exercise 4 — who are the 22 stars?

    Table 12 removes 22 stars from the kinematic list and improves the age.
    Inspect those 22 stars: what are their abundances, their kinematics and
    their position relative to M 67? Write the three-sentence case for and
    against calling them contaminants.

The 22 stars are identified here by running the workbook's own two-stage
pipeline on M 67 (``scripts/two_stage_pipeline.py``: kinematic candidates, then
rejection of latent-space outliers at median + 2.5 x 1.4826 MAD) and taking the
difference between the stage-1 list and the stage-2 list — 259 candidates, 22
rejected, 237 kept, which is the chapter's own 259 -> 22 -> 237. The module
then characterises the 22 against the 237 on the quantities the chapter's
claims rest on: abundances (all 16), kinematics (parallax, proper motions,
radial velocity), and position (angular separation from the cluster centre).

The answer is written as the exercise asks — the case for and the case against,
three sentences each — but every adjective in it is a measured number, because
"these stars look like field stars" is exactly the kind of claim this workbook
exists to make someone prove.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import DataNotAvailable, catalogue_path, settings

#: The cluster, and the rejection threshold from ``scripts/two_stage_pipeline.py``.
TARGET_CLUSTER = "M 67"
DEFAULT_K = 2.5
#: The published numbers this is checked against (Table 12).
PUBLISHED = {"stage1": 259, "removed": 22, "stage2": 237}
#: Kinematic columns the stage-1 embedding uses, as in the two-stage script.
KIN_COLS = ("GAIAEDR3_PARALLAX", "GAIAEDR3_PMRA", "GAIAEDR3_PMDEC", "VHELIO_AVG")
#: Abundance columns of the 16 elements.
ELEMENTS = (
    "C_FE", "N_FE", "O_FE", "NA_FE", "MG_FE", "AL_FE", "SI_FE", "S_FE",
    "K_FE", "CA_FE", "TI_FE", "V_FE", "CR_FE", "MN_FE", "NI_FE", "FE_H",
)


def _standardise(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> np.ndarray:
    """Column-standardised matrix with the median filling the missing entries."""
    values = frame[list(columns)].to_numpy(dtype=float)
    median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    values = np.where(np.isfinite(values), values, median[np.newaxis, :])
    scale = np.nanstd(values, axis=0)
    scale[scale == 0] = 1.0
    return (values - median) / scale


def stage_lists(k: float = DEFAULT_K) -> dict[str, object]:
    """The stage-1 candidates, the stage-2 survivors, and the 22 removed.

    Follows ``scripts/two_stage_pipeline.py`` exactly: kinematics-only t-SNE
    over the cluster's region (with the field stars present, because the
    rejection is a *comparison* against the field), HDBSCAN, the best-overlap
    group as the candidate list, then the robust latent-outlier rejection.
    """
    import hdbscan

    from cluster import config, spectral
    from cluster.benchmark import fit_tsne
    from cluster.clusters import CLUSTER_BY_NAME
    from cluster.data import apply_quality_cuts, load_allstar
    from cluster.membership import angular_separation, kinematic_members
    from exercises.utils import embedding_path

    cluster = CLUSTER_BY_NAME[TARGET_CLUSTER]
    cfg = settings()
    cfg.require_aspcap_flag_clean = False

    frame = load_allstar(catalogue_path(), cfg.elements)
    frame = apply_quality_cuts(frame, cfg)
    separation = angular_separation(
        frame["RA"].to_numpy(dtype=float), frame["DEC"].to_numpy(dtype=float),
        cluster.ra_deg, cluster.dec_deg,
    )
    region = pd.DataFrame(frame[separation <= cluster.region_deg]).copy()
    kinematic = kinematic_members(
        region, cluster,
        config.SEED_POSITION_RADIUS_DEG, config.SEED_PARALLAX_FRAC,
        config.SEED_PM_TOL, config.SEED_RV_TOL,
        config.N_REFINE_PASSES, config.REFINE_SIGMA,
    ).to_numpy()
    region["cluster"] = np.where(kinematic, TARGET_CLUSTER, "field")

    try:
        embedding = spectral.load_embedding_frame(embedding_path("attention_broad_merged.parquet"))
    except DataNotAvailable:
        embedding = spectral.load_embedding_frame(embedding_path("masked_latent.parquet"))
    embedded_columns = spectral.embedding_columns(embedding)
    merged = spectral.align_embeddings(
        region.drop_duplicates(subset=["APOGEE_ID"]), embedding,
    )
    if len(merged) < 50:
        raise DataNotAvailable(
            "fewer than 50 stars with both catalogue rows and latents for "
            f"{TARGET_CLUSTER}; the rejection step needs the field present",
        )

    true = merged["cluster"].to_numpy()
    kinematic_mask = true == TARGET_CLUSTER

    predicted = hdbscan.HDBSCAN(**cfg.hdbscan).fit_predict(
        fit_tsne(_standardise(merged, KIN_COLS), cfg.tsne, cfg.random_state),
    )
    best_group, best_overlap = -1, 0
    for group in np.unique(predicted):
        if group < 0:
            continue
        overlap = int(((predicted == group) & kinematic_mask).sum())
        if overlap > best_overlap:
            best_group, best_overlap = int(group), overlap
    if best_group < 0:
        raise DataNotAvailable("stage 1 found no candidate group for M 67")
    candidates = predicted == best_group

    spectral_space = _standardise(merged, embedded_columns)
    centroid = np.nanmedian(spectral_space[candidates], axis=0)
    distance = np.linalg.norm(spectral_space[candidates] - centroid, axis=1)
    median = float(np.nanmedian(distance))
    mad = float(np.nanmedian(np.abs(distance - median)))
    keep = distance <= median + k * 1.4826 * mad

    kept = np.zeros(len(merged), dtype=bool)
    kept[candidates] = keep
    removed = candidates & ~kept

    return {
        "cluster": TARGET_CLUSTER,
        "k": k,
        "merged": merged,
        "angular_separation": separation[
            np.isin(region["APOGEE_ID"].to_numpy(), merged["APOGEE_ID"].to_numpy())
        ] if len(merged) == len(region) else None,
        "candidates": candidates,
        "kept": kept,
        "removed": removed,
        "kinematic_flag": kinematic_mask,
        "counts": {
            "candidates": int(candidates.sum()),
            "true_members_among_candidates": int((candidates & kinematic_mask).sum()),
            "field_among_candidates": int((candidates & ~kinematic_mask).sum()),
            "removed": int(removed.sum()),
            "removed_that_were_kinematic_members": int((removed & kinematic_mask).sum()),
            "removed_that_were_field_labelled": int((removed & ~kinematic_mask).sum()),
            "kept": int(kept.sum()),
            "kept_that_were_field_labelled": int((kept & ~kinematic_mask).sum()),
            "published": PUBLISHED,
        },
        "threshold": {"median": round(median, 4), "mad": round(mad, 4),
                      "cut": round(median + k * 1.4826 * mad, 4)},
    }


def characterise(result: dict[str, object] | None = None) -> pd.DataFrame:
    """Median properties of the 22 removed stars against the 237 kept.

    The 237 kept rows include the field contaminants that the kinematic stage
    accepted, which is deliberate: the chapter's claim is that the *removed*
    stars look like field stars, and the honest comparison is against both the
    kept candidates and, where the columns allow, the field population.
    """
    result = result if result is not None else stage_lists()
    merged = result["merged"]
    assert isinstance(merged, pd.DataFrame)
    removed = np.asarray(result["removed"], dtype=bool)
    kept = np.asarray(result["kept"], dtype=bool)

    columns = [c for c in ELEMENTS if c in merged.columns] + [
        c for c in KIN_COLS + ("TEFF", "LOGG", "SNR") if c in merged.columns
    ]
    rows = []
    for column in columns:
        values = merged[column].to_numpy(dtype=float)
        rows.append({
            "property": column,
            "removed_median": round(float(np.nanmedian(values[removed])), 4),
            "kept_median": round(float(np.nanmedian(values[kept])), 4),
            "n_removed": int(np.isfinite(values[removed]).sum()),
            "n_kept": int(np.isfinite(values[kept]).sum()),
        })
    table = pd.DataFrame(rows)
    table["difference"] = (table["removed_median"] - table["kept_median"]).round(4)
    return table


def separation_from_centre(result: dict[str, object] | None = None) -> pd.DataFrame:
    """Angular separation from the cluster centre, removed versus kept."""
    from cluster.clusters import CLUSTER_BY_NAME
    from cluster.membership import angular_separation

    result = result if result is not None else stage_lists()
    merged = result["merged"]
    assert isinstance(merged, pd.DataFrame)
    cluster = CLUSTER_BY_NAME[TARGET_CLUSTER]
    separation = angular_separation(
        merged["RA"].to_numpy(dtype=float), merged["DEC"].to_numpy(dtype=float),
        cluster.ra_deg, cluster.dec_deg,
    )
    removed = np.asarray(result["removed"], dtype=bool)
    kept = np.asarray(result["kept"], dtype=bool)
    return pd.DataFrame([
        {"group": "removed (22)", "n": int(removed.sum()),
         "median_sep_deg": round(float(np.median(separation[removed])), 4),
         "p90_sep_deg": round(float(np.percentile(separation[removed], 90)), 4),
         "within_1_deg": int((separation[removed] <= 1.0).sum())},
        {"group": "kept (237)", "n": int(kept.sum()),
         "median_sep_deg": round(float(np.median(separation[kept])), 4),
         "p90_sep_deg": round(float(np.percentile(separation[kept], 90)), 4),
         "within_1_deg": int((separation[kept] <= 1.0).sum())},
        {"group": "region limit", "n": 0,
         "median_sep_deg": round(float(cluster.region_deg), 4),
         "p90_sep_deg": float("nan"), "within_1_deg": 0},
    ])


def removal_recoverability(result: dict[str, object] | None = None, seed: int = 0) -> dict[str, object]:
    """Can the removal flag be predicted from abundances alone? From kinematics?

    Cross-validated logistic regression on each block of features separately.
    An AUC near 0.5 means the rejections are not explained by the block at all;
    high AUC means the stage-2 rule is effectively a threshold on that block —
    which matters, because a rejection rule that is really a metallicity cut
    would be removing chemistry, not contamination.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    result = result if result is not None else stage_lists()
    merged = result["merged"]
    assert isinstance(merged, pd.DataFrame)
    candidates = np.asarray(result["candidates"], dtype=bool)
    removed = np.asarray(result["removed"], dtype=bool)
    target = removed[candidates].astype(int)

    blocks = {
        "abundances (16-d)": [c for c in ELEMENTS if c in merged.columns],
        "kinematics (4-d)": [c for c in KIN_COLS if c in merged.columns],
        "atmospheric (Teff, logg, SNR)": [c for c in ("TEFF", "LOGG", "SNR") if c in merged.columns],
    }
    out: dict[str, object] = {"n_candidates": int(candidates.sum()),
                              "n_removed": int(target.sum())}
    minority = int(min(np.bincount(target)))
    folds = max(2, min(5, minority))
    for tag, columns in blocks.items():
        if not columns:
            continue
        matrix = _standardise(merged, columns)[candidates]
        if len(np.unique(target)) < 2:
            out[tag] = float("nan")
            continue
        scores = cross_val_score(
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            matrix, target,
            cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
            scoring="roc_auc",
        )
        out[tag] = round(float(scores.mean()), 3)
    return out


def solve() -> dict[str, object]:
    """The counts, the characterisation, and the recoverability of the flag."""
    result = stage_lists()
    return {
        "counts": result["counts"],
        "threshold": result["threshold"],
        "properties": characterise(result),
        "separation": separation_from_centre(result),
        "recoverability": removal_recoverability(result),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Separation from the cluster centre and the [Fe/H] distributions."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    properties = result["properties"]
    assert isinstance(properties, pd.DataFrame)

    feh = properties[properties["property"] == "FE_H"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    if len(feh):
        row = feh.iloc[0]
        axes[0].bar(["removed (22)", "kept (237)"],
                    [float(row["removed_median"]), float(row["kept_median"])],
                    color=["#c44e52", "#4c72b0"])
        axes[0].set_ylabel("median [Fe/H]")
        axes[0].set_title("Metallicity of the rejected stars", fontsize=10)
    separation = result["separation"]
    assert isinstance(separation, pd.DataFrame)
    subset = separation[separation["group"] != "region limit"]
    axes[1].bar(subset["group"], subset["median_sep_deg"], color=["#c44e52", "#4c72b0"])
    axes[1].set_ylabel("median separation from centre (deg)")
    axes[1].set_title("Where the rejected stars sit", fontsize=10)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the lists, reproduced": (
        "Running the workbook's own two-stage pipeline on M 67 reproduces the "
        "chapter's list changes exactly: stage 1 (kinematic candidates inside "
        "the cluster's 4.17-degree region, with the field present) returns 259 "
        "stars, the latent-outlier rejection removes 22, and 237 survive. The "
        "candidate list is 219 stars that the kinematic selection labels as "
        "members plus 40 region stars it labels as field; the rejection removes "
        "15 of those 219 member-labelled stars and 7 of the 40 field-labelled "
        "ones, and leaves 33 field-labelled stars in the kept list. The "
        "rejection cut sits at latent distance 15.98 — median 9.70 plus 2.5 x "
        "1.4826 x MAD(1.69) — so the 22 stars are between roughly 3.7 and 12 "
        "robust standard deviations from the candidate centroid."
    ),
    "their abundances": (
        "This is where the measurement contradicts the expectation. The 22 "
        "removed stars are not chemically indistinguishable: their median "
        "[Fe/H] is -0.092 against +0.005 for the 237 kept, a 0.10 dex offset "
        "from a cluster whose own literature value is +0.03, and the deficit is "
        "carried mainly by nitrogen (N_FE -0.094 against +0.080, a 0.17 dex "
        "gap), chromium (-0.060), vanadium (-0.050), titanium (-0.048) and "
        "oxygen (-0.035), against a potassium excess (+0.054). A logistic probe "
        "on the 16 abundances alone recovers the rejection flag at a "
        "cross-validated AUC of 0.66, i.e. the chemical difference is real and "
        "not one noisy element. It is also small: 0.1 dex is a factor of 1.25 "
        "in abundance, and M 67's own members span more than that."
    ),
    "their kinematics": (
        "Kinematics do not separate them at all, which is the strongest single "
        f"fact in the exercise. On the Gaia astrometry {cite('Gaia:23')}, "
        "medians, removed against kept: parallax 1.141 "
        "against 1.163 mas, PM_RA -11.084 against -10.990 mas/yr, PM_DEC "
        "-2.912 against -2.890 mas/yr, V_helio 34.22 against 34.29 km/s — "
        "every offset is a few per cent of the quantity itself and smaller "
        "than the typical uncertainty of a single star. A logistic probe on the "
        "four kinematic columns returns AUC 0.473, i.e. indistinguishable from "
        "chance. So the 22 stars were accepted by the membership criterion that "
        "this workbook treats as the recall stage, and are being rejected by a "
        "rule that does not use those columns at all."
    ),
    "their position and their spectra": (
        "The removed stars sit slightly farther from the cluster centre: median "
        "angular separation 0.284 degrees against 0.218 for the kept stars, and "
        "a 90th percentile of 1.541 degrees against 0.530 — so the removal is "
        "concentrated in the outer tail of the region, though 18 of the 22 are "
        "still within 1 degree of the centre. They are also cooler and noisier: "
        "median Teff 4750 K against 5105 K, logg 3.94 against 4.00, SNR 156 "
        "against 168. A probe on those three atmospheric columns recovers the "
        "flag at AUC 0.671, higher than the abundances manage — which is the "
        "first hint that what stage 2 is really selecting on is 'cool, "
        "faint, and therefore spectrally atypical', rather than 'not a member'."
    ),
    "the case for calling them contaminants (three sentences)": (
        "They are chemically distinct from the cluster in the direction that "
        "matters — median [Fe/H] -0.09 against +0.00, with nitrogen 0.17 dex "
        "lower and four more elements 0.03-0.06 dex lower — and the flag is 66% "
        "recoverable from those abundances alone, so this is a population "
        "difference rather than scatter — and a 0.1 dex offset is the scale on "
        f"which chemical tagging claims to work at all {cite('Freeman:02')}. "
        "They are cooler (4750 against 5105 K) "
        "at slightly lower SNR and sit systematically farther from the centre "
        "(90th percentile separation 1.54 degrees against 0.53), which is what "
        "a superposed field population looks like in colour, magnitude and "
        "radius simultaneously. And the decisive external evidence is that "
        "removing them improves the fitted age from 1.89 to 2.65 Gyr against a "
        f"literature 2.82 ({cite('Dias:02', bare=True)}) — a factor of five in "
        "residual — which would be a "
        "coincidence if they were ordinary members."
    ),
    "the case against (three sentences)": (
        "Their kinematics are indistinguishable from the kept members (parallax, "
        "both proper motions and radial velocity agree to a few per cent, probe "
        "AUC 0.473), so 15 of the 22 are stars the workbook's own membership "
        "criterion calls members, and the rejection overrides that criterion "
        "using a masked-autoencoder latent space "
        f"({cite('He:22', bare=True)}) that §14.5 showed carries a pipeline "
        "signature. "
        "The rule is not even a contaminant filter: it also leaves 33 "
        "field-labelled stars in the kept list, and the features that best "
        "predict rejection are atmospheric (Teff, logg, SNR at AUC 0.671), so "
        "what it selects is 'cool, faint, spectrally atypical' rather than 'not "
        "a member'. The chemistry is the honest middle: a 0.1 dex [Fe/H] offset "
        "is real but small, and without an independent membership assessment "
        "the defensible statement is that these are the 22 stars whose removal "
        "makes the fit agree with the literature — a statement about the fit, "
        "not yet about the stars."
    ),
    "what would settle it": (
        "Three tests, none of them run here. (1) Cross-match the 22 against "
        "Simbad's M 67 membership list — the workbook's §13 kinematic-versus-"
        "Simbad comparison exists exactly for this arbitration, and the 22 "
        "should be enriched in Simbad non-members if the contamination reading "
        "is right. (2) Re-fit the age with the 22 restored and weighted by Gaia "
        f"astrometric membership probabilities {cite('Gaia:23')}, testing "
        "whether the improvement "
        "survives a probabilistic membership. (3) Repeat the age fit over the "
        "seven-seed set: the published 1.89 -> 2.65 Gyr shift is a single-run "
        "number, and §16's own rubric asks for mean ± s.d. over at least three "
        "seeds before a difference is quoted."
    ),
    "the methodological note": (
        "The removal criterion is a robust outlier rule (median + 2.5 x 1.4826 "
        "x MAD) applied to a 256-dimensional latent, and the number of stars it "
        "rejects is therefore set by the threshold, not by the data: lower k "
        "removes more, higher k fewer, and nothing in the rule is calibrated to "
        "'contaminant'. That is why the exercise asks for the case for *and* "
        "against — a rejection rule with a tunable threshold can always be "
        "tuned to improve a downstream fit, and the improvement is not evidence "
        "for what the rule claims to be removing."
    ),
    "references": reference_list(
        "Gaia:23", "Dias:02", "Freeman:02", "He:22",
    ),
}
