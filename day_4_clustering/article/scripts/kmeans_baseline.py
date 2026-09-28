#!/usr/bin/env python3
"""K-means on the same member matrix the rest of the benchmark uses.

The workbook's K-means chapter quotes real numbers from this run rather than
a textbook example:

  * homogeneity / completeness / v-measure / accuracy for
    K = number of true clusters (25) and K = 2 (the "families" reading);
  * the SSE curve over K = 2..30, for the elbow;
  * the silhouette at the chosen K.

Writes results/kmeans_baseline.csv (one row per K) next to the other result
artifacts, so the number quoted in the text has a file behind it.

Usage:
    uv run python article/scripts/kmeans_baseline.py \
        --allstar data/astraAllStarASPCAP-0.6.0.fits.gz \
        --out article/results/kmeans_baseline.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cluster import config  # noqa: E402
from cluster.baseline import baseline_matrix, cluster_only, separation_scores  # noqa: E402
from cluster.clusters import CLUSTERS  # noqa: E402
from cluster.data import prepare  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--out", default="article/results/kmeans_baseline.csv")
    ap.add_argument("--min-members", type=int, default=5)
    args = ap.parse_args()

    settings = config.Settings()
    names = settings.resolve_cluster_names([c.name for c in CLUSTERS])
    clusters = [c for c in CLUSTERS if c.name in names]

    print(f"loading {args.allstar} …", flush=True)
    prepared = prepare(
        args.allstar, settings, clusters,
        seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
        seed_parallax_frac=config.SEED_PARALLAX_FRAC,
        seed_pm_tol=config.SEED_PM_TOL,
        seed_rv_tol=config.SEED_RV_TOL,
        n_refine_passes=config.N_REFINE_PASSES,
        refine_sigma=config.REFINE_SIGMA,
    )

    # exactly the member selection the other arms use (cluster-only, >=5 members)
    sub = cluster_only(prepared.df)
    counts = sub["cluster"].value_counts()
    keep = [c for c in counts.index if counts[c] >= args.min_members]
    sub = sub[sub["cluster"].isin(keep)]
    true_labels = sub["cluster"].to_numpy()
    X = baseline_matrix(sub, settings, False, elements=list(prepared.elements))
    n_true = len(set(true_labels))
    print(f"member matrix: {X.shape[0]} stars, {X.shape[1]} features, {n_true} clusters", flush=True)

    rows = []
    for k in [2, 3, 5, 10, n_true, 40]:
        km = KMeans(n_clusters=k, n_init=10, random_state=settings.random_state).fit(X)
        scores = separation_scores(true_labels, km.labels_)
        sil = float(silhouette_score(X, km.labels_)) if 1 < k < X.shape[0] else float("nan")
        n_groups = len(set(km.labels_))
        largest = float(np.bincount(km.labels_).max() / len(km.labels_))
        rows.append({"K": k, "n_groups": n_groups, "largest_frac": largest,
                     "sse": float(km.inertia_), "silhouette": sil, **scores})
        print(f"  K={k:3d}  homog={scores['homogeneity']:.3f} compl={scores['completeness']:.3f} "
              f"v={scores['v_measure']:.3f} acc={scores['accuracy']:.3f} "
              f"sse={km.inertia_:.1f} sil={sil:.3f} largest={largest:.2f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd  # noqa: PLC0415

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(f"n_stars={X.shape[0]} n_clusters={n_true} seed={settings.random_state} "
          f"normalize_rows={settings.normalize_rows} standardize={settings.standardize}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
