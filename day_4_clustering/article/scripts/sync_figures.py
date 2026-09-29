#!/usr/bin/env python3
"""Vendor the presentation's figures into article/figures/.

Three kinds of output land here:

  copy       a PNG straight from the deck's asset directory
  filmstrip  N frames of an animated GIF, side by side, so the PDF keeps
             the sweep that the animation shows (labels printed under
             each panel in the LaTeX caption)
  qr         a QR code PNG for a URL the workbook points at

Provenance for every file is written to figures/MANIFEST.md so it is
always clear which figure came from where.

Usage:
    uv run python scripts/sync_figures.py \
        --deck-assets ../garciadias.github.io/public/presentations/iaa-so-chemical-tagging-2026 \
        --out figures
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------- the plan
# Every entry: output filename -> spec
#   ("copy", "<deck asset>")
#   ("filmstrip", "<deck gif>", [fractions of the animation to sample], panel width)
#   ("qr", "<url>")

DECK_URL = "https://garciadias.github.io/#/presentations/iaa-so-chemical-tagging-2026"
SCHOOL_URL = "https://www.granadacongresos.com/ai-ml"
REPO_URL = "https://github.com/garciadias/iaa-advanced-neural-networks-2026-draft"
LIBRARY_URL = "https://github.com/garciadias/iaa-advanced-neural-networks-2026-draft/tree/main/docs"

# Natural (declared) width, in points, that every vendored PNG is rewritten to
# carry in its pHYs chunk. Must stay below the class's \textwidth (366 pt).
TARGET_NATURAL_PT = 280.0

MANIFEST: dict[str, tuple] = {
    # --- front matter / introduction
    "qr_deck.png": ("qr", DECK_URL),
    "qr_school.png": ("qr", SCHOOL_URL),
    "qr_repo.png": ("qr", REPO_URL),
    "qr_library.png": ("qr", LIBRARY_URL),
    "montage25.png": ("copy", "sky_montage_25.png"),
    # --- the data
    "cspace.png": ("copy", "cspace_corner.png"),
    "sky_m67.png": ("copy", "sky_m67.png"),
    "sky_m3.png": ("copy", "sky_m3.png"),
    "sky_ngc188.png": ("copy", "sky_ngc188.png"),
    "sky_ic166.png": ("copy", "sky_ic166.png"),
    "sky_berkeley66.png": ("copy", "sky_berkeley66.png"),
    # --- K-means
    "kmeans_film.png": ("filmstrip", "kmeans.gif", [0.0, 0.33, 0.66, 1.0], 420),
    "kmeans_step1.png": ("copy", "kmeans_step1_init.png"),
    "kmeans_step2.png": ("copy", "kmeans_step2_assign.png"),
    "kmeans_step3.png": ("copy", "kmeans_step3_update.png"),
    "kmeans_step4.png": ("copy", "kmeans_step4_converged.png"),
    "kmeans_storyboard.png": ("copy", "kmeans_storyboard.png"),
    "kmeans_init.png": ("copy", "kmeans_init.png"),
    "kmeans_fail.png": ("copy", "kmeans_fail.png"),
    # --- KNN
    "knn_step1.png": ("copy", "knn_step1_distances.png"),
    "knn_step2.png": ("copy", "knn_step2_neighbourhood.png"),
    "knn_step3.png": ("copy", "knn_step3_core_distance.png"),
    "knn_step4.png": ("copy", "knn_step4_graph.png"),
    "knn_core_distance.png": ("copy", "knn_core_distance.png"),
    # --- DBSCAN
    "dbscan_film.png": ("filmstrip", "dbscan.gif", [0.0, 0.33, 0.66, 1.0], 420),
    "dbscan_step1.png": ("copy", "dbscan_step1_knobs.png"),
    "dbscan_step2.png": ("copy", "dbscan_step2_classify.png"),
    "dbscan_step3.png": ("copy", "dbscan_step3_grow.png"),
    "dbscan_step4.png": ("copy", "dbscan_step4_result.png"),
    # --- HDBSCAN*
    "mutual_reachability.png": ("copy", "mutual_reachability.png"),
    "hdbscan_film.png": ("filmstrip", "hdbscan_density.gif", [0.0, 0.33, 0.66, 1.0], 420),
    "hdbscan_tree.png": ("copy", "hdbscan_step3_tree.png"),
    "hdbscan_sweep.png": ("copy", "hdbscan_step4_sweep.png"),
    # --- PLSCAN
    "plscan_film.png": ("filmstrip", "plscan_persistence.gif", [0.0, 0.33, 0.66, 1.0], 420),
    "plscan_step1.png": ("copy", "plscan_step1_density.png"),
    "plscan_step2.png": ("copy", "plscan_step2_barcode.png"),
    "plscan_step3.png": ("copy", "plscan_step3_read.png"),
    "plscan_barcode.png": ("copy", "plscan_barcode.png"),
    # --- validation
    "proper_motions.png": ("copy", "proper_motions.png"),
    "confusion_tsne_chem.png": ("copy", "baseline_confusion_chem.png"),
    "confusion_tsne_kin.png": ("copy", "baseline_confusion_kin.png"),
    # --- t-SNE
    "tsne_film.png": ("filmstrip", "tsne.gif", [0.0, 0.33, 0.66, 1.0], 420),
    "tsne_step1.png": ("copy", "tsne_step1_highd.png"),
    "tsne_step2.png": ("copy", "tsne_step2_lowd.png"),
    "tsne_step3.png": ("copy", "tsne_step3_descent.png"),
    "kos_pleiades_tsne.png": ("copy", "kos_pleiades_tsne.png"),
    # --- UMAP
    "umap_film.png": ("filmstrip", "umap.gif", [0.0, 0.33, 0.66, 1.0], 420),
    "umap_step1.png": ("copy", "umap_step1_graph.png"),
    "umap_step2.png": ("copy", "umap_step2_fuzzy.png"),
    "umap_step3.png": ("copy", "umap_step3_layout.png"),
    # --- EVoC
    "evoc_pipeline.png": ("copy", "evoc_pipeline.png"),
    "evoc_step1.png": ("copy", "evoc_step1_graph.png"),
    "evoc_step2.png": ("copy", "evoc_step2_embed.png"),
    "evoc_step3.png": ("copy", "evoc_step3_cluster.png"),
    "evoc_step4.png": ("copy", "evoc_step4_persistence.png"),
    # --- the benchmark
    "benchmark_grid.png": ("copy", "benchmark_grid.png"),
    "headtohead_pca.png": ("copy", "headtohead_pca.png"),
    "paired_control.png": ("copy", "paired_control.png"),
    # --- spectral embeddings
    "mae_arch.png": ("copy", "mae_arch.png"),
    "mae_step1_mask.png": ("copy", "mae_step1_mask.png"),
    "mae_why_blocks.png": ("copy", "mae_why_blocks.png"),
    "mae_step2_encode.png": ("copy", "mae_step2_encode.png"),
    "mae_step3_recon.png": ("copy", "mae_step3_recon.png"),
    "mae_step4_latent.png": ("copy", "mae_step4_latent.png"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def filmstrip(
    src: Path, dst: Path, fractions: list[float], panel_w: int, tmp: Path, cols: int = 2
) -> str:
    """N frames of a GIF laid out in a grid (default 2 columns), with a rule.

    A single row of four panels renders each frame about 1.2 in wide inside the
    workbook's 5 in text column, which is too small to read. Two columns
    doubles the panel size at the cost of a taller figure.
    """
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(src) as im:
        n = getattr(im, "n_frames", 1)
        picks = [min(n - 1, max(0, round(f * (n - 1)))) for f in fractions]
        frames = []
        for idx in picks:
            im.seek(idx)
            frames.append(im.convert("RGB").copy())

    # all panels to the same height, then scale to the requested width
    h = min(f.height for f in frames)
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    frames = [f.resize((round(f.width * h / f.height), h), resample) for f in frames]

    # grid layout: `cols` panels per row, a little room under each for its label
    cols = max(1, min(cols, len(frames)))
    rows = (len(frames) + cols - 1) // cols
    pw = max(f.width for f in frames)
    label_h = 26
    gap = 8
    strip_w = cols * pw + (cols - 1) * gap
    strip = Image.new("RGB", (strip_w, rows * (h + label_h) + (rows - 1) * gap), "white")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:  # pragma: no cover - font always present on this host
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(strip)
    for k, (f, idx) in enumerate(zip(frames, picks)):
        r, c = divmod(k, cols)
        x = c * (pw + gap)
        y = r * (h + label_h + gap)
        strip.paste(f, (x, y))
        draw.text((x + 6, y + h + 4), f"frame {idx + 1}/{n}", fill=(60, 60, 60), font=font)
    # scale so a panel renders at panel_w px in the PDF
    scale = panel_w / pw
    strip = strip.resize((round(strip.width * scale), round(strip.height * scale)), resample)
    strip.save(dst, "PNG", optimize=True)
    return f"{len(frames)} frames of {n} ({', '.join(f'{f:.0%}' for f in fractions)})"


def make_qr(url: str, dst: Path) -> str:
    try:
        import segno  # noqa: PLC0415
    except ImportError:
        if dst.exists():
            return "kept (segno not importable here: run `uv run --with segno python scripts/sync_figures.py`)"
        raise SystemExit(
            "segno is needed to build the QR codes. Run:\n"
            "  uv run --with segno python scripts/sync_figures.py ...\n"
            "  (or keep the committed figures/qr_*.png)"
        )
    segno.make(url, error="m").save(dst, scale=6, border=2, dark="#111111")
    return url


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck-assets", default="", help="deck asset directory to vendor from")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    deck = Path(args.deck_assets) if args.deck_assets else None

    rows: list[tuple[str, str, str]] = []
    missing: list[str] = []

    for name, spec in MANIFEST.items():
        dst = out / name
        kind = spec[0]
        try:
            if kind == "copy":
                if deck is None or not (deck / spec[1]).exists():
                    missing.append(f"{name} <- {spec[1]}")
                    continue
                shutil.copy2(deck / spec[1], dst)
                note = spec[1]
            elif kind == "filmstrip":
                if deck is None or not (deck / spec[1]).exists():
                    missing.append(f"{name} <- {spec[1]}")
                    continue
                note = filmstrip(deck / spec[1], dst, spec[2], spec[3], out)
            elif kind == "qr":
                note = make_qr(spec[1], dst)
            else:  # pragma: no cover - manifest is static
                raise ValueError(f"unknown spec {spec!r}")
            rows.append((name, f"{kind}: {note}" if kind != "qr" else f"qr: {note}", sha256(dst)))
            if not args.quiet:
                print(f"  ok  {name:26s} {rows[-1][1][:70]}")
        except SystemExit:
            raise
        except Exception as exc:  # pragma: no cover
            print(f"  ERR {name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    # pdfLaTeX sizes a PNG by its declared physical resolution (the pHYs chunk),
    # and the ar-1col class then *left-shifts* any figure whose natural width
    # exceeds \textwidth (366 pt) before the \resizebox in workbook.tex gets a
    # chance to scale it. The deck's PNGs declare ~183 dpi, i.e. natural widths
    # of 370-780 pt, so half the figures were landing off the left edge of the
    # page. Rewriting the density to a fixed natural width keeps every inclusion
    # inside the column; the on-page size is still set by \resizebox.
    for i, (name, note, _) in enumerate(rows):
        path = out / name
        if path.suffix.lower() != ".png":
            continue
        from PIL import Image

        with Image.open(path) as im:
            width_px = im.size[0]
            dpi = max(36, round(width_px / (TARGET_NATURAL_PT / 72.0)))
            im.save(path, dpi=(dpi, dpi), optimize=True)
        rows[i] = (name, note, sha256(path))

    man = out / "MANIFEST.md"
    with man.open("w") as fh:
        fh.write("# Vendored figures\n\n")
        fh.write("Generated by `scripts/sync_figures.py`. Do not edit by hand.\n\n")
        fh.write(f"- deck assets: `{deck}`\n" if deck else "- deck assets: (not found)\n")
        fh.write(f"- deck URL: {DECK_URL}\n\n")
        fh.write("| figure | provenance | sha256 (first 16) |\n|---|---|---|\n")
        for name, note, digest in rows:
            fh.write(f"| `{name}` | {note} | `{digest}` |\n")
        if missing:
            fh.write("\n## Not vendored (deck assets missing)\n\n")
            for m in missing:
                fh.write(f"- {m}\n")

    print(f"\n{len(rows)} figures in {out}/ ; manifest: {man}")
    if missing:
        print(f"{len(missing)} not vendored (no deck assets at {deck})", file=sys.stderr)
        print("  set DECK_ASSETS=/path/to/presentations/iaa-so-chemical-tagging-2026", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
