# Companion workbook: chemical tagging (IAA-SO 2026)

Long-form companion to the lecture deck at
`garciadias.github.io/#/presentations/iaa-so-chemical-tagging-2026`, built from
the benchmark in this repository with the Annual Reviews `ar-1col` class.

    make            # check sources, vendor the deck figures, build workbook.pdf
    make check      # static checks only (citations, refs, figures, environments)
    make figures    # re-vendor figures from the deck's asset directory
    make pdf        # LaTeX only, when figures/ is already populated
    make clean      # drop LaTeX intermediates

The output is `workbook.pdf` (~63 pp, A4/letter single column, ~23 000 words).

## Layout

| path | what it is |
|---|---|
| `workbook.tex` | main file: class setup, title page, class patches, figure macros, chapter list |
| `chapters/NN_*.tex` | one file per chapter, 17 in total, in reading order |
| `figures/` | vendored deck PNGs, generated filmstrips, QR codes, `MANIFEST.md` |
| `scripts/sync_figures.py` | copies/vendors every figure the workbook uses, from the deck |
| `scripts/kmeans_baseline.py` | the K-means run added for the workbook (§4, Table 3) |
| `scripts/check_workbook.py` | pre-build static checks; run by `make check` |
| `references.bib` | bibliography, Annual Reviews Harvard via `ar-style2.bst` |
| `results/kmeans_baseline.csv` | output of the K-means script |
| `ar-1col.cls`, `ar-style2.bst` | vendored from the publisher's template zip |

## Provenance of every number

The workbook quotes three kinds of result, and the text says which is which:

1. **The repository's own benchmark**, at a recorded commit: the cluster-only
   tables (§13), the field-retrieval table (§13), the lever sweep (§13, DR17),
   the two-stage pipeline and isochrone fits (§15, DR17 study). Sources are
   named in the sections that use them.
2. **The repository's own diagnostics**, reported as they stand: the
   duplicate-`APOGEE_ID` audit and the row-order sensitivity (§9), the
   data-product mismatch and the withdrawn field-retrieval claim (§14).
3. **One new measurement**, `scripts/kmeans_baseline.py`: K-means on the same
   member matrix as the other arms, which had not been run before. It is
   Table 3 and the K-means summary box in §4.

## Three things the class needed

`ar-1col` is designed for a journal article of a few thousand words and five
figures, and three of its design decisions break at workbook scale. All three
are patched in `workbook.tex`, with comments explaining why:

1. **Figures ignore `width=`.** The class's `\Gin@setfile` hook rebuilds every
   figure at its natural (declared) size and then left-shifts anything wider
   than the text column off the page. Every inclusion therefore goes through
   `\fig[<fraction>]{<file>}`, which is a `\resizebox`, and
   `sync_figures.py` rewrites each PNG's declared resolution so the natural
   width stays under the column width.
2. **The running footer prints `www.annualreviews.org`.** The class calls
   `\pagestyle{headings}` while loading, which *executes* its footer
   definition, so replacing `\ps@headings` is not enough. The replacement must
   be followed by another `\pagestyle{headings}`.
3. **The contents list is one unbreakable 36 pc box.** Fine for six sections;
   a 17-chapter contents overflows it by ~490 pt and bleeds off the page. The
   contents is re-set as ordinary page-breakable material, with wider
   number boxes so two-digit chapter numbers do not collide with their titles.

## Figures

`scripts/sync_figures.py` is the only thing that writes into `figures/`; every
file it produces is listed in `figures/MANIFEST.md` with its provenance and a
hash. Animated GIFs from the deck become 2×2 filmstrips (readable in a 5-inch
column) plus a QR code pointing at the live animation. The step figures are the
deck's PNGs, re-encoded losslessly only to fix their declared resolution.

Figure quality is the one place the workbook is limited by its sources: the
teaching figures are generated at `figure.dpi: 100`
(`scripts/teaching_assets/common.py:40`) and the result figures at 160, so a
few of them print soft. Regenerating them requires the model checkpoint and the
full sample, which live on the analysis desktop.

## Open items

The workbook's own `FUTURE ISSUES` box (§17) is the authoritative list; the two
that affected its numbers are:

- the member matrix contains duplicate `APOGEE_ID` rows (298 of 1 002), which
  inflates the largest clusters and makes the abundances-only t-SNE row
  row-order unstable;
- the spectral latent was trained on a mixture of `apStar` and `aspcapStar`
  data products, which invalidates field retrieval for it until the re-embed on
  736 staged `mwmStar` spectra is done.
