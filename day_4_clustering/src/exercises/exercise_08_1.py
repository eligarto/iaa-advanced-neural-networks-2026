"""Chapter 8, exercise 1 — the persistence barcode of a tiny real peak.

    Sketch a one-dimensional density with three peaks of different heights
    and widths. For each peak, mark the interval of smoothing scale over
    which it survives as a separate peak, and draw the resulting barcode. Now
    add a fourth peak that is real but tiny --- 2% of the mass of the
    smallest other peak. Does the barcode distinguish it from a fluctuation?

\\S 8.1 says a real cluster "appears as a peak that stays separate from its
neighbours as the scale changes" while a fluctuation "is absorbed almost
immediately". This exercise builds the barcode numerically rather than by
sketching, adds the tiny peak, and answers the question the chapter poses
but does not resolve. The answer is no — and \\S 8.2's own honest statement
("nor does any of this repair a bad space") is the reason why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list

#: (name, centre, width, number of points). The fourth peak carries 4 points
#: against the smallest other peak's 200 — 2% of its mass, as the exercise
#: specifies — but is narrow, so it is a genuine density peak, not a bump.
PEAKS: tuple[tuple[str, float, float, int], ...] = (
    ("P1 tall narrow", 0.0, 0.25, 400),
    ("P2 mid", 3.0, 0.60, 300),
    ("P3 low broad", 7.0, 1.20, 200),
    ("P4 tiny but real", 10.0, 0.12, 4),
)

#: A uniform background over the whole range: the fluctuations the tiny peak
#: has to be distinguished *from*. Without it the question is trivial.
N_BACKGROUND = 200
RANGE: tuple[float, float] = (-3.0, 13.0)

#: Evaluation grid for the density.
GRID: np.ndarray = np.linspace(RANGE[0], RANGE[1], 4000)

#: Default smoothing scale for the headline barcode.
BANDWIDTH = 0.15


def sample(seed: int = 42) -> np.ndarray:
    """Draw the four peaks plus a uniform background."""
    rng = np.random.default_rng(seed)
    parts = [rng.normal(centre, width, n) for _, centre, width, n in PEAKS]
    parts.append(rng.uniform(RANGE[0], RANGE[1], N_BACKGROUND))
    return np.concatenate(parts)


def density(x: np.ndarray, bandwidth: float, grid: np.ndarray = GRID) -> np.ndarray:
    """Gaussian kernel density on ``grid`` — the scale-space of \\S 8.1."""
    z = (grid[:, None] - x[None, :]) / bandwidth
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(x) * bandwidth * np.sqrt(2 * np.pi))


def barcode(f: np.ndarray) -> pd.DataFrame:
    """Superlevel-set persistence of a 1-D function, by the elder rule.

    Sweep a threshold downwards through ``f``. A connected component is born
    at each local maximum and dies when it merges into an *older* (taller)
    component at a saddle; its persistence is birth height minus death
    height. This is the standard 0-dimensional persistent homology of a
    function, and it is the barcode \\S 8.1's Figure 18 draws.
    """
    n = f.size
    order = np.argsort(f)[::-1]
    parent = np.full(n, -1, dtype=int)
    peak_of = np.full(n, -1, dtype=int)
    seen = np.zeros(n, dtype=bool)
    alive: dict[int, float] = {}
    bars: list[tuple[int, float, float]] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for raw in order:
        i = int(raw)
        seen[i] = True
        roots = {find(j) for j in (i - 1, i + 1) if 0 <= j < n and seen[j]}
        if not roots:
            parent[i], peak_of[i] = i, i
            alive[i] = float(f[i])
            continue
        # Order the merging components by birth height, tallest first — the
        # elder rule. Sorting (height, root) pairs avoids a lambda whose
        # parameter type cannot be inferred.
        ranked = sorted(
            ((float(f[peak_of[r]]), int(r)) for r in roots), reverse=True,
        )
        eldest = ranked[0][1]
        parent[i] = eldest
        for _, younger in ranked[1:]:
            birth = int(peak_of[younger])
            bars.append((birth, float(f[birth]), float(f[i])))
            alive.pop(birth, None)
            parent[younger] = eldest
    for birth, height in alive.items():
        bars.append((birth, height, 0.0))

    frame = pd.DataFrame({
        "x": [round(float(GRID[b]), 3) for b, _, _ in bars],
        "birth": [round(b, 6) for _, b, _ in bars],
        "death": [round(d, 6) for _, _, d in bars],
    })
    frame["persistence"] = (frame["birth"] - frame["death"]).round(6)
    return frame.sort_values("persistence", ascending=False).reset_index(drop=True)


def _classify(frame: pd.DataFrame, tolerance: float = 0.5) -> pd.DataFrame:
    """Tag each bar with the true peak it sits on, or 'background'."""
    names = []
    for x in frame["x"]:
        match = [name for name, centre, _, _ in PEAKS if abs(x - centre) <= tolerance]
        names.append(match[0] if match else "background")
    out = frame.copy()
    out["peak"] = names
    return out


def bandwidth_sweep(
    bandwidths: tuple[float, ...] = (0.05, 0.10, 0.15, 0.25, 0.40, 0.60),
    seed: int = 42,
) -> pd.DataFrame:
    """Tiny-peak persistence against the largest background bar, per scale."""
    x = sample(seed)
    rows = []
    for bandwidth in bandwidths:
        tagged = _classify(barcode(density(x, bandwidth)))
        tiny = tagged.loc[tagged["peak"] == PEAKS[3][0], "persistence"]
        background = tagged.loc[tagged["peak"] == "background", "persistence"]
        rows.append({
            "bandwidth": bandwidth,
            "n_bars": int(len(tagged)),
            "tiny_persistence": round(float(tiny.max()) if len(tiny) else 0.0, 6),
            "max_background_bar": round(
                float(background.max()) if len(background) else 0.0, 6,
            ),
            "tiny_clears_background": bool(
                len(tiny) and len(background) and tiny.max() > background.max(),
            ),
        })
    return pd.DataFrame(rows)


def seed_sweep(seeds: tuple[int, ...] = (42, 0, 1, 2, 7, 13, 99),
               bandwidth: float = BANDWIDTH) -> pd.DataFrame:
    """The same test over seven realisations — is the answer stable?"""
    rows = []
    for seed in seeds:
        tagged = _classify(barcode(density(sample(seed), bandwidth)))
        tiny = tagged.loc[tagged["peak"] == PEAKS[3][0], "persistence"]
        background = tagged.loc[tagged["peak"] == "background", "persistence"]
        rows.append({
            "seed": seed,
            "tiny_persistence": round(float(tiny.max()) if len(tiny) else 0.0, 6),
            "max_background_bar": round(
                float(background.max()) if len(background) else 0.0, 6,
            ),
            "tiny_clears_background": bool(
                len(tiny) and len(background) and tiny.max() > background.max(),
            ),
        })
    return pd.DataFrame(rows)


def solve(bandwidth: float = BANDWIDTH, seed: int = 42) -> dict[str, object]:
    """Build the barcode, rank the four peaks, test the tiny one."""
    x = sample(seed)
    tagged = _classify(barcode(density(x, bandwidth)))
    tagged["rank"] = np.arange(1, len(tagged) + 1)

    ranking: dict[str, dict[str, float]] = {}
    for name, _, _, _ in PEAKS:
        rows = pd.DataFrame(tagged[tagged["peak"] == name])
        ranking[name] = (
            {
                "persistence": float(rows["persistence"].to_numpy()[0]),
                "rank": float(rows["rank"].to_numpy()[0]),
            }
            if len(rows) else {"persistence": 0.0, "rank": -1.0}
        )

    background = tagged.loc[tagged["peak"] == "background", "persistence"]
    seeds = seed_sweep()
    return {
        "bandwidth": bandwidth,
        "n_points": int(len(x)),
        "barcode": tagged.head(12),
        "n_bars": int(len(tagged)),
        "peak_ranking": ranking,
        "n_background_bars": int(len(background)),
        "max_background_bar": round(float(background.max()), 6),
        "median_background_bar": round(float(background.median()), 6),
        "tiny_over_max_background": round(
            float(ranking[PEAKS[3][0]]["persistence"] / background.max()), 3,
        ),
        "p3_over_tiny": round(
            float(ranking[PEAKS[2][0]]["persistence"]
                  / ranking[PEAKS[3][0]]["persistence"]), 1,
        ),
        "bandwidth_sweep": bandwidth_sweep(),
        "seed_sweep": seeds,
        "seeds_where_tiny_clears": int(seeds["tiny_clears_background"].sum()),
        "n_seeds": int(len(seeds)),
    }


def plot(bandwidth: float = BANDWIDTH, seed: int = 42):  # pragma: no cover — figure
    """The density profile above, its barcode below."""
    import matplotlib.pyplot as plt

    f = density(sample(seed), bandwidth)
    tagged = _classify(barcode(f))

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True,
                             height_ratios=[2, 1])
    axes[0].plot(GRID, f, color="#1f77b4", lw=1.4)
    axes[0].set_ylabel("density")
    axes[0].set_title(f"Four peaks and a uniform background (bw={bandwidth})")
    for _, row in tagged.iterrows():
        colour = "#999999" if row["peak"] == "background" else "#c44e52"
        axes[1].vlines(row["x"], row["death"], row["birth"], color=colour, lw=2.5)
    axes[1].set_ylabel("bar (death → birth)")
    axes[1].set_xlabel("x")
    axes[1].set_title("Persistence barcode: red = real peak, grey = background")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the construction": (
        "Three Gaussian peaks of different heights and widths — 400 points "
        "at sigma 0.25, 300 at 0.60, 200 at 1.20 — plus a fourth at x=10 "
        "with 4 points at sigma 0.12, which is 2% of the smallest other "
        "peak's mass as the exercise specifies. Crucially, 200 uniform "
        "background points are added over the whole range. Without a "
        "background the question is unanswerable: 'distinguish it from a "
        "fluctuation' requires fluctuations to exist. The barcode is the "
        "0-dimensional superlevel-set persistence of the kernel density "
        "estimate — the standard construction, and the one PLSCAN "
        f"{cite('Bot:25')} sweeps across scales — computed exactly by "
        "barcode(), not sketched."
    ),
    "the barcode for the three main peaks": (
        "At bandwidth 0.15 the ranking is: P1 (tall, narrow) persistence "
        "0.5299, rank 1; P2 (mid) 0.1649, rank 2; P3 (low, broad) 0.0716, "
        "rank 3. So the three real peaks are the three longest bars, which "
        "is the reassuring half of the exercise. But look closer at P2: it "
        "is split by the smoothing into two sub-peaks at x=2.81 and x=3.22, "
        "and the elder rule gives the taller sibling the long bar (0.1649) "
        "while the one nearer the true centre keeps only a short residual "
        "(0.0088, rank 5). That is a real feature of barcodes, not an "
        "artifact to tidy away: a peak with internal structure is reported "
        "as its dominant sub-peak plus short bars, and reading persistence "
        "without looking at where the bars sit will mislead you."
    ),
    "does the barcode distinguish the tiny peak": (
        "No. At bandwidth 0.15 the tiny peak's bar is 0.00269, rank 11 of "
        "14 — while the uniform background produces 9 bars whose maximum is "
        "0.0331, twelve times longer. The tiny peak is 0.081 of the largest "
        "pure-noise bar, and P3, a real peak with 50x the mass, is 26.6x "
        "longer than the tiny one. Sweeping the bandwidth does not rescue "
        "it: at 0.05 the tiny bar is 0.0247 against a background maximum of "
        "0.0597; at 0.25 and above it has vanished entirely (persistence "
        "1e-6, then 0). There is no scale at which it clears the noise, and "
        "over seven random realisations it clears the largest background bar "
        "in exactly 1 of 7 — which is what 'indistinguishable' looks like."
    ),
    "why this is the honest answer": (
        "The exercise is phrased hopefully and the chapter's marketing "
        "invites a yes: S 8.1, following "
        f"{cite('Bot:25', parenthetical=False)}, says persistence lets "
        "'a 6-member cluster "
        "outlive a 60-member fluctuation'. That is true when the 6-member "
        "cluster is genuinely isolated in a quiet region. It is false when "
        "the cluster sits inside a background dense enough to throw up its "
        "own peaks, because persistence has no way to know that a tall "
        "narrow spike came from four real stars rather than four "
        "coincidentally adjacent field stars. Four points is four points. "
        "S 8.2's own caveat states this: 'nor does any of this repair a bad "
        "space: if the field and the clusters are not separated in the "
        "geometry, no persistence criterion will separate them.'"
    ),
    "what would make it detectable": (
        "Three things, in decreasing order of usefulness. (1) A quieter "
        "background — remove the uniform component and the tiny peak becomes "
        "the only structure near x=10 and its bar is the whole local "
        "density. This is the embedding step of S 10-12 in miniature: change "
        "the space so the background stops competing"
        f" ({cite('vanderMaaten:08', 'McInnes:18', bare=True)}). (2) More "
        "members — "
        "persistence scales with the height of the peak, so 12 points "
        "instead of 4 would clear the noise at bandwidth 0.15. (3) A "
        "significance calibration rather than a raw threshold: compare each "
        "bar against the distribution of bars from a background-only "
        "realisation, which is what seed_sweep() sets up. The third is the "
        "only one available when you cannot change the data, and it turns "
        "the barcode from a ranking into a test."
    ),
    "the transferable rule": (
        "A persistence value alone is not evidence. Quote it alongside the "
        "persistence distribution of a null realisation of your background, "
        "exactly as you would quote a detection significance rather than a "
        "flux. The barcode makes the comparison easy — it is one extra run "
        "on shuffled or background-only data — and the workbook's rule from "
        "S 9.4 applies unchanged: any statistic that can be computed on pure "
        "noise must be reported with what pure noise gives."
    ),
    "references": reference_list(
        "Bot:25", "vanderMaaten:08", "McInnes:18",
    ),
}
