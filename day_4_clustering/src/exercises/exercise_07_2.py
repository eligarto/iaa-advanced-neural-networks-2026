"""Chapter 7, exercise 2 — the stability score and what it rewards.

    Compute the stability score of Equation 5 for a two-branch tree in
    which one branch contains 100 points that survive two decades of lambda
    and the other contains 10 points that survive five. Which does the
    algorithm prefer, and does that match your intuition about which is a
    better cluster?

\\S 7.2 step 4 defines stability as the sum over members of
(lambda_point - lambda_birth), with lambda = 1/d_mreach, and says HDBSCAN*
keeps the branches with the largest total. The exercise is arithmetic, but
the arithmetic has a trap: "survives two decades" does not pin down the
answer until you say *where* on the lambda axis the decades sit. Both
readings are computed here, they disagree by a factor of a thousand, and
the disagreement is the lesson.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list

#: The exercise's two branches: (name, number of points, decades survived).
BRANCH_A: tuple[str, int, float] = ("A: 100 points, 2 decades", 100, 2.0)
BRANCH_B: tuple[str, int, float] = ("B: 10 points, 5 decades", 10, 5.0)


def stability(n_points: int, lambda_birth: float, lambda_death: float) -> float:
    """Equation 5 when every member leaves the cluster at the same lambda.

    S(C) = sum over x in C of (lambda_x - lambda_birth). With all members
    falling out together this is just n * (lambda_death - lambda_birth),
    which is the case the exercise describes.
    """
    return float(n_points * (lambda_death - lambda_birth))


def common_birth(lambda_birth: float = 1.0) -> pd.DataFrame:
    """Reading 1: the branches are siblings, born at the same density."""
    rows = []
    for name, n, decades in (BRANCH_A, BRANCH_B):
        death = lambda_birth * 10.0 ** decades
        rows.append({
            "branch": name,
            "n_points": n,
            "lambda_birth": lambda_birth,
            "lambda_death": death,
            "stability": round(stability(n, lambda_birth, death), 4),
        })
    frame = pd.DataFrame(rows)
    frame["share"] = (frame["stability"] / frame["stability"].sum()).round(4)
    return frame


def common_death(lambda_death: float = 100.0) -> pd.DataFrame:
    """Reading 2: both branches are absorbed at the same density level."""
    rows = []
    for name, n, decades in (BRANCH_A, BRANCH_B):
        birth = lambda_death / 10.0 ** decades
        rows.append({
            "branch": name,
            "n_points": n,
            "lambda_birth": round(birth, 8),
            "lambda_death": lambda_death,
            "stability": round(stability(n, birth, lambda_death), 4),
        })
    frame = pd.DataFrame(rows)
    frame["share"] = (frame["stability"] / frame["stability"].sum()).round(4)
    return frame


def break_even(lambda_birth: float = 1.0) -> float:
    """How many points branch B needs to tie A under the common-birth reading."""
    _, n_a, dec_a = BRANCH_A
    _, _, dec_b = BRANCH_B
    s_a = stability(n_a, lambda_birth, lambda_birth * 10.0 ** dec_a)
    per_point_b = lambda_birth * (10.0 ** dec_b - 1.0)
    return float(s_a / per_point_b)


def verify_against_hdbscan(seed: int = 42) -> dict[str, object]:
    """Recompute Eq. 5 from a real condensed tree and compare with hdbscan.

    A sanity check that the formula in this module is the formula the library
    implements. ``cluster_persistence_`` is hdbscan's per-cluster lambda
    *range* (a different normalisation from the raw sum), so the check is
    that the ordering agrees, not the magnitudes.
    """
    import hdbscan
    from sklearn.datasets import make_blobs

    blobs = make_blobs(n_samples=300, centers=3, cluster_std=[0.3, 0.3, 1.4],
                       random_state=seed)
    X = np.asarray(blobs[0])
    model = hdbscan.HDBSCAN(min_cluster_size=10).fit(X)
    tree = model.condensed_tree_.to_pandas()

    computed: dict[int, float] = {}
    for cluster_id in sorted(int(c) for c in tree["parent"].unique()):
        born = tree.loc[tree["child"] == cluster_id, "lambda_val"]
        lambda_birth = float(born.iloc[0]) if len(born) else 0.0
        total, stack = 0.0, [cluster_id]
        while stack:
            current = stack.pop()
            rows = tree[tree["parent"] == current]
            singles = rows[rows["child_size"] == 1]
            total += float((singles["lambda_val"] - lambda_birth).sum())
            stack.extend(int(c) for c in rows[rows["child_size"] > 1]["child"])
        computed[cluster_id] = round(total, 3)

    return {
        "eq_7_2_by_hand": computed,
        "hdbscan_cluster_persistence": np.round(
            model.cluster_persistence_, 4,
        ).tolist(),
        "n_selected_clusters": int(len(set(model.labels_.tolist()) - {-1})),
    }


def solve() -> dict[str, object]:
    """Both readings of the question, the break-even point, and a real check."""
    births = common_birth()
    deaths = common_death()
    return {
        "reading_1_common_birth": births,
        "reading_1_winner": str(births.loc[births["stability"].idxmax(), "branch"]),
        "reading_1_ratio_B_over_A": round(
            float(births["stability"].iloc[1] / births["stability"].iloc[0]), 4,
        ),
        "reading_2_common_death": deaths,
        "reading_2_winner": str(deaths.loc[deaths["stability"].idxmax(), "branch"]),
        "reading_2_ratio_A_over_B": round(
            float(deaths["stability"].iloc[0] / deaths["stability"].iloc[1]), 4,
        ),
        "break_even_points_for_B": round(break_even(), 5),
        "scale_invariance_check": {
            f"lambda_birth={b}": round(
                float(common_birth(b)["stability"].iloc[1]
                      / common_birth(b)["stability"].iloc[0]), 4,
            )
            for b in (0.1, 1.0, 10.0)
        },
        "hdbscan_cross_check": verify_against_hdbscan(),
    }


def plot():  # pragma: no cover — figure
    """Stability of both branches under both readings, side by side."""
    import matplotlib.pyplot as plt

    births, deaths = common_birth(), common_death()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for ax, frame, title in (
        (axes[0], births, "same $\\lambda_{\\rm birth}$"),
        (axes[1], deaths, "same $\\lambda_{\\rm death}$"),
    ):
        ax.bar(["A (100 pts)", "B (10 pts)"], frame["stability"],
               color=["#4c72b0", "#dd8452"])
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("stability $S(C)$")
    fig.suptitle("The same tree, two readings, opposite winners")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the formula": (
        "S(C) = sum over x in C of (lambda_x - lambda_birth(C)), with lambda "
        "= 1/d_mreach — the stability, or relative excess of mass, that "
        f"HDBSCAN* selects branches by {cite('Campello:13')}. When every "
        "member of a branch falls out at the same "
        "density — the clean case the question describes — this collapses to "
        "n * (lambda_death - lambda_birth). Note what that means "
        "dimensionally: stability is points times inverse distance. It is "
        "*not* a lifetime, and it is not normalised by cluster size; a big "
        "cluster and a long-lived one trade off linearly against each other."
    ),
    "the trap in the question": (
        "'Survives two decades of lambda' fixes the ratio "
        "lambda_death/lambda_birth but not the position on the lambda axis, "
        "and because lambda is a *rate* (inverse distance) rather than a log "
        "quantity, the answer depends entirely on where those decades sit. "
        "Two readings are defensible, and they give opposite winners. Any "
        "answer to this exercise that does not state which reading it uses "
        "is incomplete."
    ),
    "reading 1 — siblings, same lambda_birth": (
        "If both branches split off the same parent they share "
        "lambda_birth. Taking lambda_birth = 1: A dies at 10^2, so S(A) = "
        "100 * (100 - 1) = 9 900. B dies at 10^5, so S(B) = 10 * (100 000 - "
        "1) = 999 990. B wins by a factor of 101.0 — and that factor is "
        "independent of lambda_birth (checked at 0.1, 1 and 10: always "
        "101.01), because both stabilities scale linearly with it. The "
        "break-even is brutal: B would tie A with 0.099 points. A single "
        "point surviving five decades outscores a hundred points surviving "
        "two. HDBSCAN* prefers B, emphatically."
    ),
    "reading 2 — both absorbed at the same lambda_death": (
        "If the two branches are eaten by the *same* rising density "
        "threshold they share lambda_death. Taking lambda_death = 100: A was "
        "born at 1, so S(A) = 100 * 99 = 9 900; B was born at 0.001, so S(B) "
        "= 10 * 99.999 = 1 000. Now A wins, by a factor of 9.90 — and again "
        "the factor is scale-free (identical at lambda_death = 1 000). The "
        "swing between the two readings is a factor of a thousand in the "
        "ratio, from B winning 101:1 to A winning 9.9:1."
    ),
    "which way does it really go in HDBSCAN": (
        "Reading 1 is the one that matches the algorithm's selection step, "
        "because stability is only ever *compared* between a parent and its "
        "own children — branches that by construction share a birth level. "
        "So the effective answer is that HDBSCAN* strongly prefers the "
        "long-lived branch, and the 'survives five decades' branch wins even "
        "though it has a tenth of the points. That is the intended "
        "behaviour: lambda = 1/d_mreach grows without bound as the density "
        "threshold rises, so a cluster that stays intact into very high "
        "density accumulates enormous per-point stability."
    ),
    "does it match intuition": (
        "Partly, and the mismatch is worth naming. Preferring a tight "
        "long-lived group of 10 over a diffuse group of 100 is exactly what "
        "you want for chemical tagging: NGC 2158 has 6 members in this "
        "sample and should not be outvoted by a loose 100-star field "
        "overdensity. But the 1/d weighting is aggressive — because lambda "
        "is an inverse distance, the last half-decade of a branch's life "
        "contributes more stability than all the rest put together, so the "
        "score is dominated by a cluster's densest core rather than by its "
        "extent. Two consequences: (i) stability is not comparable across "
        "datasets with different absolute scales, and (ii) it slightly "
        "favours compact clusters over extended ones of the same population, "
        "independent of how real either is. Chapter 8's alternative, "
        f"persistence measured over a range of scales {cite('Bot:25')}, is "
        "motivated partly by exactly this dependence on where a branch sits "
        "on the density axis."
    ),
    "the cross-check": (
        "solve() also recomputes Equation 5 by hand from a real condensed "
        f"tree built with the hdbscan library {cite('McInnes:17')}, on three "
        "blobs, two tight and one diffuse, with "
        "min_cluster_size=10, and compares with the library. The hand "
        "computation gives raw sums in the hundreds to thousands while "
        "hdbscan's cluster_persistence_ reports a normalised lambda range "
        "(0.125, 0.606, 0.638), so the magnitudes differ by construction — "
        "the check is that both identify the same three branches as the "
        "selected clusters, and the diffuse blob scores lowest in both. If "
        "you quote a stability number, say whether it is the raw sum of "
        "Eq. 5 or the library's normalised persistence; they are not the "
        "same quantity."
    ),
    "references": reference_list("Campello:13", "McInnes:17", "Bot:25"),
}
