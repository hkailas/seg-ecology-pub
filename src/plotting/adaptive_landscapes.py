"""Draw precomputed adaptive fitness landscapes."""

import matplotlib.pyplot as plt

from src.plotting.style import PUBLICATION_RC

COLORS = ("#E69F00", "#56B4E9", "#009E73", "#F0E442",
          "#0072B2", "#D55E00", "#CC79A7", "#000000")


def plot_adaptive_landscape(result, ax=None):
    with plt.rc_context(PUBLICATION_RC):
        figure, ax = plt.subplots(figsize=(8, 5)) if ax is None else (ax.figure, ax)
        ax.plot(result.mutant_traits, result.mutant_fitness, color="black",
                linewidth=2, label=r"$G(v)$")
        ax.axhline(0, color="#666666", linestyle="--", linewidth=1)
        for i, (N, u, fitness) in enumerate(zip(result.resident_populations,
                                                 result.resident_traits,
                                                 result.resident_fitness)):
            ax.scatter([u], [fitness], s=70, zorder=3,
                       color=COLORS[i % len(COLORS)], edgecolors="black",
                       linewidths=0.5, label=f"Species {i + 1} ($u={u:.2f}$, $N={N:.2f}$)")
        ax.set_xlabel(r"Trait value ($v$)")
        ax.set_ylabel(r"Fitness $G(v)$")
        ax.set_title(f"Adaptive landscape ($E={result.energy:g}$, $t={result.resident_time:g}$)")
        if result.x_limits is not None:
            ax.set_xlim(*result.x_limits)
        else:
            ax.set_xlim(float(result.mutant_traits[0]), float(result.mutant_traits[-1]))
        if result.y_limits is not None:
            ax.set_ylim(*result.y_limits)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=False)
        return figure, ax
