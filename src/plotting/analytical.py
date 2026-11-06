"""Plot persisted analytical data without numerical solves."""

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.style import (
    EQUILIBRIUM_MARKER_EDGE_COLOR,
    EQUILIBRIUM_MARKER_EDGE_WIDTH,
    EQUILIBRIUM_MARKER_SIZE,
    PUBLICATION_RC,
)

GREEN, ORANGE, BLUE = "#009E73", "#D55E00", "#0072B2"


def _axis(ax, size=(10, 6.67)):
    return plt.subplots(figsize=size) if ax is None else (ax.figure, ax)


def _finish(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def plot_best_responses(result, ax=None):
    with plt.rc_context(PUBLICATION_RC):
        figure, ax = _axis(ax)
        species = np.where(result.species_u == 0, np.nan, result.species_u)
        leader = np.where(result.leader_E < 0.05, np.nan, result.leader_E)
        ax.plot(result.E, species, color=GREEN, linewidth=3, label="Species BR")
        ax.plot(leader, result.u_grid, color=BLUE, linewidth=3, label="Leader BR")
        ax.scatter(result.E_nash, result.u_nash, marker="X",
                   s=EQUILIBRIUM_MARKER_SIZE, facecolors=BLUE,
                   edgecolors=EQUILIBRIUM_MARKER_EDGE_COLOR,
                   linewidths=EQUILIBRIUM_MARKER_EDGE_WIDTH, label="Nash",
                   zorder=4)
        ax.scatter(result.E_stackelberg, result.u_stackelberg, marker="o",
                   s=EQUILIBRIUM_MARKER_SIZE, facecolors=ORANGE,
                   edgecolors=EQUILIBRIUM_MARKER_EDGE_COLOR,
                   linewidths=EQUILIBRIUM_MARKER_EDGE_WIDTH,
                   label="Stackelberg", zorder=4)
        ax.legend(loc="best", frameon=False)
        ax.set_xlim(0, 1)
        _finish(ax, "Best responses", r"$E$", r"$u^*$")
        return figure, ax


def plot_payoffs(result, ax=None, *, label="Trait objective", color=ORANGE):
    with plt.rc_context(PUBLICATION_RC):
        figure, ax = _axis(ax)
        ax.plot(result.E, result.evolving_payoff, color=color,
                linestyle="-", linewidth=2.5, label=f"{label}: evo-aware")
        ax.plot(result.E, result.fixed_trait_payoff, color=color,
                linestyle="--", linewidth=2.5, label=f"{label}: eco-aware")
        ax.scatter(result.E_stackelberg, result.payoff_stackelberg, marker="o",
                   s=EQUILIBRIUM_MARKER_SIZE, facecolors=color,
                   edgecolors=EQUILIBRIUM_MARKER_EDGE_COLOR,
                   linewidths=EQUILIBRIUM_MARKER_EDGE_WIDTH, zorder=4)
        ax.scatter(result.E_fixed_optimum, result.payoff_fixed_optimum,
                   marker="X", s=EQUILIBRIUM_MARKER_SIZE, facecolors=color,
                   edgecolors=EQUILIBRIUM_MARKER_EDGE_COLOR,
                   linewidths=EQUILIBRIUM_MARKER_EDGE_WIDTH, zorder=4)
        ax.legend(loc="best", frameon=False)
        _finish(ax, "Leader payoff", r"$E$", r"$Q(E)$")
        return figure, ax
