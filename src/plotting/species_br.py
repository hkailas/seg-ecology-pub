"""Plots of species best-response populations and equilibrium traits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .annotations import (
    add_nash_trait_markers,
    add_stackelberg_trait_markers,
)
from .style import (
    BRANCH_COLORS,
    BRANCH_DISPLAY,
    DEFAULT_BRANCH_COLOR,
    EQUILIBRIUM_MARKER_SIZE,
    PUBLICATION_RC,
    ordered_branches,
)


def _aligned_species_arrays(
    E: Any,
    N_star: Any,
    branch: Any,
    equilibrium: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert and validate aligned record-level species-BR arrays."""
    E_values = np.asarray(E, dtype=float)
    populations = np.asarray(N_star, dtype=float)
    branches = np.asarray(branch, dtype=object)
    equilibrium_flags = np.asarray(equilibrium, dtype=bool)

    if E_values.ndim != 1:
        raise ValueError("E must be a one-dimensional array.")
    if populations.ndim != 2:
        raise ValueError("N_star must have shape (records, species).")
    if branches.ndim != 1 or equilibrium_flags.ndim != 1:
        raise ValueError("branch and equilibrium must be one-dimensional arrays.")

    record_count = len(E_values)
    if not (
        len(populations)
        == len(branches)
        == len(equilibrium_flags)
        == record_count
    ):
        raise ValueError(
            "E, N_star, branch, and equilibrium must have matching record counts."
        )
    return E_values, populations, branches, equilibrium_flags


def _figure_and_axis(ax: Any, figsize: tuple[float, float]) -> tuple[Any, Any]:
    """Create an axis when the caller did not provide one."""
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def plot_sum_populations_vs_e(
    E: Any,
    N_star: Any,
    branch: Any,
    equilibrium: Any,
    *,
    ax: Any = None,
    marker_size: float = 12,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Species best response: total population",
) -> tuple[Any, Any]:
    """Scatter total equilibrium population against E, grouped by branch."""

    E_values, populations, branches, equilibrium_flags = _aligned_species_arrays(
        E,
        N_star,
        branch,
        equilibrium,
    )
    total_population = np.sum(populations, axis=1)
    retained = equilibrium_flags & np.isfinite(E_values) & np.isfinite(total_population)

    with mpl.rc_context(PUBLICATION_RC):
        fig, ax = _figure_and_axis(ax, figsize)
        for branch_name in ordered_branches(branches[retained]):
            branch_mask = retained & (branches == branch_name)
            branch_E = E_values[branch_mask]
            branch_N = total_population[branch_mask]
            color = BRANCH_COLORS.get(branch_name, DEFAULT_BRANCH_COLOR)
            label = BRANCH_DISPLAY.get(branch_name, branch_name)

            ax.scatter(
                branch_E,
                branch_N,
                s=marker_size,
                color=color,
                alpha=0.7,
                edgecolors="none",
                label=label,
            )

        ax.set_xlabel(r"$E$")
        ax.set_ylabel(r"Total population $\sum_i N_i^*$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False)
        fig.tight_layout()
    return fig, ax


def plot_equilibrium_traits_vs_e(
    E: Any,
    N_star: Any,
    u_star: Any,
    branch: Any,
    equilibrium: Any,
    nash_points: Sequence[Mapping[str, Any]] = (),
    *,
    E_stackelberg: float | None = None,
    N_star_stackelberg: Any = None,
    u_star_stackelberg: Any = None,
    branch_stackelberg: str | None = None,
    ax: Any = None,
    population_threshold: float = 1e-4,
    marker_size: float = 12,
    equilibrium_marker_size: float = EQUILIBRIUM_MARKER_SIZE,
    figsize: tuple[float, float] = (14, 8),
    title: str = "Species best response: equilibrium traits",
) -> tuple[Any, Any]:
    """Plot traits against E with trait-marker game equilibria."""
    if population_threshold < 0:
        raise ValueError("population_threshold must be non-negative.")

    E_values, populations, branches, equilibrium_flags = _aligned_species_arrays(
        E,
        N_star,
        branch,
        equilibrium,
    )
    traits = np.asarray(u_star, dtype=float)
    if traits.ndim != 2:
        raise ValueError("u_star must have shape (records, species).")
    if traits.shape != populations.shape:
        raise ValueError("N_star and u_star must have matching shapes.")

    retained_records = equilibrium_flags & np.isfinite(E_values)
    with mpl.rc_context(PUBLICATION_RC):
        fig, ax = _figure_and_axis(ax, figsize)
        for branch_name in ordered_branches(branches[retained_records]):
            record_mask = retained_records & (branches == branch_name)
            branch_populations = populations[record_mask]
            branch_traits = traits[record_mask]
            branch_E = np.broadcast_to(
                E_values[record_mask, np.newaxis],
                branch_traits.shape,
            )
            alive = branch_populations >= population_threshold
            finite = np.isfinite(branch_traits) & np.isfinite(branch_populations)
            plot_mask = alive & finite
            if not np.any(plot_mask):
                continue

            ax.scatter(
                branch_E[plot_mask],
                branch_traits[plot_mask],
                s=marker_size,
                color=BRANCH_COLORS.get(branch_name, DEFAULT_BRANCH_COLOR),
                alpha=0.7,
                edgecolors="none",
                label=BRANCH_DISPLAY.get(branch_name, branch_name),
            )

        add_nash_trait_markers(
            ax,
            nash_points,
            population_threshold=population_threshold,
            marker_size=equilibrium_marker_size,
        )
        add_stackelberg_trait_markers(
            ax,
            E_stackelberg,
            N_star_stackelberg,
            u_star_stackelberg,
            branch_stackelberg,
            population_threshold=population_threshold,
            marker_size=equilibrium_marker_size,
        )
        ax.set_xlabel(r"$E$")
        ax.set_ylabel(r"Equilibrium trait $u_i^*$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, loc="best", ncol=2)
        fig.tight_layout()
    return fig, ax


def plot_single_species_best_responses(
    species_E: Any,
    species_u: Any,
    leader_E: Any,
    leader_u: Any,
    nash_points: Sequence[Mapping[str, Any]] = (),
    *,
    E_stackelberg: float | None = None,
    N_star_stackelberg: Any = None,
    u_star_stackelberg: Any = None,
    branch_stackelberg: str | None = None,
    population_threshold: float = 1e-4,
    near_zero_E_threshold: float = 0.01,
    equilibrium_marker_size: float = EQUILIBRIUM_MARKER_SIZE,
    ax: Any = None,
    figsize: tuple[float, float] = (10, 7),
    title: str = "Single-species best responses",
) -> tuple[Any, Any]:
    """Plot species and leader best-response curves on the same axes."""
    if population_threshold < 0:
        raise ValueError("population_threshold must be non-negative.")
    if near_zero_E_threshold < 0:
        raise ValueError("near_zero_E_threshold must be non-negative.")

    species_E = np.asarray(species_E, dtype=float)
    species_u = np.asarray(species_u, dtype=float)
    leader_E = np.asarray(leader_E, dtype=float)
    leader_u = np.asarray(leader_u, dtype=float)
    arrays = (species_E, species_u, leader_E, leader_u)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Best-response coordinates must be one-dimensional.")
    if len(species_E) != len(species_u):
        raise ValueError("Species best-response coordinates must be aligned.")
    if len(leader_E) != len(leader_u):
        raise ValueError("Leader best-response coordinates must be aligned.")
    if not len(species_E) or not len(leader_E):
        raise ValueError("Both best-response curves require at least one point.")

    species_finite = np.isfinite(species_E) & np.isfinite(species_u)
    leader_finite = (
        np.isfinite(leader_E)
        & np.isfinite(leader_u)
        & (leader_E > near_zero_E_threshold)
    )
    if not np.any(species_finite):
        raise ValueError("The species best-response curve requires finite points.")

    with mpl.rc_context(PUBLICATION_RC):
        fig, ax = _figure_and_axis(ax, figsize)
        ax.plot(
            species_E[species_finite],
            species_u[species_finite],
            color="#2ca02c",
            linewidth=2,
            label="Species best response",
        )
        if np.any(leader_finite):
            plotted_leader_E = np.where(leader_finite, leader_E, np.nan)
            plotted_leader_u = np.where(leader_finite, leader_u, np.nan)
            ax.plot(
                plotted_leader_E,
                plotted_leader_u,
                color="#1f77b4",
                linewidth=2,
                linestyle="--",
                label="Leader best response",
            )

        add_nash_trait_markers(
            ax,
            nash_points,
            population_threshold=population_threshold,
            marker_size=equilibrium_marker_size,
        )
        add_stackelberg_trait_markers(
            ax,
            E_stackelberg,
            N_star_stackelberg,
            u_star_stackelberg,
            branch_stackelberg,
            population_threshold=population_threshold,
            marker_size=equilibrium_marker_size,
        )

        ax.set_xlabel(r"$E$")
        ax.set_ylabel(r"Trait $u$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, loc="best")
        fig.tight_layout()
    return fig, ax
