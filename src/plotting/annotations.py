"""Shared annotations for result plots."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .style import (
    BRANCH_COLORS,
    BRANCH_DISPLAY,
    DEFAULT_BRANCH_COLOR,
    EQUILIBRIUM_MARKER_EDGE_COLOR,
    EQUILIBRIUM_MARKER_EDGE_WIDTH,
    EQUILIBRIUM_MARKER_SIZE,
)


def _surviving_traits(
    populations: Any,
    traits: Any,
    population_threshold: float,
) -> np.ndarray:
    """Return finite traits whose corresponding populations survive."""
    population_values = np.asarray(populations, dtype=float)
    trait_values = np.asarray(traits, dtype=float)
    if population_values.ndim != 1 or trait_values.ndim != 1:
        raise ValueError(
            "Equilibrium populations and traits must be one-dimensional."
        )
    if population_values.shape != trait_values.shape:
        raise ValueError(
            "Equilibrium populations and traits must have matching shapes."
        )

    retained = (
        (population_values >= population_threshold)
        & np.isfinite(population_values)
        & np.isfinite(trait_values)
    )
    return trait_values[retained]


def add_nash_trait_markers(
    ax: Any,
    nash_points: Sequence[Mapping[str, Any]],
    *,
    population_threshold: float,
    marker_size: float = EQUILIBRIUM_MARKER_SIZE,
) -> list[Any]:
    """Add an X at every surviving species trait for each Nash point."""
    artists = []
    labelled_branches: set[str] = set()
    for point in nash_points:
        if "N_nash" not in point:
            raise ValueError(
                "Nash point is missing N_nash; regenerate the analysis result "
                "before using marker annotations."
            )

        branch = str(point["branch"])
        traits = _surviving_traits(
            point["N_nash"],
            point["u_nash"],
            population_threshold,
        )
        if not len(traits):
            continue

        label = None
        if branch not in labelled_branches:
            label = f"Nash ({BRANCH_DISPLAY.get(branch, branch)})"
            labelled_branches.add(branch)
        artists.append(
            ax.scatter(
                np.full(len(traits), float(point["E_nash"])),
                traits,
                marker="X",
                s=marker_size,
                facecolors=BRANCH_COLORS.get(branch, DEFAULT_BRANCH_COLOR),
                edgecolors=EQUILIBRIUM_MARKER_EDGE_COLOR,
                linewidths=EQUILIBRIUM_MARKER_EDGE_WIDTH,
                label=label,
                zorder=4,
            )
        )
    return artists


def add_stackelberg_trait_markers(
    ax: Any,
    E_stackelberg: float | None,
    N_star_stackelberg: Any,
    u_star_stackelberg: Any,
    branch_stackelberg: str | None,
    *,
    population_threshold: float,
    marker_size: float = EQUILIBRIUM_MARKER_SIZE,
) -> list[Any]:
    """Add a filled circle at each surviving global Stackelberg trait."""
    if E_stackelberg is None:
        return []
    if (
        N_star_stackelberg is None
        or u_star_stackelberg is None
        or branch_stackelberg is None
    ):
        raise ValueError(
            "Stackelberg populations, traits, and branch are required when "
            "E_stackelberg is provided."
        )

    traits = _surviving_traits(
        N_star_stackelberg,
        u_star_stackelberg,
        population_threshold,
    )
    if not len(traits):
        return []

    branch = str(branch_stackelberg)
    return [
        ax.scatter(
            np.full(len(traits), float(E_stackelberg)),
            traits,
            marker="o",
            s=marker_size,
            facecolors=BRANCH_COLORS.get(branch, DEFAULT_BRANCH_COLOR),
            edgecolors=EQUILIBRIUM_MARKER_EDGE_COLOR,
            linewidths=EQUILIBRIUM_MARKER_EDGE_WIDTH,
            label=f"Stackelberg ({BRANCH_DISPLAY.get(branch, branch)})",
            zorder=4,
        )
    ]
