"""Leader-payoff plots for species best responses and fixed Nash traits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .style import (
    BRANCH_COLORS,
    BRANCH_DISPLAY,
    DEFAULT_BRANCH_COLOR,
    EQUILIBRIUM_MARKER_EDGE_COLOR,
    EQUILIBRIUM_MARKER_EDGE_WIDTH,
    EQUILIBRIUM_MARKER_SIZE,
    PUBLICATION_RC,
    ordered_branches,
)


def prepare_stackelberg_payoff_series(
    E: Any,
    payoff: Any,
    branch: Any,
    equilibrium: Any,
) -> dict[str, dict[str, np.ndarray]]:
    """Collect all finite equilibrium payoffs and per-E branch maxima."""
    E_values = np.asarray(E, dtype=float)
    payoffs = np.asarray(payoff, dtype=float)
    branches = np.asarray(branch, dtype=object)
    equilibrium_flags = np.asarray(equilibrium, dtype=bool)
    arrays = (E_values, payoffs, branches, equilibrium_flags)
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("E, payoff, branch, and equilibrium must be one-dimensional.")
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("E, payoff, branch, and equilibrium must be aligned.")

    finite_E = np.isfinite(E_values)
    full_E_grid = np.unique(E_values[finite_E])
    retained = equilibrium_flags & finite_E & np.isfinite(payoffs)
    series = {}
    for branch_name in ordered_branches(branches[retained]):
        branch_mask = retained & (branches == branch_name)
        branch_E = E_values[branch_mask]
        branch_payoffs = payoffs[branch_mask]
        maximum_payoffs = np.full(len(full_E_grid), np.nan, dtype=float)
        for index, E_value in enumerate(full_E_grid):
            at_E = branch_E == E_value
            if np.any(at_E):
                maximum_payoffs[index] = np.max(branch_payoffs[at_E])
        series[branch_name] = {
            "E": branch_E,
            "payoff": branch_payoffs,
            "line_E": full_E_grid.copy(),
            "line_payoff": maximum_payoffs,
        }
    return series


def _figure_and_axis(ax: Any, figsize: tuple[float, float]) -> tuple[Any, Any]:
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def _plot_stackelberg_series(
    ax: Any,
    series: Mapping[str, Mapping[str, Any]],
    *,
    E_stackelberg: float | None,
    payoff_stackelberg: float | None,
    branch_stackelberg: str | None,
    line_plot: bool,
    show_marker: bool,
    marker_size: float,
) -> None:
    for branch_name in ordered_branches(series):
        values = series[branch_name]
        color = BRANCH_COLORS.get(branch_name, DEFAULT_BRANCH_COLOR)
        display = BRANCH_DISPLAY.get(branch_name, branch_name)
        ax.scatter(
            values["E"],
            values["payoff"],
            color=color,
            s=18,
            alpha=0.65,
            edgecolors="none",
            label=display,
            zorder=2,
        )
        if line_plot:
            ax.plot(
                values["line_E"],
                values["line_payoff"],
                color=color,
                linestyle="-",
                linewidth=2,
                label="_nolegend_",
                zorder=3,
            )

    if not show_marker or E_stackelberg is None:
        return
    if payoff_stackelberg is None or branch_stackelberg is None:
        raise ValueError(
            "Stackelberg payoff and branch are required with E_stackelberg."
        )
    branch_name = str(branch_stackelberg)
    ax.scatter(
        [float(E_stackelberg)],
        [float(payoff_stackelberg)],
        s=marker_size,
        marker="o",
        facecolors=BRANCH_COLORS.get(branch_name, DEFAULT_BRANCH_COLOR),
        edgecolors=EQUILIBRIUM_MARKER_EDGE_COLOR,
        linewidths=EQUILIBRIUM_MARKER_EDGE_WIDTH,
        label="Global Stackelberg point",
        zorder=5,
    )


def _curve_label(curve: Mapping[str, Any]) -> str:
    branch = str(curve["branch"])
    display = BRANCH_DISPLAY.get(branch, branch)
    return f"Nash ({display}, E={float(curve['E_nash']):.4f})"


def _plot_nash_series(
    ax: Any,
    curves: Sequence[Mapping[str, Any]],
    *,
    show_markers: bool,
    marker_size: float,
    linestyle: str,
) -> None:
    for curve in curves:
        E_values = np.asarray(curve["E"], dtype=float)
        payoffs = np.asarray(curve["payoff"], dtype=float)
        equilibrium = np.asarray(curve["equilibrium"], dtype=bool)
        if E_values.ndim != 1 or payoffs.ndim != 1 or equilibrium.ndim != 1:
            raise ValueError("Nash curve E, payoff, and equilibrium must be 1D.")
        if len({len(E_values), len(payoffs), len(equilibrium)}) != 1:
            raise ValueError("Nash curve E, payoff, and equilibrium must align.")
        plotted_payoffs = payoffs.copy()
        plotted_payoffs[~equilibrium] = np.nan
        ax.plot(
            E_values,
            plotted_payoffs,
            color="black",
            linestyle=linestyle,
            linewidth=2,
            label=_curve_label(curve),
        )

        if not show_markers or curve.get("source") != "detected_nash":
            continue
        E_nash = curve.get("E_nash")
        payoff_nash = curve.get("payoff_nash")
        branch = curve.get("branch")
        if E_nash is None or payoff_nash is None or branch is None:
            raise ValueError("Detected Nash curve is missing its Nash point data.")
        if not np.isfinite(payoff_nash):
            continue
        branch_name = str(branch)
        ax.scatter(
            [float(E_nash)],
            [float(payoff_nash)],
            s=marker_size,
            marker="X",
            facecolors=BRANCH_COLORS.get(branch_name, DEFAULT_BRANCH_COLOR),
            edgecolors=EQUILIBRIUM_MARKER_EDGE_COLOR,
            linewidths=EQUILIBRIUM_MARKER_EDGE_WIDTH,
            label=f"Nash point ({BRANCH_DISPLAY.get(branch_name, branch_name)})",
            zorder=5,
        )


def _finish_payoff_plot(ax: Any, title: str) -> None:
    ax.set_xlabel(r"$E$")
    ax.set_ylabel("Leader payoff")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle=":", linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, loc="best")
    ax.figure.tight_layout()


def plot_combined_payoff_vs_e(
    E: Any,
    payoff: Any,
    branch: Any,
    equilibrium: Any,
    nash_payoff_curves: Sequence[Mapping[str, Any]],
    *,
    E_stackelberg: float | None = None,
    payoff_stackelberg: float | None = None,
    branch_stackelberg: str | None = None,
    line_plot: bool = False,
    show_markers: bool = True,
    marker_size: float = EQUILIBRIUM_MARKER_SIZE,
    nash_linestyle: str = ":",
    ax: Any = None,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Stackelberg and Nash payoff vs E",
) -> tuple[Any, Any]:
    """Plot all species-BR and fixed-trait leader payoffs on one axis."""
    series = prepare_stackelberg_payoff_series(E, payoff, branch, equilibrium)
    with mpl.rc_context(PUBLICATION_RC):
        fig, ax = _figure_and_axis(ax, figsize)
        _plot_stackelberg_series(
            ax,
            series,
            E_stackelberg=E_stackelberg,
            payoff_stackelberg=payoff_stackelberg,
            branch_stackelberg=branch_stackelberg,
            line_plot=line_plot,
            show_marker=show_markers,
            marker_size=marker_size,
        )
        _plot_nash_series(
            ax,
            nash_payoff_curves,
            show_markers=show_markers,
            marker_size=marker_size,
            linestyle=nash_linestyle,
        )
        _finish_payoff_plot(ax, title)
    return fig, ax
