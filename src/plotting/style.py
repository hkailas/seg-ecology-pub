"""Shared labels, colours, and Matplotlib settings for result plots."""

from collections.abc import Iterable


BRANCH_ORDER = ("branch_0", "branch_1", "branch_2", "branch_3", "extinct")

BRANCH_DISPLAY = {
    "branch_0": "1 species",
    "branch_1": "2 species",
    "branch_2": "3 species",
    "branch_3": "4 species",
    "extinct": "Extinct",
}

BRANCH_COLORS = {
    "branch_0": "#2ca02c",
    "branch_1": "#ff7f0e",
    "branch_2": "#e377c2",
    "branch_3": "#0072B2",
    "extinct": "#7f7f7f",
}

DEFAULT_BRANCH_COLOR = "#888888"

EQUILIBRIUM_MARKER_SIZE = 110
EQUILIBRIUM_MARKER_EDGE_COLOR = "black"
EQUILIBRIUM_MARKER_EDGE_WIDTH = 1.5

PUBLICATION_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def ordered_branches(branches: Iterable[object]) -> list[str]:
    """Return known branches first and any additional labels alphabetically."""
    available = {str(branch) for branch in branches}
    known = [branch for branch in BRANCH_ORDER if branch in available]
    other = sorted(available - set(BRANCH_ORDER))
    return known + other
