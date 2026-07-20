"""Shared chart style: a validated categorical palette on a clean light
surface, styled for a quant/analytics register rather than matplotlib
defaults."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fbfbfa"
INK = "#111318"
INK_2 = "#4b4f58"
MUTED = "#8b8f98"
GRID = "#e4e4e2"
BASELINE = "#c7c7c4"

# Fixed-order categorical set (validated: adjacent-pair CVD dE >= 8, normal
# vision floor >= 15 on this surface). Never cycle past what's needed.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948", "#eda100"]
BLUE, ORANGE, AQUA, VIOLET, RED, YELLOW = SERIES

DIVERGE_NEG, DIVERGE_MID, DIVERGE_POS = "#e34948", "#f0efec", "#2a78d6"


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "sans-serif"],
        "lines.linewidth": 2,
        "legend.frameon": False,
        "figure.dpi": 120,
    })


def diverging_cmap():
    return LinearSegmentedColormap.from_list(
        "diverge", [DIVERGE_NEG, DIVERGE_MID, DIVERGE_POS])
