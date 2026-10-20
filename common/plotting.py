"""Shared matplotlib style for report figures (all tasks) and the figure saver (report/figures/<name>.png)."""
from __future__ import annotations

import matplotlib.pyplot as plt

from common.data import repo_path

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })


style()


def save(fig, name):
    if fig is None:
        print(f"(skipped {name}: results not available yet)")
        return None
    out = repo_path("report/figures") / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    return out
