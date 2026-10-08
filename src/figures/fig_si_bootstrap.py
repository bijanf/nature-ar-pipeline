"""Supplement: bootstrap distributions of the wind-change term and block-length comparison.

  a-h  per corridor, the distribution of 1000 moving-block bootstrap replicates
       of the wind-change term for block lengths 1, 3 and 5 years (kernel density
       estimates), the point estimate (vertical line) and zero.

Usage: python -m src.figures.fig_si_bootstrap bootstrap_wind.npz OUT.pdf
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from src.figures.fig_decomp_terms import COL, MM, SHORT, _panel_label, _style  # noqa: E402
from src.figures.fig_period_sensitivity import ORDER  # noqa: E402

SHADE = {1: 0.35, 3: 0.65, 5: 1.0}
LS = {1: ":", 3: "--", 5: "-"}


def _key(name):
    return "".join(c for c in name.lower() if c.isalnum())


def draw(npz, out):
    _style()
    fig, axs = plt.subplots(2, 4, figsize=(183 * MM, 85 * MM))
    for ax, name, letter in zip(axs.flat, ORDER, "abcdefgh"):
        k = _key(name)
        point = npz[f"{k}__point"][0]
        allv = np.concatenate([npz[f"{k}__b{b}"][:, 0] for b in (1, 3, 5)])
        grid = np.linspace(allv.min() - 1, allv.max() + 1, 300)
        for b in (1, 3, 5):
            v = npz[f"{k}__b{b}"][:, 0]
            lo, hi = np.percentile(v, [2.5, 97.5])
            ax.plot(grid, gaussian_kde(v)(grid), color=COL["wind"], alpha=SHADE[b], linestyle=LS[b], linewidth=1.0,
                    label=f"block {b} yr" + (" (independent years)" if b == 1 else ""))
            ax.plot([lo, hi], [-0.004 * b] * 2, color=COL["wind"], alpha=SHADE[b], linewidth=1.2)
        ax.axvline(point, color="black", linewidth=0.8)
        ax.axvline(0, color="0.5", linewidth=0.5)
        ax.set_yticks([])
        ax.text(0.03, 0.95, SHORT[name], transform=ax.transAxes, va="top", ha="left", fontsize=6.5)
        _panel_label(ax, letter)
    fig.supxlabel("wind-change term (kg m$^{-1}$ s$^{-1}$)", fontsize=7, y=0.11)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(hspace=0.4, wspace=0.18, bottom=0.2)
    fig.savefig(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    a = ap.parse_args()
    draw(np.load(a.npz), a.out)


if __name__ == "__main__":
    main()
