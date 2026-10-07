"""Supplement: the moisture-weighted transport speed split against the direct terms.

  a  the "dynamic" term of the V-hat split (W0 dV-hat) against the wind-change
     term of the level-resolved split, per corridor, for the three period pairs;
  b  the "thermodynamic" term (V-hat0 dW) against the moisture-change term.
The 1:1 line is drawn; departures from it show where V-hat mixes a change of the
vertical moisture distribution into the apparent circulation term.

Usage: python -m src.figures.fig_si_vhat fullcolumn_headline.json OUT.pdf
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.figures.fig_decomp_terms import MM, SHORT, _panel_label, _style  # noqa: E402
from src.figures.fig_period_sensitivity import ORDER  # noqa: E402

PAIRS = {"headline": ("1940–69 vs 1995–2024", "o"), "original": ("1940–59 vs 2015–24", "s"),
         "satellite": ("1979–2001 vs 2002–24", "^")}
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999", "#000000"]


def draw(res, out):
    _style()
    fig, axs = plt.subplots(1, 2, figsize=(183 * MM, 80 * MM))
    for ax, (vi, direct, lab), letter in zip(axs, [(1, "wind", "wind-change term"), (0, "moist", "moisture-change term")], "ab"):
        allv = []
        for pair, (plab, mk) in PAIRS.items():
            for ci, n in enumerate(ORDER):
                r = next(x for x in res[pair] if x["corridor"] == n)
                xv, yv = r[direct], r["vhat_thermo_dyn_cov_total_base"][vi]
                allv += [xv, yv]
                ax.scatter(xv, yv, marker=mk, s=18, color=PAL[ci], edgecolor="black", linewidths=0.3,
                           label=SHORT[n] if pair == "headline" else None, zorder=3)
        lim = [min(allv) - 1, max(allv) + 1]
        ax.plot(lim, lim, color="0.4", linewidth=0.6, linestyle="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"{lab}, level-resolved split (kg m$^{{-1}}$ s$^{{-1}}$)")
        ax.set_ylabel(("$W_0\\,\\Delta\\hat V$" if vi == 1 else "$\\hat V_0\\,\\Delta W$") + " (kg m$^{-1}$ s$^{-1}$)")
        ax.set_aspect("equal")
        _panel_label(ax, letter)
    h, lab = axs[0].get_legend_handles_labels()
    for plab, mk in PAIRS.values():
        h.append(plt.Line2D([], [], marker=mk, color="0.3", linestyle="none", markersize=4))
        lab.append(plab)
    fig.legend(h, lab, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.04), fontsize=5.5)
    fig.subplots_adjust(wspace=0.35, bottom=0.3)
    fig.savefig(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    a = ap.parse_args()
    draw(json.load(open(a.results)), a.out)


if __name__ == "__main__":
    main()
