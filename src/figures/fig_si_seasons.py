"""Supplement: seasonal decomposition terms per corridor.

One panel per corridor; for each season (DJF, MAM, JJA, SON) the moisture,
wind, covariance and transient terms with 95 % intervals and the total change as
a hollow bar; wind terms significant after false-discovery control are marked.

Usage: python -m src.figures.fig_si_seasons fullcolumn_seasons.json OUT.pdf
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.figures.fig_decomp_terms import COL, HATCH, LABEL, MM, SHORT, _panel_label, _style  # noqa: E402
from src.figures.fig_period_sensitivity import ORDER  # noqa: E402

SEASONS = ["DJF", "MAM", "JJA", "SON"]
TERMS = ["moist", "wind", "cov", "trans"]


def draw(res, out):
    _style()
    fig, axs = plt.subplots(2, 4, figsize=(183 * MM, 90 * MM), sharex=True)
    x = np.arange(len(SEASONS))
    w = 0.19
    for ax, name, letter in zip(axs.flat, ORDER, "abcdefgh"):
        rows = [next(r for r in res[s] if r["corridor"] == name) for s in SEASONS]
        ax.bar(x, [r["total"] for r in rows], width=0.86, facecolor="none", edgecolor="0.25", linewidth=0.7,
               label=LABEL.get("total", "total change"), zorder=1)
        for i, t in enumerate(TERMS):
            vals = np.array([r[t] for r in rows])
            ci = np.array([r[f"{t}_ci95"] for r in rows])
            ax.bar(x + (i - 1.5) * w, vals, width=w, color=COL[t], hatch=HATCH[t], edgecolor="white", linewidth=0.3,
                   label=LABEL[t], zorder=2, yerr=np.abs(ci.T - vals),
                   error_kw={"elinewidth": 0.5, "capsize": 1.2, "ecolor": "0.2"})
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.15 * (hi - lo))
        for i, r in enumerate(rows):
            if r["wind_pfdr"] < 0.05:
                ax.text(x[i] - 0.5 * w, max(r["wind_ci95"][1], r["total"], 0) + 0.02 * (hi - lo), "*",
                        ha="center", va="bottom", fontsize=8, color=COL["wind"])
        ax.axhline(0, color="0.3", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(SEASONS)
        ax.text(0.03, 0.95, SHORT[name], transform=ax.transAxes, va="top", ha="left", fontsize=6.5)
        _panel_label(ax, letter)
    for ax in axs[:, 0]:
        ax.set_ylabel("change (kg m$^{-1}$ s$^{-1}$)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(hspace=0.3, wspace=0.35, bottom=0.14)
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
