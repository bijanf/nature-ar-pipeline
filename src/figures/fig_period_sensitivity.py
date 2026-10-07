"""Period-sensitivity figure: the wind-change term as a function of period length
and base-period start, one panel per corridor.

Reads the JSON with the sensitivity matrix (rows: L, base_start, corridor, wind,
wind_pfdr, total). Lines show the wind-change term (blue) and the total change
(grey) against the period length for each base start; filled markers mark wind
terms that stay significant after false-discovery control (p < 0.05).

Usage: python -m src.figures.fig_period_sensitivity RESULTS.json OUT.pdf
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.figures.fig_decomp_terms import COL, MM, SHORT, _panel_label, _style  # noqa: E402

ORDER = ["Amazon outflow", "SE South America", "Western Europe", "SE US / Gulf",
         "US West Coast", "East Asia", "South Africa", "SE Australia / NZ"]
MARK = {1940: "o", 1945: "s", 1950: "^"}
SHADE = {1940: 1.0, 1945: 0.7, 1950: 0.45}


def draw(res: dict, out: str):
    _style()
    rows = res["matrix"]
    fig, axs = plt.subplots(2, 4, figsize=(183 * MM, 85 * MM), sharex=True)
    for ax, name, letter in zip(axs.flat, ORDER, "abcdefgh"):
        sub = [r for r in rows if r["corridor"] == name]
        for b in sorted({r["base_start"] for r in sub}):
            s = sorted([r for r in sub if r["base_start"] == b], key=lambda r: r["L"])
            L = np.array([r["L"] for r in s])
            wind = np.array([r["wind"] for r in s])
            tot = np.array([r["total"] for r in s])
            sig = np.array([r["wind_pfdr"] < 0.05 for r in s])
            ax.plot(L, tot, color="0.55", alpha=SHADE[b], linewidth=0.8, marker=MARK[b], markersize=2.5,
                    markerfacecolor="none", label=f"total, base from {b}")
            ax.plot(L, wind, color=COL["wind"], alpha=SHADE[b], linewidth=1.0, marker=MARK[b],
                    markersize=3, markerfacecolor="none", label=f"wind change, base from {b}")
            ax.scatter(L[sig], wind[sig], s=14, color=COL["wind"], alpha=SHADE[b], marker=MARK[b], zorder=4)
        ax.axhline(0, color="0.3", linewidth=0.5)
        ax.text(0.03, 0.95, SHORT[name], transform=ax.transAxes, va="top", ha="left", fontsize=6.5)
        _panel_label(ax, letter)
        ax.set_xticks([10, 20, 30, 40])
    for ax in axs[1]:
        ax.set_xlabel("period length (years)")
    for ax in axs[:, 0]:
        ax.set_ylabel("change (kg m$^{-1}$ s$^{-1}$)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=5.5)
    fig.subplots_adjust(hspace=0.35, wspace=0.35, bottom=0.22)
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
