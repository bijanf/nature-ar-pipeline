"""Yearly contributions figure: one panel per corridor, 1940-2024.

Thin lines: yearly moisture-change (orange) and wind-change (blue) contributions
relative to the 1940-2024 climatology, with the transient contribution (pink)
where it matters; bold lines: 9-year running means; dashed: Theil-Sen trend of
the wind contribution over 1940-2024 (drawn only where p < 0.05). The two
headline periods are shaded.

Usage: python -m src.figures.fig_yearly yearly_headline.npz fullcolumn_headline.json OUT.pdf
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import theilslopes  # noqa: E402

from src.figures.fig_decomp_terms import COL, MM, SHORT, _panel_label, _style  # noqa: E402
from src.figures.fig_period_sensitivity import ORDER  # noqa: E402

YEARS = np.arange(1940, 2025)
W0, W1 = (1940, 1969), (1995, 2024)


def _key(name):
    return "".join(c for c in name.lower() if c.isalnum())


def _run(x, w=9):
    k = np.ones(w) / w
    out = np.full_like(x, np.nan)
    out[w // 2: -(w // 2)] = np.convolve(x, k, mode="valid")
    return out


def draw(npz, res, out):
    _style()
    fig, axs = plt.subplots(4, 2, figsize=(183 * MM, 170 * MM), sharex=True)
    for ax, name, letter in zip(axs.T.flat, ORDER, "abcdefgh"):
        k = _key(name)
        m, w, tr = npz[f"{k}__moist"], npz[f"{k}__wind"], npz[f"{k}__trans"]
        for (y0, y1) in (W0, W1):
            ax.axvspan(y0 - 0.5, y1 + 0.5, color="0.92", zorder=0)
        ax.axhline(0, color="0.3", linewidth=0.5)
        ax.plot(YEARS, m, color=COL["moist"], linewidth=0.5, alpha=0.6)
        ax.plot(YEARS, w, color=COL["wind"], linewidth=0.5, alpha=0.6)
        ax.plot(YEARS, _run(m), color=COL["moist"], linewidth=1.4, label="moisture change")
        ax.plot(YEARS, _run(w), color=COL["wind"], linewidth=1.4, label="wind change")
        if np.abs(tr).max() > 0.5:
            ax.plot(YEARS, _run(tr), color=COL["trans"], linewidth=1.0, label="transient")
        t = res["trends_full"][name]["wind"]
        if t["p"] < 0.05:
            slope, intercept, *_ = theilslopes(w, YEARS)
            ax.plot(YEARS, intercept + slope * YEARS, color=COL["wind"], linewidth=0.8, linestyle="--")
            ax.text(0.98, 0.04, f"wind trend {t['pct_per_decade']:+.1f} % per decade", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=5.5, color=COL["wind"])
        ax.text(0.02, 0.95, SHORT[name], transform=ax.transAxes, va="top", ha="left", fontsize=6.5)
        _panel_label(ax, letter)
    for ax in axs[-1]:
        ax.set_xlabel("year")
        ax.set_xlim(1939, 2025)
    for ax in axs[:, 0]:
        ax.set_ylabel("contribution (kg m$^{-1}$ s$^{-1}$)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(hspace=0.3, wspace=0.25, bottom=0.09)
    fig.savefig(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("results")
    ap.add_argument("out")
    a = ap.parse_args()
    draw(np.load(a.npz), json.load(open(a.results)), a.out)


if __name__ == "__main__":
    main()
