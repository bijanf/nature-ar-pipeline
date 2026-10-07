"""Headline figure: decomposition of the corridor transport change into terms.

Reads the JSON written by the decomposition (window_summary rows plus trend
tables) and draws three panels:
  a  per corridor, the moisture-change, wind-change, covariance and transient
     terms with 95 % bootstrap intervals; the total change as a hollow bar;
     significant wind terms (FDR-adjusted p < 0.05) marked;
  b  the moisture-change term split into its Clausius-Clapeyron and
     relative-humidity parts;
  c  trends of the yearly wind-change contribution over 1940-2024 and 1979-2024,
     with significant trends (Hamed-Rao p < 0.05) marked.

Usage: python -m src.figures.fig_decomp_terms RESULTS.json OUT.pdf [--pair headline]
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

MM = 1 / 25.4
SHORT = {"Western Europe": "W Europe", "Amazon outflow": "Amazon", "East Asia": "E Asia",
         "SE South America": "SE S America", "South Africa": "S Africa", "SE US / Gulf": "SE US/Gulf",
         "US West Coast": "US W Coast", "SE Australia / NZ": "SE Aus/NZ"}
# Okabe-Ito palette (colour-blind safe); hatching keeps the series apart in black and white
COL = {"moist": "#E69F00", "wind": "#0072B2", "cov": "#999999", "trans": "#CC79A7",
       "cc": "#D55E00", "rh": "#56B4E9"}
HATCH = {"moist": "", "wind": "///", "cov": "", "trans": "xx", "cc": "", "rh": "..."}
LABEL = {"moist": "moisture change", "wind": "wind change", "cov": "covariance", "trans": "transient",
         "cc": "fixed relative humidity (Clausius–Clapeyron)", "rh": "relative-humidity change"}


def _style():
    fam = [f for f in ("Arial", "Helvetica", "Liberation Sans", "Arimo", "DejaVu Sans")
           if any(f == x.name for x in font_manager.fontManager.ttflist)]
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": fam or ["DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.4,
    })


def _panel_label(ax, letter):
    ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")


def draw(res: dict, out: str, pair: str = "headline"):
    _style()
    rows = sorted(res[pair], key=lambda r: -r["wind"])
    names = [r["corridor"] for r in rows]
    x = np.arange(len(rows))
    fig = plt.figure(figsize=(183 * MM, 120 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.55, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1])

    # a: four terms and the total
    terms = ["moist", "wind", "cov", "trans"]
    w = 0.19
    ax.bar(x, [r["total"] for r in rows], width=0.86, facecolor="none", edgecolor="0.25",
           linewidth=0.7, label="total change", zorder=1)
    for i, t in enumerate(terms):
        vals = np.array([r[t] for r in rows])
        ci = np.array([r.get(f"{t}_ci95", [np.nan, np.nan]) for r in rows])
        err = np.abs(ci.T - vals) if np.isfinite(ci).all() else None
        ax.bar(x + (i - 1.5) * w, vals, width=w, color=COL[t], hatch=HATCH[t], edgecolor="white",
               linewidth=0.3, label=LABEL[t], zorder=2,
               yerr=err, error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "0.2"})
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))
    for i, r in enumerate(rows):
        if r.get("wind_pfdr", 1) < 0.05:
            top = max(r["wind_ci95"][1], r["total"], 0)
            ax.text(x[i] - 0.5 * w, top + 0.02 * (hi - lo), "*", ha="center",
                    va="bottom", fontsize=9, color=COL["wind"])
    ax.axhline(0, color="0.3", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[n] for n in names], rotation=0)
    ax.set_ylabel("change in transport along corridor path\n(kg m$^{-1}$ s$^{-1}$)")
    ax.legend(ncol=3, frameon=False, loc="upper right", handlelength=1.6)
    _panel_label(ax, "a")

    # b: Clausius-Clapeyron vs relative-humidity parts of the moisture term
    if "moist_cc" in rows[0]:
        cc = np.array([r["moist_cc"] for r in rows])
        rh = np.array([r["moist_rh"] for r in rows])
        axb.bar(x - 0.2, cc, width=0.4, color=COL["cc"], label=LABEL["cc"], edgecolor="white", linewidth=0.3)
        axb.bar(x + 0.2, rh, width=0.4, color=COL["rh"], hatch=HATCH["rh"], label=LABEL["rh"],
                edgecolor="white", linewidth=0.3)
        axb.axhline(0, color="0.3", linewidth=0.5)
        axb.set_xticks(x)
        axb.set_xticklabels([SHORT[n] for n in names], rotation=45, ha="right")
        axb.set_ylabel("moisture-change term\n(kg m$^{-1}$ s$^{-1}$)")
        lo, hi = axb.get_ylim()
        axb.set_ylim(lo, hi + 0.35 * (hi - lo))  # headroom so the legend clears the bars
        axb.legend(frameon=False, loc="upper left", fontsize=5.5, ncol=1)
    _panel_label(axb, "b")

    # c: trends of the yearly wind-change contribution
    tf, ts = res.get("trends_full", {}), res.get("trends_sat", {})
    if tf:
        full = np.array([tf[n]["wind"]["pct_per_decade"] for n in names])
        sat = np.array([ts[n]["wind"]["pct_per_decade"] for n in names])
        pf = np.array([tf[n]["wind"]["p"] for n in names])
        psat = np.array([ts[n]["wind"]["p"] for n in names])
        axc.bar(x - 0.2, full, width=0.4, color=COL["wind"], label="1940–2024", edgecolor="white", linewidth=0.3)
        axc.bar(x + 0.2, sat, width=0.4, color=COL["wind"], alpha=0.45, hatch="///",
                label="1979–2024", edgecolor="white", linewidth=0.3)
        top = axc.get_ylim()[1]
        for i in range(len(names)):
            if pf[i] < 0.05:
                axc.text(x[i] - 0.2, max(full[i], 0) + 0.03 * top, "*", ha="center", va="bottom", fontsize=9)
            if psat[i] < 0.05:
                axc.text(x[i] + 0.2, max(sat[i], 0) + 0.03 * top, "*", ha="center", va="bottom", fontsize=9)
        axc.axhline(0, color="0.3", linewidth=0.5)
        axc.set_xticks(x)
        axc.set_xticklabels([SHORT[n] for n in names], rotation=45, ha="right")
        axc.set_ylabel("trend of wind-change contribution\n(% of mean transport per decade)")
        lo, hi = axc.get_ylim()
        axc.set_ylim(lo, hi + 0.25 * (hi - lo))
        axc.legend(frameon=False, loc="upper right")
    _panel_label(axc, "c")
    fig.savefig(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--pair", default="headline")
    a = ap.parse_args()
    draw(json.load(open(a.results)), a.out, a.pair)


if __name__ == "__main__":
    main()
