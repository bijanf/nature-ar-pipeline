"""CMIP6 figure: observed terms against single-model large ensembles.

Reads results_esd/cmip6_members.json and draws, per corridor,
  a  the wind-change term: every member of each large ensemble as a dot, the
     ensemble mean as a bar, the 5-95 % member range as a line; the models with
     one member as grey dots; the observation (ERA5 processed like the models) as
     a diamond;
  b  the same for the moisture-change term;
  c  the standardised departure z = (obs - ensemble mean) / ensemble sd of the
     wind term, with the +-2 reference lines.

Usage: python -m src.figures.fig_cmip6_detection cmip6_members.json OUT.pdf [--pair headline]
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

LARGE = {"CanESM5": "#0072B2", "MIROC6": "#009E73", "ACCESS-ESM1-5": "#CC79A7", "EC-Earth3": "#E69F00"}
OBS = "#D55E00"


def _collect(res: dict, pair: str, term: str):
    """{corridor: {model: array of member values}} and the observed value."""
    vals = {n: {} for n in ORDER}
    for key, m in res["members"].items():
        model = key.split("/")[0]
        for n in ORDER:
            if n in m and pair in m[n]:
                vals[n].setdefault(model, []).append(m[n][pair][term])
    obs = {n: res["obs_cmiplevels"][n][pair][term] for n in ORDER}
    return vals, obs


def _strip(ax, vals, obs, rng, ylabel):
    x = np.arange(len(ORDER))
    models = list(LARGE)
    k = len(models) + 1
    w = 0.8 / k
    for i, n in enumerate(ORDER):
        singles = [v[0] for mdl, v in vals[n].items() if mdl not in LARGE and len(v) == 1]
        for j, mdl in enumerate(models):
            v = np.array(vals[n].get(mdl, []))
            if v.size == 0:
                continue
            xc = x[i] - 0.4 + (j + 0.5) * w
            ax.scatter(xc + rng.uniform(-0.3, 0.3, v.size) * w, v, s=3, color=LARGE[mdl], alpha=0.5,
                       linewidths=0, zorder=2)
            lo, hi = np.percentile(v, [5, 95])
            ax.plot([xc, xc], [lo, hi], color=LARGE[mdl], linewidth=0.8, zorder=3)
            ax.plot([xc - 0.45 * w, xc + 0.45 * w], [v.mean(), v.mean()], color=LARGE[mdl], linewidth=1.4, zorder=4)
        if singles:
            xc = x[i] - 0.4 + (len(models) + 0.5) * w
            ax.scatter(xc + rng.uniform(-0.3, 0.3, len(singles)) * w, singles, s=5, color="0.45",
                       linewidths=0, zorder=2)
        ax.scatter([x[i]], [obs[n]], marker="D", s=22, color=OBS, edgecolor="black", linewidths=0.4, zorder=6)
    ax.axhline(0, color="0.3", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[n] for n in ORDER], rotation=45, ha="right")
    ax.set_ylabel(ylabel)


def draw(res: dict, out: str, pair: str = "headline"):
    _style()
    rng = np.random.default_rng(0)
    # a model counts as a large ensemble only with at least five usable members
    counts = {}
    for key in res["members"]:
        counts[key.split("/")[0]] = counts.get(key.split("/")[0], 0) + 1
    for mdl in list(LARGE):
        if counts.get(mdl, 0) < 5:
            LARGE.pop(mdl)
    fig, axs = plt.subplots(3, 1, figsize=(183 * MM, 150 * MM), gridspec_kw={"height_ratios": [1.3, 1.3, 1]})
    vals_w, obs_w = _collect(res, pair, "wind")
    vals_m, obs_m = _collect(res, pair, "moist")
    _strip(axs[0], vals_w, obs_w, rng, "wind-change term\n(kg m$^{-1}$ s$^{-1}$)")
    _strip(axs[1], vals_m, obs_m, rng, "moisture-change term\n(kg m$^{-1}$ s$^{-1}$)")
    for mdl, c in LARGE.items():
        n_mem = max((len(vals_w[n].get(mdl, [])) for n in ORDER), default=0)
        axs[0].scatter([], [], color=c, s=8, label=f"{mdl} ({n_mem} members)")
    axs[0].scatter([], [], color="0.45", s=8, label="other models (1 member)")
    axs[0].scatter([], [], marker="D", color=OBS, edgecolor="black", linewidths=0.4, s=22, label="ERA5")
    axs[0].legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.5)
    # c: z-scores of the wind term
    x = np.arange(len(ORDER))
    for j, (mdl, c) in enumerate(LARGE.items()):
        z = []
        for n in ORDER:
            v = np.array(vals_w[n].get(mdl, []))
            z.append((obs_w[n] - v.mean()) / v.std(ddof=1) if v.size > 2 else np.nan)
        axs[2].scatter(x + (j - 1.5) * 0.18, z, color=c, s=14, zorder=3)
    for yv in (-2, 2):
        axs[2].axhline(yv, color="0.5", linewidth=0.5, linestyle="--")
    axs[2].axhline(0, color="0.3", linewidth=0.5)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels([SHORT[n] for n in ORDER], rotation=45, ha="right")
    axs[2].set_ylabel("standardised departure\nof observed wind term", labelpad=2)
    for ax, letter in zip(axs, "abc"):
        ax.text(-0.1, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    fig.subplots_adjust(hspace=0.6, left=0.11)
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
