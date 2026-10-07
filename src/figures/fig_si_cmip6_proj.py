"""Supplement: end-of-century CMIP6 decomposition against the observed change, per kelvin.

  a  wind-change term per kelvin of global warming: each single-member model
     (1995-2014 to 2081-2100, SSP5-8.5) as a dot, the multi-model median as a bar,
     and the observed 1940-1969 to 1995-2024 change per kelvin of ERA5 global
     warming as a diamond;
  b  the same for the moisture-change term;
  c  the same for the total change.

Usage: python -m src.figures.fig_si_cmip6_proj cmip6_projection.json fullcolumn_levels.json DT_OBS OUT.pdf
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.figures.fig_cmip6_detection import OBS  # noqa: E402
from src.figures.fig_decomp_terms import MM, SHORT, _panel_label, _style  # noqa: E402
from src.figures.fig_period_sensitivity import ORDER  # noqa: E402


def draw(proj, obs_levels, dT_obs, out):
    _style()
    # models whose corridor fields are entirely masked in the store give zeros everywhere; drop them
    proj = {k: v for k, v in proj.items() if any(abs(v[n]["total"]) > 0 for n in ORDER if n in v)}
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(3, 1, figsize=(183 * MM, 140 * MM), sharex=True)
    x = np.arange(len(ORDER))
    for ax, term, letter, lab in zip(axs, ("wind", "moist", "total"), "abc",
                                     ("wind-change term", "moisture-change term", "total change")):
        for i, n in enumerate(ORDER):
            vals = np.array([m[n]["per_K"][term] for m in proj.values() if n in m])
            ax.scatter(x[i] + rng.uniform(-0.22, 0.22, vals.size), vals, s=9, color="0.45", linewidths=0, zorder=2)
            med = np.median(vals)
            ax.plot([x[i] - 0.3, x[i] + 0.3], [med, med], color="black", linewidth=1.4, zorder=3)
            o = next(r for r in obs_levels["headline"] if r["corridor"] == n)[term] / dT_obs
            ax.scatter([x[i]], [o], marker="D", s=24, color=OBS, edgecolor="black", linewidths=0.4, zorder=5)
        ax.axhline(0, color="0.3", linewidth=0.5)
        ax.set_ylabel(f"{lab}\n(kg m$^{{-1}}$ s$^{{-1}}$ K$^{{-1}}$)")
        _panel_label(ax, letter)
    axs[0].scatter([], [], s=9, color="0.45", label=f"CMIP6 models, SSP5-8.5 ({len(proj)})")
    axs[0].plot([], [], color="black", linewidth=1.4, label="multi-model median")
    axs[0].scatter([], [], marker="D", s=24, color=OBS, edgecolor="black", linewidths=0.4, label="ERA5, 1940–69 to 1995–2024")
    axs[0].legend(frameon=False, loc="upper right", fontsize=5.5, ncol=3)
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels([SHORT[n] for n in ORDER], rotation=45, ha="right")
    fig.subplots_adjust(hspace=0.25)
    fig.savefig(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projection")
    ap.add_argument("obs_levels")
    ap.add_argument("dT_obs", type=float)
    ap.add_argument("out")
    a = ap.parse_args()
    draw(json.load(open(a.projection)), json.load(open(a.obs_levels)), a.dT_obs, a.out)


if __name__ == "__main__":
    main()
