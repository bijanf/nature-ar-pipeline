"""Climate-index figure.

  a  regression coefficients of the yearly wind-change contribution on the six
     detrended, standardised indices (annual), corridors by indices, in
     kg m-1 s-1 per standard deviation; cells significant at p < 0.05 are
     outlined;
  b  the same for the moisture-change contribution;
  c  change of the wind-change contribution between the two headline periods,
     raw and after removal of the index-congruent part;
  d  validation of the ERA5-derived indices against the published series
     (annual correlation over the common period).

Usage: python -m src.figures.fig_indices index_regression.json indices_validation.json OUT.pdf
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.figures.fig_decomp_terms import COL, MM, SHORT, _panel_label, _style  # noqa: E402
from src.figures.fig_period_sensitivity import ORDER  # noqa: E402

INDICES = ["rnino34", "pdo", "amv", "nao", "sam", "pna"]
INAME = {"rnino34": "rel. Niño3.4", "pdo": "PDO", "amv": "AMV", "nao": "NAO", "sam": "SAM", "pna": "PNA"}


def _heat(ax, res, term, vmax):
    mat = np.array([[res[n][term]["coef"][k] for k in INDICES] for n in ORDER])
    pv = np.array([[res[n][term]["p"][k] for k in INDICES] for n in ORDER])
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(mat[i, j]) > 0.6 * vmax else "black")
            if pv[i, j] < 0.05:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.2))
    ax.set_xticks(range(len(INDICES)))
    ax.set_xticklabels([INAME[k] for k in INDICES], rotation=45, ha="right")
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([SHORT[n] for n in ORDER])
    ax.tick_params(length=0)
    return im


def draw(reg: dict, val: dict, out: str):
    _style()
    res = reg["annual"]
    fig = plt.figure(figsize=(183 * MM, 125 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], wspace=0.55, hspace=0.6)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    vmax = max(abs(res[n][t]["coef"][k]) for n in ORDER for t in ("wind", "moist") for k in INDICES)
    vmax = float(np.ceil(vmax))
    _heat(axa, res, "wind", vmax)
    im = _heat(axb, res, "moist", vmax)
    cb = fig.colorbar(im, ax=[axa, axb], orientation="horizontal", fraction=0.05, pad=0.28, aspect=40)
    cb.set_label("regression coefficient (kg m$^{-1}$ s$^{-1}$ per standard deviation)")
    axa.set_title("wind-change contribution", fontsize=7, pad=3)
    axb.set_title("moisture-change contribution", fontsize=7, pad=3)
    _panel_label(axa, "a")
    _panel_label(axb, "b")
    # c: raw vs index-removed change of the wind contribution
    x = np.arange(len(ORDER))
    raw = [res[n]["wind"]["wind_diff_raw"] for n in ORDER]
    rem = [res[n]["wind"]["wind_diff_index_removed"] for n in ORDER]
    axc.bar(x - 0.2, raw, width=0.4, color=COL["wind"], label="raw", edgecolor="white", linewidth=0.3)
    axc.bar(x + 0.2, rem, width=0.4, color=COL["wind"], alpha=0.45, hatch="///", label="index-congruent part removed",
            edgecolor="white", linewidth=0.3)
    axc.axhline(0, color="0.3", linewidth=0.5)
    axc.set_xticks(x)
    axc.set_xticklabels([SHORT[n] for n in ORDER], rotation=45, ha="right")
    axc.set_ylabel("change of wind contribution,\n1940–69 to 1995–2024 (kg m$^{-1}$ s$^{-1}$)")
    lo, hi = axc.get_ylim()
    axc.set_ylim(lo, hi + 0.3 * (hi - lo))
    axc.legend(frameon=False, loc="upper right", fontsize=5.5)
    _panel_label(axc, "c")
    # d: validation
    r_ann = [val[k]["r_annual"] for k in INDICES]
    r_mon = [val[k]["r_monthly"] for k in INDICES]
    xi = np.arange(len(INDICES))
    axd.bar(xi - 0.2, r_ann, width=0.4, color="0.35", label="annual means", edgecolor="white", linewidth=0.3)
    axd.bar(xi + 0.2, r_mon, width=0.4, color="0.65", label="monthly", edgecolor="white", linewidth=0.3)
    for i, k in enumerate(INDICES):
        axd.text(xi[i], 0.02, val[k]["overlap"], rotation=90, ha="center", va="bottom", fontsize=4.5, color="white")
    axd.set_xticks(xi)
    axd.set_xticklabels([INAME[k] for k in INDICES], rotation=45, ha="right")
    axd.set_ylim(0, 1.12)
    axd.set_ylabel("correlation with published index")
    axd.legend(frameon=False, loc="upper right", fontsize=5.5, ncol=2)
    _panel_label(axd, "d")
    fig.savefig(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regression")
    ap.add_argument("validation")
    ap.add_argument("out")
    a = ap.parse_args()
    draw(json.load(open(a.regression)), json.load(open(a.validation)), a.out)


if __name__ == "__main__":
    main()
