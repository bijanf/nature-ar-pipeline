"""SI figure: dynamic-term confidence intervals under i.i.d.-year and moving-block
bootstraps.

The headline uses an i.i.d. whole-year bootstrap. Because corridor series carry some
serial dependence, we also report a moving-block bootstrap with block lengths of two
and three years, which preserves short-range temporal structure at the cost of fewer
independent blocks (hence wider, more conservative intervals over the short windows).
The figure shows the dynamic term with its 95% interval under each scheme: the robust
core (Western Europe, Amazon) keeps an interval clear of zero even under the most
conservative block length, while the marginal corridors' intervals touch zero as the
block grows -- which is exactly why we class them as supported-but-marginal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.analysis.decomposition import _series_cache, compute_pair
from src.figures._style import apply_nature_style, COL_DOUBLE_IN

_BLOCKS = [1, 2, 3]
_COL = {1: "#16415f", 2: "#2980b9", 3: "#7fb3d5"}


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    apply_nature_style()
    cache = _series_cache()
    res = {b: compute_pair("presat", "recent", block=b, cache=cache) for b in _BLOCKS}
    # order corridors by the i.i.d. dynamic term
    order = sorted(res[1], key=lambda r: r["d_dyn"])
    names = [r["region"] for r in order]
    byb = {b: {r["region"]: r for r in res[b]} for b in _BLOCKS}
    yy = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.46))
    off = {1: 0.24, 2: 0.0, 3: -0.24}
    for b in _BLOCKS:
        d = [byb[b][n]["d_dyn"] for n in names]
        ci = np.array([byb[b][n]["d_dyn_ci95"] for n in names])
        err = np.array([[d[i] - ci[i, 0], ci[i, 1] - d[i]] for i in range(len(names))]).T
        ax.errorbar(d, yy + off[b], xerr=err, fmt="o", ms=3, color=_COL[b],
                    lw=0.8, capsize=1.5, label=f"block = {b} yr" if b > 1 else "i.i.d. years")
    ax.axvline(0, color="0.4", lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("dynamic term with 95% bootstrap interval (kg m$^{-1}$ s$^{-1}$)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_si_blockboot.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
