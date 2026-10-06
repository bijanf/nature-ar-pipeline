"""SI figure: time series of the leading circulation-mode principal components.

The six leading EOFs of the global annual 500 hPa height anomaly are the predictors
in the machine-learning attribution. Here we show their standardised principal-
component time series over 1940-2024 with a linear trend on each, demonstrating that
several modes carry a low-frequency secular component -- consistent with a dynamic
term dominated by a slow reorganisation of the circulation rather than by interannual
mode swings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.analysis.mode_attribution import circulation_modes
from src.figures._style import apply_nature_style, COL_DOUBLE_IN


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    apply_nature_style()
    m = circulation_modes(6)
    yr = m["year"]; pcs = m["pcs"]; ve = m["varexp"]
    fig, axes = plt.subplots(3, 2, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.62), sharex=True)
    for k, ax in enumerate(axes.flat):
        y = pcs[:, k]
        ax.axhline(0, color="0.7", lw=0.5)
        ax.plot(yr, y, lw=0.6, color="#34495e")
        sl, b = np.polyfit(yr, y, 1)
        ax.plot(yr, b + sl * yr, ls="--", lw=1.0, color="#c0392b")
        ax.set_title(f"PC{k+1} — {100*ve[k]:.0f}% variance  ({sl*10:+.2f}/dec)",
                     fontsize=7.5, pad=2)
        ax.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("year", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("standardised PC", fontsize=7.5)
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_si_pc_timeseries.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
