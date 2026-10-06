"""Continuous transport-efficiency (Vhat) trends — the dynamic term, year by year.

Vhat(t) = IVT(t) / IWV(t) is the moisture-weighted transport efficiency: it is the
corridor's moisture transport with the column-moisture amount divided out, so a
trend in Vhat is the circulation's contribution to the intensification, isolated
from the thermodynamic (moistening) term. We plot Vhat for each corridor over the
continuous 1940-2024 record (five ERA5 windows joined), with a Theil-Sen slope and
a Hamed-Rao autocorrelation-corrected Mann-Kendall p-value. The four corridors the
two-window bootstrap flags as dynamically driven are exactly those whose efficiency
rises significantly here, so the dynamic term is a continuous secular signal, not a
two-window artefact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.analysis.mode_attribution import corridor_records
from src.analysis.trend_stats import hamed_rao_mk
from src.figures._style import apply_nature_style, COL_DOUBLE_IN
from src.figures.fig_global_ar_regions import _REGIONS

# corridors robustly dynamic by the two-window bootstrap (Figure 2)
_DYN = {"Western Europe", "Amazon outflow", "East Asia", "SE South America"}


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    from scipy.stats import theilslopes
    apply_nature_style()
    recs = corridor_records()

    fig, axes = plt.subplots(2, 4, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.48),
                             sharex=True)
    for ax, (name, *_ ) in zip(axes.flat, _REGIONS):
        r = recs[name]
        yr, vhat = r["year"], r["vhat"]
        tr = hamed_rao_mk(vhat, yr.astype(float))
        sl, b, _, _ = theilslopes(vhat, yr)
        pct_dec = 100 * sl * 10 / vhat.mean()
        col = "#2980b9" if name in _DYN else "#7a4410"
        ax.plot(yr, vhat, lw=0.6, color=col, alpha=0.5)
        # 7-year running mean
        if len(vhat) >= 7:
            rm = np.convolve(vhat, np.ones(7) / 7, mode="valid")
            ax.plot(yr[3:-3], rm, lw=1.3, color=col)
        ax.plot(yr, b + sl * yr, ls="--", lw=0.8, color="0.25")
        star = "*" if tr.p < 0.05 else ("†" if tr.p < 0.10 else "")
        ax.set_title(f"{name}\n{pct_dec:+.2f}%/dec{star}  p={tr.p:.2f}",
                     fontsize=7, pad=2)
        ax.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\hat{V}$ = IVT/IWV" + "\n(m s$^{-1}$)", fontsize=7.5)
    for ax in axes[1, :]:
        ax.set_xlabel("year", fontsize=7.5)
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_efficiency_trends.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
