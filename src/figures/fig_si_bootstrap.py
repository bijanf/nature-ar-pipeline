"""SI figure: year-block bootstrap distributions of the dynamic term.

For each corridor we show the full distribution of the dynamic-term bootstrap
replicates (1000 whole-year resamples of the 1940-59 and 2015-24 windows), the
point estimate and the 95% interval, and shade the fraction on the wrong side of
zero -- the one-sided bootstrap p-value before false-discovery adjustment. This
exposes the inference the headline figure summarises, rather than reporting only
the interval bounds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.analysis.decomposition import _series_cache, _three_term
from src.figures._style import apply_nature_style, COL_DOUBLE_IN
from src.figures.fig_global_ar_regions import _REGIONS

_BOOT = 1000
_DYN = {"Western Europe", "Amazon outflow", "East Asia", "SE South America"}


def _replicates(cache, name):
    ivt0, iwv0 = cache["presat"][name]
    ivt1, iwv1 = cache["recent"][name]
    rng = np.random.default_rng(0)
    n0, n1 = len(ivt0), len(ivt1)
    out = np.empty(_BOOT)
    for b in range(_BOOT):
        i0 = rng.integers(0, n0, n0); i1 = rng.integers(0, n1, n1)
        _, _, dy, _ = _three_term(ivt0[i0], iwv0[i0], ivt1[i1], iwv1[i1])
        out[b] = dy
    _, _, dy0, _ = _three_term(ivt0, iwv0, ivt1, iwv1)
    return out, dy0


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    apply_nature_style()
    cache = _series_cache()
    fig, axes = plt.subplots(2, 4, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.48))
    for ax, (name, *_ ) in zip(axes.flat, _REGIONS):
        rep, dy0 = _replicates(cache, name)
        col = "#2980b9" if name in _DYN else "#7a4410"
        ax.hist(rep, bins=30, color=col, alpha=0.55, density=True)
        lo, hi = np.percentile(rep, [2.5, 97.5])
        ax.axvline(0, color="0.3", lw=0.8)
        ax.axvline(dy0, color="#c0392b", lw=1.1)
        ax.axvspan(lo, hi, color=col, alpha=0.12)
        p = np.mean(rep <= 0) if dy0 >= 0 else np.mean(rep >= 0)
        ax.set_title(f"{name}\n$p$={p:.3f}", fontsize=7, pad=2,
                     fontweight="bold" if name in _DYN else "normal")
        ax.tick_params(labelsize=6.5); ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("dynamic term (kg m$^{-1}$ s$^{-1}$)", fontsize=7)
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_si_bootstrap.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
