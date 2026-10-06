"""Five-window evolution of the thermodynamic and dynamic terms.

The headline contrast (Figure 2) compares only the first and last windows. To show
it is not an endpoint artefact, we recompute the exact three-term decomposition at
ALL five ERA5 windows relative to the 1940-1959 base: 1940-59, 1960-79, 1980-99,
2000-14, 2015-24. For the circulation-driven corridors the dynamic term grows
progressively across the record, and for southeastern South America the
thermodynamic term is negative in every window -- so the result is a sustained
feature of the observed evolution, not a two-window coincidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.analysis.decomposition import compute_5window
from src.figures._style import apply_nature_style, COL_DOUBLE_IN
from src.figures.fig_global_ar_regions import _REGIONS

_DYN = {"Western Europe", "Amazon outflow", "East Asia", "SE South America"}
# decade-midpoint x positions for the five windows
_XMID = [1949.5, 1969.5, 1989.5, 2007, 2019.5]


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    apply_nature_style()
    evo = compute_5window()

    fig, axes = plt.subplots(2, 4, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.5),
                             sharex=True)
    for ax, (name, *_ ) in zip(axes.flat, _REGIONS):
        s = evo[name]
        th = [w["d_thermo"] for w in s]
        dy = [w["d_dyn"] for w in s]
        ax.axhline(0, color="0.6", lw=0.5)
        ax.plot(_XMID, th, "-o", ms=3, lw=1.0, color="#e67e22", label="thermo")
        ax.plot(_XMID, dy, "-s", ms=3, lw=1.0, color="#2980b9", label="dynamic")
        emph = name in _DYN
        ax.set_title(name, fontsize=7.5, pad=2,
                     fontweight="bold" if emph else "normal")
        ax.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("ΔIVT vs 1940–59\n(kg m$^{-1}$ s$^{-1}$)", fontsize=7.5)
    for ax in axes[1, :]:
        ax.set_xlabel("window mid-year", fontsize=7.5)
        ax.set_xticks([1950, 1980, 2010])
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_5window_evolution.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
