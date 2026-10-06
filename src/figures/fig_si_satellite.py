"""SI figure: satellite-era replication of the decomposition.

The headline contrast uses the pre-satellite window (1940-59) as base. Because the
pre-1979 reanalysis moisture is less constrained, we repeat the entire three-term
decomposition + FDR-controlled bootstrap using only the two fully observed windows,
1980-99 (base) and 2015-24. The dynamic-term SIGN is preserved for Western Europe,
the Amazon outflow and southeastern South America (including the negative
southeastern-South-America thermodynamic term), but the shorter contrast loses the
statistical power to retain FDR significance -- documenting honestly that the
full-record contrast is needed, while the qualitative result does not depend on the
pre-satellite data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.analysis.decomposition import compute_pair
from src.figures._style import apply_nature_style, COL_DOUBLE_IN


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    apply_nature_style()
    rows = compute_pair("modern", "recent", block=1)
    rr = sorted(rows, key=lambda r: r["d_dyn"])
    names = [r["region"] for r in rr]; yy = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.42))
    th = np.array([r["d_thermo"] for r in rr]); dy = np.array([r["d_dyn"] for r in rr])
    dy_err = np.array([[r["d_dyn"] - r["d_dyn_ci95"][0], r["d_dyn_ci95"][1] - r["d_dyn"]] for r in rr]).T
    ax.barh(yy - 0.2, th, height=0.4, color="#e67e22", label="thermodynamic")
    ax.barh(yy + 0.2, dy, height=0.4, color="#2980b9", label="dynamic")
    ax.errorbar(dy, yy + 0.2, xerr=dy_err, fmt="none", ecolor="#16415f", lw=0.6, capsize=1.5)
    for i, r in enumerate(rr):
        if r["dyn_robust90"]:
            ax.text(max(r["d_dyn_ci95"][1], 0) + 0.2, i + 0.2,
                    "*" if r["dyn_robust"] else "†", va="center", fontsize=9, color="#16415f")
    ax.axvline(0, color="0.4", lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("contribution to IVT change, 1980–99 → 2015–24 (kg m$^{-1}$ s$^{-1}$)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_si_satellite.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
