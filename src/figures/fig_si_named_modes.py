"""SI figure: correlation of each corridor's transport efficiency with named
teleconnection modes (NAO, SAM, PNA, ENSO proxy).

Puts a recognised name on the data-driven modes of the machine-learning attribution:
Western Europe loads strongly on the NAO, the tropical/subtropical South American and
East Asian corridors on the ENSO proxy, and the Southern-Hemisphere corridors on the
SAM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.analysis.named_modes import regress
from src.figures._style import apply_nature_style, COL_DOUBLE_IN

_DYN = {"Western Europe", "Amazon outflow", "East Asia", "SE South America"}


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    apply_nature_style()
    names, rows = regress()
    M = np.array([[r["corr"][n][0] for n in names] for r in rows])
    P = np.array([[r["corr"][n][1] for n in names] for r in rows])
    regions = [r["region"] for r in rows]

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN * 0.7, COL_DOUBLE_IN * 0.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.set_yticks(range(len(regions))); ax.set_yticklabels(regions, fontsize=8)
    for tick, nm in zip(ax.get_yticklabels(), regions):
        tick.set_fontweight("bold" if nm in _DYN else "normal")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            star = "*" if P[i, j] < 0.05 else ""
            ax.text(j, i, f"{M[i, j]:+.2f}{star}", ha="center", va="center",
                    fontsize=6.5, color="k" if abs(M[i, j]) < 0.5 else "w")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("correlation with $\\hat{V}$ anomaly", fontsize=8); cb.ax.tick_params(labelsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_si_named_modes.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
