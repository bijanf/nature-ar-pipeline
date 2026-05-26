"""
Figure 1 — Pipeline schematic.

Two-column overview of the data flow:

    ARCO-ERA5  ->  Holton-features  ->  GW detector  ->  Stage 1 (LightGBM)
                                                                |
                                                                v
                                        event_post  --  Stage 2 quantile regression
                                                                |
                                                                v
                              CMIP6 four-SSP (regridded) --> direct + delta inference
                                                                |
                                                                v
                                                       SHAP θ_e vs PV attribution

The figure is pure vector matplotlib — no data dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.figures._style import COL_DOUBLE_IN, apply_nature_style

_BOXES: tuple[tuple[float, float, float, float, str], ...] = (
    # (x, y, w, h, label)
    (0.04, 0.66, 0.22, 0.20, "ARCO-ERA5\n(Pangeo, streamed)"),
    (0.30, 0.66, 0.22, 0.20, "Holton features\nIVT, θe, PV, σBI"),
    (0.56, 0.66, 0.22, 0.20, "Guan-Waliser\nAR mask"),
    (0.18, 0.36, 0.22, 0.20, "Stage 1\nLGBM linear-tree"),
    (0.46, 0.36, 0.22, 0.20, "Events\n3-D label"),
    (0.74, 0.36, 0.22, 0.20, "Stage 2\nquantile reg.\nα=0.05/0.50/0.95"),
    (0.04, 0.06, 0.30, 0.20, "CMIP6 4-SSP\n(SSP2-4.5/3-7.0/4-6.0/5-8.5)"),
    (0.38, 0.06, 0.22, 0.20, "direct +\ndelta inference"),
    (0.64, 0.06, 0.30, 0.20, "SHAP attribution\nθe vs PV per landfall band"),
)

_ARROWS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (1, 3),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (5, 7),
)


def build_figure() -> plt.Figure:
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    centres: list[tuple[float, float]] = []
    for x, y, w, h, label in _BOXES:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            linewidth=0.6,
            edgecolor="black",
            facecolor="#f0f0f0",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=6)
        centres.append((x + w / 2, y + h / 2))

    for src_idx, dst_idx in _ARROWS:
        x0, y0 = centres[src_idx]
        x1, y1 = centres[dst_idx]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "lw": 0.5, "color": "black"},
        )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="figures/fig1_pipeline_schematic.pdf")
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
