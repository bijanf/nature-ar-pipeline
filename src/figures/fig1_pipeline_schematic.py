"""
Figure 1 — Observational pipeline schematic.

A clean top-to-bottom flow of the actual (machine-learning-free) pipeline:

    ERA5 reanalysis
        -> moisture transport & dynamics (IVT, IWV, theta_e, PV)
        -> Guan-Waliser AR detection (period-internal 85th-pct IVT)
        -> event extraction (intensity, footprint, duration, landfall)
        -> { three-window period contrast ;  thermodynamic/dynamic split }

Pure vector matplotlib, no data dependency. Designed for legibility: evenly
spaced rounded cards, a single clean arrow spine, one symmetric branch at the
end, and no overlapping elements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.figures._style import COL_SINGLE_IN, apply_nature_style

# Palette: muted, print-friendly. Source (blue), process (slate), output (green).
_SRC = dict(fc="#e6eef7", ec="#3b6ea5")
_PROC = dict(fc="#eef1f4", ec="#566573")
_DET = dict(fc="#e6f0ee", ec="#2f7d6b")   # detection — the methodological heart
_OUT = dict(fc="#e9f3ea", ec="#3c7d52")

_TITLE_KW = dict(ha="center", va="center", fontsize=7.6, fontweight="bold", color="#1b2733")
_SUB_KW = dict(ha="center", va="center", fontsize=6.2, color="#3d4a57")


def _card(ax, cx, cy, w, h, title, sub, style):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.1, facecolor=style["fc"], edgecolor=style["ec"], zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx, cy + h * 0.17, title, zorder=3, **_TITLE_KW)
    ax.text(cx, cy - h * 0.22, sub, zorder=3, **_SUB_KW)


def _arrow(ax, x0, y0, x1, y1, color="#566573"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.2, color=color, zorder=1,
        shrinkA=0, shrinkB=0,
    ))


def plot(out_path: Path) -> Path:
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(COL_SINGLE_IN * 1.32, COL_SINGLE_IN * 1.46))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    cx = 5.0
    w_main, h = 7.4, 1.18
    ys = [8.85, 7.05, 5.25, 3.45]            # four stacked stages
    stages = [
        ("ERA5 reanalysis", "1940–2024  ·  6-hourly  ·  0.25°  ·  CDS", _SRC),
        ("Moisture transport & dynamics", "IVT, IWV, $\\theta_e$, PV  (850 / 500 / 250 hPa)", _PROC),
        ("Atmospheric-river detection", "Guan–Waliser  ·  period-internal 85th-pct IVT", _DET),
        ("Event extraction", "intensity · footprint · duration · landfall", _PROC),
    ]
    for (title, sub, style), y in zip(stages, ys):
        _card(ax, cx, y, w_main, h, title, sub, style)
    # Spine arrows between consecutive stages.
    for ytop, ybot in zip(ys[:-1], ys[1:]):
        _arrow(ax, cx, ytop - h / 2, cx, ybot + h / 2)

    # Symmetric branch from "Event extraction" into the two analyses.
    y_out = 1.25
    junction_y = 2.18
    lx, rx, w_out = 2.55, 7.45, 4.5
    _arrow(ax, cx, ys[-1] - h / 2, cx, junction_y + 0.02)        # down to junction
    ax.plot([lx, rx], [junction_y, junction_y], color="#566573", lw=1.2, zorder=1)  # cross-bar
    _arrow(ax, lx, junction_y, lx, y_out + h / 2)
    _arrow(ax, rx, junction_y, rx, y_out + h / 2)
    _card(ax, lx, y_out, w_out, h, "Three-window contrast",
          "intensity trajectory  →  Fig. 2, 4", _OUT)
    _card(ax, rx, y_out, w_out, h, "Thermo / dynamic split",
          "IVT = IWV·$\\hat{V}$  →  Fig. 3", _OUT)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig1_pipeline_schematic.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
