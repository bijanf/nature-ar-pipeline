"""Machine-learning attribution of the dynamic term to circulation modes.

We test whether each corridor's dynamic intensification is organised by recognised
modes of the large-scale circulation. The modes are the leading EOFs (principal
components) of the global annual 500 hPa height anomaly field, 1940-2024; the
target is the corridor transport-efficiency anomaly Vhat. For each corridor we fit
a ridge (linear) and a random-forest (non-linear) regression on the modes, scored
by leave-one-block-out cross-validation, and rank the modes by permutation
importance. The circulation-driven corridors are predictable from the modes
(positive CV skill, loading on a few modes); the thermodynamic null corridors are
not. CV skill is deliberately modest, because the dynamic term is dominated by a
secular trend rather than interannual mode swings -- which is itself the point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import config
from src.figures._style import COL_DOUBLE_IN, apply_nature_style

_DYN = {"Western Europe", "Amazon outflow", "East Asia", "SE South America"}


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    data = json.loads((config.CACHE_DIR / "mode_attribution.json").read_text())
    rows = data["corridors"]
    varexp = data["varexp"]
    names = [r["region"] for r in rows]
    n_pc = data["n_pc"]

    fig, (axb, axh) = plt.subplots(
        1,
        2,
        figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.42),
        gridspec_kw=dict(width_ratios=[1.05, 1.0], wspace=0.42),
    )

    # (a) cross-validated skill: ridge vs random forest
    y = np.arange(len(names))
    h = 0.38
    r_ridge = [r["r2_ridge"] for r in rows]
    r_rf = [r["r2_rf"] for r in rows]
    axb.barh(y - h / 2, r_ridge, height=h, color="#2980b9", label="ridge")
    axb.barh(y + h / 2, r_rf, height=h, color="#7fb3d5", label="random forest")
    axb.axvline(0, color="0.4", lw=0.6)
    axb.set_yticks(y)
    axb.set_yticklabels(names, fontsize=7)
    for tick, nm in zip(axb.get_yticklabels(), names, strict=False):
        tick.set_fontweight("bold" if nm in _DYN else "normal")
    axb.set_xlabel("cross-validated $R^2$ (modes → $\\hat{V}$)", fontsize=8)
    axb.tick_params(labelsize=7)
    # Legend on the lower-left: the lower corridors (US West Coast, SE S. America,
    # Amazon) have positive bars extending right, leaving that corner clear, while
    # the only left-extending (negative-R^2) bars sit in the upper half.
    axb.legend(fontsize=6.5, frameon=False, loc="lower left")
    axb.set_title("(a) predictability of the dynamic term", fontsize=8, pad=3)

    # (b) permutation importance heatmap: corridor x mode
    imp = np.array([r["perm_importance"] for r in rows])
    im = axh.imshow(
        imp, aspect="auto", cmap="YlOrRd", vmin=0, vmax=float(np.nanpercentile(imp, 95))
    )
    axh.set_xticks(range(n_pc))
    axh.set_xticklabels([f"PC{i + 1}\n{100 * varexp[i]:.0f}%" for i in range(n_pc)], fontsize=6.5)
    axh.set_yticks(range(len(names)))
    axh.set_yticklabels(names, fontsize=7)
    for tick, nm in zip(axh.get_yticklabels(), names, strict=False):
        tick.set_fontweight("bold" if nm in _DYN else "normal")
    axh.set_title("(b) mode permutation importance", fontsize=8, pad=3)
    cb = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.03)
    cb.set_label("importance (Δ$R^2$)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_mode_attribution.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
