"""Observed vs CMIP6 *ensemble* dynamic contribution per corridor.

The observed intensification (ERA5, 1940-59 -> 2015-24) carries a large,
regionally organised dynamic (circulation) fraction in several corridors. We apply
the identical three-term decomposition to a CMIP6 multi-model ensemble (ssp585,
1995-2014 -> 2081-2100) and compare the dynamic fraction corridor by corridor. The
ensemble (individual models as dots, median as a square, inter-model interquartile
range as a bar) is overwhelmingly thermodynamic and under-represents the observed
dynamic contribution in every dynamically driven corridor -- a robust feature of
the CMIP6 circulation response, not one model's quirk. Because the observed and
projected changes span different periods and warming amounts, we compare the
dynamic FRACTION of each corridor's change, not its magnitude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import config
from src.figures._style import COL_DOUBLE_IN, apply_nature_style
from src.figures.fig_global_ar_regions import _REGIONS

_DYN = {"Western Europe", "Amazon outflow", "East Asia", "SE South America"}
_CLIP = (-60.0, 160.0)


def _obs_frac(rows):
    out = {}
    for r in rows:
        t = r["d_total"]
        out[r["region"]] = 100 * r["d_dyn"] / t if t else np.nan
    return out


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    obs = json.loads((config.CACHE_DIR / "decomposition.json").read_text())["full"]
    ens = json.loads((config.CACHE_DIR / "cmip6_ensemble.json").read_text())
    fo = _obs_frac(obs)
    estats = {e["region"]: e for e in ens["ensemble"]}
    n_mod = max(e["n_models"] for e in ens["ensemble"])

    names = [n for n, *_ in _REGIONS]
    names = sorted(names, key=lambda n: -fo[n])
    yy = np.arange(len(names))

    def clip(x):
        return np.clip(x, *_CLIP)

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.5))

    for i, n in enumerate(names):
        e = estats[n]
        q25, q75 = 100 * e["dyn_frac_q25"], 100 * e["dyn_frac_q75"]
        med = 100 * e["dyn_frac_median"]
        models = clip(100 * np.array(e["dyn_frac_models"]))
        # inter-quartile bar (behind), individual models (jittered dots), median square
        ax.plot(
            [clip(q25), clip(q75)], [i, i], color="#9fb6c6", lw=4, solid_capstyle="butt", zorder=1
        )
        jit = (np.arange(models.size) - models.size / 2) * 0.0  # no jitter; stack on row
        ax.scatter(
            models,
            np.full(models.size, i) + jit,
            s=9,
            color="#5b7d99",
            alpha=0.7,
            edgecolors="none",
            zorder=2,
            label="CMIP6 models" if i == 0 else None,
        )
        ax.scatter(
            [clip(med)],
            [i],
            s=42,
            color="#16415f",
            marker="s",
            zorder=3,
            label="CMIP6 median" if i == 0 else None,
        )
        # observed
        ax.scatter(
            [clip(fo[n])],
            [i],
            s=55,
            color="#c0392b",
            marker="D",
            zorder=4,
            label="observed (ERA5)" if i == 0 else None,
        )
        if fo[n] > _CLIP[1] - 5:
            ax.text(
                _CLIP[1] - 2,
                i,
                f"{fo[n]:.0f}%→",
                va="center",
                ha="right",
                fontsize=6.5,
                color="#c0392b",
            )
        elif fo[n] < _CLIP[0] + 5:
            ax.text(
                _CLIP[0] + 2,
                i,
                f"←{fo[n]:.0f}%",
                va="center",
                ha="left",
                fontsize=6.5,
                color="#c0392b",
            )

    ax.axvline(0, color="0.4", lw=0.6)
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=8)
    for tick, n in zip(ax.get_yticklabels(), names, strict=False):
        tick.set_fontweight("bold" if n in _DYN else "normal")
    ax.set_xlabel("dynamic fraction of the corridor's IVT change (%)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_ylim(-0.6, len(names) - 0.4)
    # key above the axes so it never sits over the points
    ax.legend(
        fontsize=7,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        handletextpad=0.3,
        columnspacing=1.3,
        borderaxespad=0.0,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}  (ensemble of {n_mod} CMIP6 models)")
    for n in names:
        e = estats[n]
        print(
            f"  {n:18s} obs {fo[n]:+6.0f}%   CMIP6 median {100 * e['dyn_frac_median']:+6.0f}% "
            f"[{100 * e['dyn_frac_q25']:+.0f}, {100 * e['dyn_frac_q75']:+.0f}]  n={e['n_models']}"
        )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_cmip6_anchor.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
