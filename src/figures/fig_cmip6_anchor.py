"""Observed vs CMIP6-projected dynamic contribution per corridor.

The observed intensification (ERA5, 1940-59 -> 2015-24) carries a large,
regionally organised dynamic (circulation) fraction in several corridors. We apply
the identical three-term decomposition to a CMIP6 projection (MPI-ESM1-2-HR ssp585,
1995-2014 -> 2081-2100) and compare the dynamic fraction corridor by corridor. The
projection is overwhelmingly thermodynamic everywhere and under-represents the
observed dynamic contribution -- most starkly for southeastern South America, where
the observed column moisture declines (dynamic fraction above 100%) but the model
moistens. Because the observed and projected changes span different periods and
warming amounts, we compare the dynamic FRACTION of each corridor's change, not its
magnitude.
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


def _frac(rows):
    out = {}
    for r in rows:
        t = r["d_total"]
        out[r["region"]] = 100 * r["d_dyn"] / t if t else np.nan
    return out


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    obs = json.loads((config.CACHE_DIR / "decomposition.json").read_text())["full"]
    mod = json.loads((config.CACHE_DIR / "cmip6_decomp.json").read_text())["corridors"]
    fo = _frac(obs)
    fm = _frac(mod)

    names = [n for n, *_ in _REGIONS]
    names = sorted(names, key=lambda n: -fo[n])
    yy = np.arange(len(names))
    o = np.clip([fo[n] for n in names], -50, 160)
    m = np.clip([fm[n] for n in names], -50, 160)

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.46))
    h = 0.38
    ax.barh(yy + h / 2, o, height=h, color="#16415f", label="observed (ERA5)")
    ax.barh(yy - h / 2, m, height=h, color="#b9c8d4", label="CMIP6 ssp585 (projected)")
    ax.axvline(0, color="0.4", lw=0.6)
    # annotate the SE South America >100% (negative-thermo) observed case
    for i, n in enumerate(names):
        if fo[n] > 155:
            ax.text(
                158,
                i + h / 2,
                f"{fo[n]:.0f}%→",
                va="center",
                ha="right",
                fontsize=6.5,
                color="#16415f",
            )
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=8)
    for tick, n in zip(ax.get_yticklabels(), names, strict=False):
        tick.set_fontweight("bold" if n in _DYN else "normal")
    ax.set_xlabel("dynamic fraction of the corridor's IVT change (%)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    # Place the key above the axes (two columns) so it never sits over the bars.
    ax.legend(
        fontsize=7.5,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        handlelength=1.2,
        columnspacing=1.6,
        borderaxespad=0.0,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
    for n in names:
        print(f"  {n:18s} obs {fo[n]:+6.0f}%   CMIP6 {fm[n]:+6.0f}%")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_cmip6_anchor.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
