"""The circulation behind the dynamic term: 500 hPa height change and leading modes.

(a) The observed change in annual-mean 500 hPa geopotential height (recent minus
pre-satellite), the reorganisation of the general circulation that the dynamic term
projects onto, with the eight corridors boxed. (b-d) The three leading EOFs of the
global annual z500 anomaly field -- the data-driven circulation modes used as
predictors in the machine-learning attribution (Figure of mode attribution). Built
from the five ERA5 monthly windows on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from src.analysis.decomposition import _DIR, _FILES
from src.analysis.mode_attribution import circulation_modes
from src.figures._style import apply_nature_style, COL_DOUBLE_IN
from src.figures.fig_global_ar_regions import _REGIONS

_G = 9.80665


def _z500_mean(win):
    ds = xr.open_dataset(_DIR / _FILES[win])
    lev = "pressure_level" if "pressure_level" in ds.dims else "level"
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    z = (ds["z"].sel({lev: 500}) / _G).mean(tdim)
    return z


def plot(out_path: Path) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    apply_nature_style()

    dz = (_z500_mean("recent") - _z500_mean("presat"))
    lon = dz["longitude"].to_numpy(); lat = dz["latitude"].to_numpy()
    modes = circulation_modes(6)
    eofs, varexp = modes["eofs"], modes["varexp"]

    proj = ccrs.Robinson(central_longitude=200)
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.62))
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.10)

    def _base(ax):
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.25, edgecolor="0.3")
        ax.set_global()

    # (a) z500 change with corridor boxes
    ax = fig.add_subplot(gs[0, 0], projection=proj); _base(ax)
    dmax = float(np.nanpercentile(np.abs(dz.to_numpy()), 98))
    pcm = ax.pcolormesh(lon, lat, dz.to_numpy(), transform=pc, cmap="RdBu_r",
                        vmin=-dmax, vmax=dmax, shading="auto")
    for _n, lo0, lo1, la0, la1 in _REGIONS:
        ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0],
                color="#111", lw=0.7, transform=pc, zorder=6)
    ax.set_title("(a) Δz500, recent − pre-sat", fontsize=8, pad=3)
    cb = fig.colorbar(pcm, ax=ax, orientation="vertical", fraction=0.024, pad=0.015)
    cb.set_label("Δz500 (m)", fontsize=7.5); cb.ax.tick_params(labelsize=7)

    # (b-d) leading three EOFs
    for k in range(3):
        ax = fig.add_subplot(gs[(k + 1) // 2, (k + 1) % 2], projection=proj); _base(ax)
        e = eofs[k]
        amax = float(np.nanpercentile(np.abs(e), 98))
        pcm = ax.pcolormesh(lon, lat, e, transform=pc, cmap="PuOr_r",
                            vmin=-amax, vmax=amax, shading="auto")
        ax.set_title(f"({'bcd'[k]}) EOF{k+1} — {100*varexp[k]:.0f}% var", fontsize=8, pad=3)
        cb = fig.colorbar(pcm, ax=ax, orientation="vertical", fraction=0.024, pad=0.015)
        cb.ax.tick_params(labelsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_circulation_modes.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
