"""Hero figure / graphical summary — clean, publication-grade, paper-consistent.

A single high-resolution (0.25 deg) map of the eastern North Pacific moisture
corridor into the US West Coast: the recent-period mean IVT as a calm filled
field, the transport direction as restrained arrows, and the 1940->2024
intensification drawn as labelled change contours (not a styled blob). A small
inset carries the continuous corridor trajectory. Light theme, matching the rest
of the manuscript's figures. Built from data already on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from src import config
from src.figures._style import apply_nature_style, COL_DOUBLE_IN


def _num():
    d = {}
    for f in ("driver_attribution", "timeseries"):
        p = config.CACHE_DIR / f"{f}.json"
        if p.exists():
            d.update(json.loads(p.read_text()))
    return d


def _significance():
    """Welch t-test on per-year annual IVT (recent vs pre-sat); return non-sig mask."""
    try:
        from scipy.stats import ttest_ind
        p0 = xr.open_dataset(config.CACHE_DIR / "spatial_peryear_presat.nc")["ivt_annual"]
        p1 = xr.open_dataset(config.CACHE_DIR / "spatial_peryear_recent.nc")["ivt_annual"]
        _, pv = ttest_ind(p1.to_numpy(), p0.to_numpy(), axis=0, equal_var=False)
        return pv < 0.05
    except Exception:
        return None


def plot(out_path: Path) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    apply_nature_style()
    ds = xr.open_dataset(config.CACHE_DIR / "spatial_composites.nc")
    rec = ds.sel(window="recent"); pre = ds.sel(window="presat")
    ivt = rec["ivt"]; d_ivt = rec["ivt"] - pre["ivt"]
    U = -rec["ivt_u"]; V = -rec["ivt_v"]          # negate: pipeline stores flipped sign
    lon = ivt["longitude"].to_numpy(); lat = ivt["latitude"].to_numpy()
    sig = _significance()
    n = _num()

    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.66))
    ax = fig.add_subplot(111, projection=pc)
    ax.set_extent([210, 248, 25, 58], crs=pc)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f4f8fb", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e9e4d8", zorder=1)

    # mean IVT corridor — calm sequential fill
    vmax = float(np.nanpercentile(ivt.to_numpy(), 99))
    levels = np.linspace(0, vmax, 11)
    cf = ax.contourf(lon, lat, ivt.to_numpy(), levels=levels, cmap="YlGnBu",
                     extend="max", transform=pc, zorder=2)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="#46505c", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="#6b7682", zorder=5)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.2, edgecolor="#9aa4ae", zorder=5)

    # restrained transport arrows
    st = 9
    ax.quiver(lon[::st], lat[::st], U.to_numpy()[::st, ::st], V.to_numpy()[::st, ::st],
              transform=pc, scale=2800, width=0.0035, color="#33404d", alpha=0.8, zorder=6)

    # intensification as labelled change contours (significant, positive core)
    dd = d_ivt.to_numpy()
    if sig is not None:
        dd = np.where(sig, dd, np.nan)
    cs = ax.contour(lon, lat, dd, levels=[6, 10, 14], colors="#c0392b",
                    linewidths=[0.8, 1.1, 1.4], transform=pc, zorder=7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="+%.0f")

    # readable lon/lat graticule (Nature convention: legible axis labels, no prose)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5, linestyle=":")
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 8, "color": "#1b2733"}
    gl.ylabel_style = {"size": 8, "color": "#1b2733"}

    cb = fig.colorbar(cf, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("mean IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    # inset trajectory — data only, legible ticks, no prose
    try:
        from src.figures.fig_timeseries import _annual_series, _theil_sen
        s = _annual_series()
        yy = s.index.to_numpy().astype(int); vv = s.to_numpy()
        slope, intercept, lo, hi, p = _theil_sen(yy, vv)
        axi = inset_axes(ax, width="38%", height="30%", loc="lower left",
                         bbox_to_anchor=(0.035, 0.07, 1, 1), bbox_transform=ax.transAxes)
        axi.plot(yy, vv, color="#2c6e8f", lw=0.8)
        axi.plot(yy, intercept + slope * yy, color="#c0392b", lw=1.4)
        axi.set_facecolor("white"); axi.patch.set_alpha(0.92)
        axi.tick_params(labelsize=7, length=2.5, colors="#1b2733")
        axi.set_xlabel("year", fontsize=7.5); axi.set_ylabel("IVT", fontsize=7.5)
        for sp in axi.spines.values():
            sp.set_color("#c7ced6")
        axi.margins(x=0.01)
    except Exception as e:
        print("inset skipped:", e)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_hero.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
