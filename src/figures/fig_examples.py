"""Worked examples: Western Europe and southeastern South America.

For each corridor, two panels on a regional map:
  left   recent-period (1995-2024) mean transport magnitude (shading) and mean
         850 hPa wind (grey vectors), corridor box;
  right  change of transport magnitude (shading) with the change of the 500 hPa
         height as labelled contours (10 m interval; solid positive, dashed
         negative) and the change of the 850 hPa wind as vectors.

Usage: python -m src.figures.fig_examples OUT.pdf
"""

from __future__ import annotations

import argparse
import glob
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.analysis import fullcolumn_decomp as fd  # noqa: E402
from src.figures.fig_decomp_terms import MM, _panel_label, _style  # noqa: E402
from src.figures.fig_global_ar_regions import _REGIONS  # noqa: E402
from src.figures.fig_maps import G, W0, W1, _anom_global, _drop, _period_mean  # noqa: E402

CASES = [("Western Europe", (300, 20, 30, 70), 250), ("SE South America", (270, 330, -50, -5), 400)]


def _sector(da, ext):
    lon0, lon1, lat0, lat1 = ext
    da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude")
    if lon0 > lon1:
        da = xr.concat([da.sel(longitude=slice(lon0, 360)), da.sel(longitude=slice(0, lon1))], "longitude")
    else:
        da = da.sel(longitude=slice(lon0, lon1))
    return da.sel(latitude=slice(lat1, lat0)) if float(da.latitude[0]) > float(da.latitude[-1]) \
        else da.sel(latitude=slice(lat0, lat1))


def load():
    sl = xr.open_dataset(fd.ERA5_DIR / "era5_sl_monthly_1940_2024.nc")
    mag = np.hypot(sl["viwve"], sl["viwvn"])
    files = sorted(glob.glob(str(fd.ERA5_DIR / "pl" / "era5_pl_monthly_*.nc")))
    pl = xr.open_mfdataset(files, combine="by_coords", preprocess=_drop)
    u, v = pl["u"].sel(pressure_level=850), pl["v"].sel(pressure_level=850)
    old = xr.open_mfdataset(sorted(glob.glob(str(fd.DATA / "era5_global_monthly" / "*_monthly_global.nc"))),
                            combine="by_coords")
    z = old["z"].sel(pressure_level=500) / G
    z = z.isel(valid_time=~z.get_index("valid_time").duplicated()).sortby("valid_time")
    return {"mag1": _period_mean(mag, W1).load(), "dmag": (_period_mean(mag, W1) - _period_mean(mag, W0)).load(),
            "u1": _period_mean(u, W1).load(), "v1": _period_mean(v, W1).load(),
            "du": (_period_mean(u, W1) - _period_mean(u, W0)).load(), "dv": (_period_mean(v, W1) - _period_mean(v, W0)).load(),
            "dz": _anom_global(_period_mean(z, W1) - _period_mean(z, W0))}


def draw(out_path):
    _style()
    f = load()
    fig = plt.figure(figsize=(183 * MM, 115 * MM))
    letters = iter("abcd")
    for row, (name, ext, vmax1) in enumerate(CASES):
        box = next(r for r in _REGIONS if r[0] == name)[1:]
        lon0, lon1, lat0, lat1 = ext
        cl = ((lon0 + (lon1 if lon1 > lon0 else lon1 + 360)) / 2) % 360
        for col in range(2):
            ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection=ccrs.PlateCarree(central_longitude=cl))
            ax.set_extent([lon0, lon1 if lon1 > lon0 else lon1 + 360, lat0, lat1], crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.4, color="0.25")
            gl = ax.gridlines(draw_labels=True, linewidth=0.15, color="0.75", alpha=0.6, linestyle=":",
                              xlocs=range(-180, 181, 10), ylocs=range(-80, 81, 10))
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = gl.ylabel_style = {"size": 5}
            if col == 0:
                da = _sector(f["mag1"], ext)
                im = ax.pcolormesh(da.longitude, da.latitude, da.values, cmap="viridis", vmin=0, vmax=vmax1,
                                   transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
                u, v = _sector(f["u1"], ext), _sector(f["v1"], ext)
                s = 3
                ax.quiver(u.longitude.values[::s], u.latitude.values[::s], u.values[::s, ::s], v.values[::s, ::s],
                          transform=ccrs.PlateCarree(), color="0.85", scale=250, width=0.002)
                cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, aspect=35)
                cb.set_label("mean transport, 1995–2024 (kg m$^{-1}$ s$^{-1}$)", fontsize=6)
            else:
                da = _sector(f["dmag"], ext)
                im = ax.pcolormesh(da.longitude, da.latitude, da.values, cmap="RdBu_r", vmin=-40, vmax=40,
                                   transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
                dz = _sector(f["dz"], ext)
                levels = np.arange(-30, 31, 4)
                levels = levels[levels != 0]
                cs = ax.contour(dz.longitude, dz.latitude, dz.values, levels=levels, colors="black", linewidths=0.5,
                                linestyles=["dashed" if lv < 0 else "solid" for lv in levels], transform=ccrs.PlateCarree())
                ax.clabel(cs, fmt="%d", fontsize=4.5, inline=True)
                du, dv = _sector(f["du"], ext), _sector(f["dv"], ext)
                s = 3
                ax.quiver(du.longitude.values[::s], du.latitude.values[::s], du.values[::s, ::s], dv.values[::s, ::s],
                          transform=ccrs.PlateCarree(), color="black", scale=40, width=0.0025)
                cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, aspect=35)
                cb.set_label("change of transport (kg m$^{-1}$ s$^{-1}$)", fontsize=6)
            cb.ax.tick_params(labelsize=5.5)
            ax.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False,
                                   edgecolor="black" if col == 0 else "0.2", linewidth=1.0,
                                   transform=ccrs.PlateCarree(), zorder=6))
            _panel_label(ax, next(letters))
    fig.subplots_adjust(wspace=0.15, hspace=0.3, left=0.05, right=0.97, top=0.96, bottom=0.04)
    fig.savefig(out_path, dpi=600)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    draw(ap.parse_args().out)


if __name__ == "__main__":
    main()
