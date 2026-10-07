"""Global maps of the observed change, 1940-1969 to 1995-2024 (ERA5 full column).

  a  recent-period mean transport magnitude (shading) with the eight corridor boxes;
  b  change of the transport magnitude (shading);
  c  change of the 500 hPa geopotential height (shading) with the change of the
     850 hPa wind as vectors;
  d  change of the column water vapour (shading).

Reads the joined single-level file and the yearly pressure-level files.

Usage: python -m src.figures.fig_maps OUT.pdf
"""

from __future__ import annotations

import argparse
import glob
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.analysis import fullcolumn_decomp as fd  # noqa: E402
from src.figures.fig_decomp_terms import MM, _panel_label, _style  # noqa: E402
from src.figures.fig_global_ar_regions import _REGIONS  # noqa: E402

G = 9.80665
W0, W1 = fd.HEADLINE


def _drop(ds):
    return ds.drop_vars([v for v in ("expver", "number") if v in ds.variables])


def _period_mean(da, w):
    return da.sel(valid_time=slice(f"{w[0]}-01-01", f"{w[1]}-12-31")).mean("valid_time")


def _anom_global(da):
    """Remove the area-weighted global mean, so that the circulation pattern is visible."""
    da = da.load()
    w = np.cos(np.deg2rad(da.latitude))
    return da - da.weighted(w).mean(("latitude", "longitude"))


def load():
    sl = xr.open_dataset(fd.ERA5_DIR / "era5_sl_monthly_1940_2024.nc")
    mag = np.hypot(sl["viwve"], sl["viwvn"])
    out = {"mag1": _period_mean(mag, W1).load(), "dmag": (_period_mean(mag, W1) - _period_mean(mag, W0)).load(),
           "dtcwv": (_period_mean(sl["tcwv"], W1) - _period_mean(sl["tcwv"], W0)).load()}
    files = sorted(glob.glob(str(fd.ERA5_DIR / "pl" / "era5_pl_monthly_*.nc")))
    pl = xr.open_mfdataset(files, combine="by_coords", preprocess=_drop)
    if "z" in pl:
        z = pl["z"].sel(pressure_level=500) / G
        out["dz500"] = _anom_global(_period_mean(z, W1) - _period_mean(z, W0))
    else:
        old = xr.open_mfdataset(sorted(glob.glob(str(fd.DATA / "era5_global_monthly" / "*_monthly_global.nc"))),
                                combine="by_coords")
        z = old["z"].sel(pressure_level=500) / G
        z = z.isel(valid_time=~z.get_index("valid_time").duplicated()).sortby("valid_time")
        out["dz500"] = _anom_global(_period_mean(z, W1) - _period_mean(z, W0))
    u, v = pl["u"].sel(pressure_level=850), pl["v"].sel(pressure_level=850)
    out["du850"] = (_period_mean(u, W1) - _period_mean(u, W0)).load()
    out["dv850"] = (_period_mean(v, W1) - _period_mean(v, W0)).load()
    return out


def _boxes(ax, color="black"):
    for name, lon0, lon1, lat0, lat1 in _REGIONS:
        ax.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fill=False, edgecolor=color,
                               linewidth=0.8, transform=ccrs.PlateCarree(), zorder=5))


def draw(out_path: str):
    _style()
    f = load()
    proj = ccrs.Robinson(central_longitude=200)
    fig, axs = plt.subplots(2, 2, figsize=(183 * MM, 100 * MM), subplot_kw={"projection": proj})
    specs = [("mag1", "viridis", 0, 400, "mean transport magnitude, 1995–2024 (kg m$^{-1}$ s$^{-1}$)"),
             ("dmag", "RdBu_r", -40, 40, "change of transport magnitude (kg m$^{-1}$ s$^{-1}$)"),
             ("dz500", "RdBu_r", -25, 25, "change of 500 hPa height, global mean removed (m), and of 850 hPa wind"),
             ("dtcwv", "BrBG", -4, 4, "change of column water vapour (kg m$^{-2}$)")]
    for ax, (key, cmap, vmin, vmax, label), letter in zip(axs.flat, specs, "abcd"):
        da = f[key]
        lon, lat = da.longitude.values, da.latitude.values
        im = ax.pcolormesh(lon, lat, da.values, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
                           shading="auto", rasterized=True)
        ax.coastlines(linewidth=0.3, color="0.3")
        ax.set_global()
        _boxes(ax, "black" if key != "mag1" else "white")
        if key == "dz500":
            s = 9
            qv = ax.quiver(lon[::s], lat[::s], f["du850"].values[::s, ::s], f["dv850"].values[::s, ::s],
                           transform=ccrs.PlateCarree(), scale=50, width=0.002, color="black", headwidth=4)
            ax.quiverkey(qv, 0.92, 1.03, 2, "2 m s$^{-1}$", labelpos="E", fontproperties={"size": 5.5})
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, aspect=35)
        cb.set_label(label, fontsize=6)
        cb.ax.tick_params(labelsize=5.5)
        _panel_label(ax, letter)
    fig.subplots_adjust(wspace=0.05, hspace=0.25, left=0.02, right=0.98, top=0.97, bottom=0.03)
    fig.savefig(out_path, dpi=600)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    draw(ap.parse_args().out)


if __name__ == "__main__":
    main()
