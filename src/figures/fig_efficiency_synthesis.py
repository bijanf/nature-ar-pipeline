"""Flagship synthesis — the convective-efficiency story on one page.

Built entirely from data already on disk (WGLC lightning + ERA5 CAPE/CIN/shear +
NOAA ONI); no downloads. Three panels make the whole argument:

  (a) The convective-EFFICIENCY climatology — realized convection (lightning) per
      unit instability (CAPE), a new gridded field. Coherent continental structure;
      the environment does not explain it (land/ocean lightning differs ~3x at
      ~equal CAPE).
  (b) Efficiency is an INDEPENDENT degree of freedom — under ENSO the environment
      and the efficiency respond with OPPOSITE sign in the deep tropics (efficiency
      compensates, it does not track, the environment).
  (c) A driver in action — the South Asia efficiency rise localises on the
      Indo-Gangetic Plain (black box), the aerosol-invigoration fingerprint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from src.figures._style import apply_nature_style, COL_DOUBLE_IN
from src.figures.poc_lightning_efficiency import _env_annual_1deg, _lightning_annual_1deg, _land_mask
from src.figures.poc_efficiency_trend import (
    _env_annual_cube, _lightning_annual_cube, _region_series, _REGIONS,
)
from src.figures.poc_efficiency_enso import _oni_annual

_IGP = dict(lon0=75, lon1=88, lat0=24, lat1=30)


def _box_sel(da, b):
    return da.where((da["latitude"] >= b["lat0"]) & (da["latitude"] <= b["lat1"])
                    & (da["longitude"] >= b["lon0"]) & (da["longitude"] <= b["lon1"]), drop=True)


def plot(out_path: Path) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    apply_nature_style()

    # ---- (a) efficiency climatology -----------------------------------------
    cape, E = _env_annual_1deg()
    light = _lightning_annual_1deg(cape)
    eff = (light / cape.where(cape > 50.0))

    # ---- shared cubes for (b),(c) -------------------------------------------
    Ec = _env_annual_cube(); Lc = _lightning_annual_cube(Ec)
    Ec, Lc = xr.align(Ec, Lc, join="inner")
    years = Ec["year"].to_numpy().astype(int)
    landm = _land_mask(Ec.isel(year=0))
    trop = landm & (np.abs(Ec["latitude"]) <= 30)
    wL = np.cos(np.deg2rad(Lc["latitude"])).broadcast_like(Lc.isel(year=0)).where(trop)
    common = (Lc.where(trop) * wL).sum(("latitude", "longitude")) / wL.sum(("latitude", "longitude"))
    eff_cube = (Lc / (common / common.mean("year"))) / Ec.where(Ec > 0)

    # (b) ENSO sign-opposition
    oni_map = _oni_annual()
    oni = np.array([oni_map[y] for y in years])
    rows = []
    for name, *box in _REGIONS:
        es = _region_series(eff_cube, box, landm); vs = _region_series(Ec, box, landm)
        re = float(np.corrcoef(es - np.nanmean(es), oni)[0, 1])
        rv = float(np.corrcoef(vs - np.nanmean(vs), oni)[0, 1])
        rows.append((name, rv, re))

    # (c) South Asia per-cell efficiency trend
    from scipy.stats import theilslopes
    eff_sa = _box_sel(eff_cube, dict(lon0=60, lon1=100, lat0=5, lat1=35))
    yrs_c = eff_sa["year"].to_numpy().astype(float)

    def _ct(v):
        m = np.isfinite(v)
        if m.sum() < 6:
            return np.nan
        mu = np.nanmean(v[m])
        return 100 * theilslopes(v[m], yrs_c[m])[0] * 10 / mu if mu else np.nan
    cell = xr.apply_ufunc(_ct, eff_sa, input_core_dims=[["year"]], vectorize=True, output_dtypes=[float])

    # ---- layout --------------------------------------------------------------
    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.66))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.30, wspace=0.28)

    ax = fig.add_subplot(gs[0, :], projection=ccrs.Robinson(central_longitude=160))
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.3); ax.set_global()
    ez = eff.where(eff > 0)
    vmax = float(np.nanpercentile(ez, 99))
    pcm = ax.pcolormesh(eff["longitude"], eff["latitude"], ez.to_numpy(),
                        transform=ccrs.PlateCarree(), cmap="magma",
                        norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax), shading="auto")
    ax.set_title("(a) Convective efficiency = lightning per unit CAPE — a new gridded field "
                 "(continents bright, oceans dark; environment alone cannot explain it)", fontsize=7.3)
    cb = fig.colorbar(pcm, ax=ax, fraction=0.022, pad=0.02); cb.set_label("efficiency (log)", fontsize=6)

    ax2 = fig.add_subplot(gs[1, 0])
    names = [r[0] for r in rows]; yy = np.arange(len(names))
    ax2.barh(yy - 0.2, [r[1] for r in rows], height=0.4, color="#e67e22", label="environment")
    ax2.barh(yy + 0.2, [r[2] for r in rows], height=0.4, color="#8e44ad", label="efficiency")
    ax2.axvline(0, color="0.4", lw=0.6); ax2.set_xlim(-1, 1)
    ax2.set_yticks(yy); ax2.set_yticklabels(names, fontsize=6.5)
    ax2.set_xlabel("correlation with ENSO (ONI)", fontsize=7)
    ax2.set_title("(b) Efficiency is independent: opposite-sign\nENSO response to the environment", fontsize=7.3)
    ax2.legend(fontsize=5.5, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    ax3.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4)
    ax3.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="0.5")
    vm = float(np.nanpercentile(np.abs(cell.to_numpy()), 95))
    pc3 = ax3.pcolormesh(cell["longitude"], cell["latitude"], cell.to_numpy(), cmap="RdBu_r",
                         vmin=-vm, vmax=vm, transform=ccrs.PlateCarree(), shading="auto")
    ax3.plot([_IGP["lon0"], _IGP["lon1"], _IGP["lon1"], _IGP["lon0"], _IGP["lon0"]],
             [_IGP["lat0"], _IGP["lat0"], _IGP["lat1"], _IGP["lat1"], _IGP["lat0"]],
             color="k", lw=1.0, transform=ccrs.PlateCarree())
    ax3.set_extent([60, 100, 5, 35], crs=ccrs.PlateCarree())
    ax3.set_title("(c) A driver in action: South Asia efficiency\nrise peaks on the Indo-Gangetic Plain", fontsize=7.3)
    fig.colorbar(pc3, ax=ax3, fraction=0.04, pad=0.03).set_label("trend (%/dec)", fontsize=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_efficiency_synthesis.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
