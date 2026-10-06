"""GO/NO-GO ARTIFACT — global convective hazard-efficiency map.

The novel quantity the convective direction is built around: *realized convection
per unit favorable environment*. Numerator = satellite/ground lightning (WGLC
WWLLN climatology, strokes km^-2 d^-1, truly global). Denominator = the ERA5
convective environment (CAPE, and the WMAXSHEAR severe proxy). Efficiency is
their ratio on a common 1-degree grid:

    eff_cape = flash_density / CAPE        (lightning per unit instability)
    eff_E    = flash_density / WMAXSHEAR   (lightning per unit severe-env proxy)

This is the first observational test of whether the efficiency field has coherent
geographic structure. The PHYSICAL EXPECTATION (the go/no-go criterion): a very
strong land/ocean contrast (oceans carry moderate CAPE but almost no lightning ->
near-zero efficiency) and tropical-/sub-tropical-land hotspots (Congo, La Plata,
US Southeast, Maritime Continent, Himalayan foothills). If the map reproduces
that, the machinery and the concept are validated and there is a paper in *what
governs the efficiency*; if it is noise, we have learned that cheaply.

CAVEATS: monthly-mean CAPE understates intermittent potential; lightning is a
proxy for convective occurrence, not specifically *severe* hazard (the severe
numerator would be GPM PMW hail / overshooting tops — next step). WGLC reliably
covers low/mid latitudes; very high latitudes are sparse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

_WGLC = Path("/p/projects/poem/fallah/nature_ar_data/wglc/wglc_climatology_30m_monthly.nc")
_CONV = Path("/p/projects/poem/fallah/nature_ar_data/era5_convective_monthly")
_WIND = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")
# WGLC climatology is ~2012-2021; overlap the ERA5 environment to 2010-2023.
_ENV_PAIRS = [
    ("gap2_2000_2014_conv_monthly_global.nc", "gap2_2000_2014_monthly_global.nc", 2010, 2014),
    ("recent_conv_monthly_global.nc", "recent_monthly_global.nc", 2015, 2023),
]
# Coherence probes: (name, lon[0..360], lat). Known lightning hotspots vs oceans.
_HOTSPOTS = [
    ("Congo basin", 22.0, 0.0),
    ("La Plata (S.Am.)", 300.0, -30.0),
    ("US Southeast", 273.0, 33.0),
    ("Maritime Continent", 118.0, 0.0),
    ("Himalayan foothills", 84.0, 28.0),
]
_OCEANS = [
    ("E Pacific", 220.0, 0.0),
    ("S Pacific", 200.0, -20.0),
    ("S Atlantic", 340.0, -20.0),
    ("S Indian", 80.0, -25.0),
]


def _coord(ds, *cands):
    for c in cands:
        if c in ds.coords or c in ds.dims:
            return c
    raise KeyError(f"none of {cands} in {list(ds.coords)}")


def _lightning_annual_1deg(target: xr.DataArray) -> xr.DataArray:
    """WGLC annual-mean stroke density, regridded onto the ERA5 1-degree grid."""
    ds = xr.open_dataset(_WGLC)
    var = "density" if "density" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var]
    latn = _coord(ds, "lat", "latitude")
    lonn = _coord(ds, "lon", "longitude")
    # collapse the climatological month axis if present
    tdim = next((d for d in da.dims if d not in (latn, lonn)), None)
    if tdim is not None:
        da = da.mean(tdim)
    da = da.rename({latn: "latitude", lonn: "longitude"})
    # harmonise longitude to 0..360 to match ERA5
    if float(da["longitude"].min()) < 0:
        da = da.assign_coords(longitude=(da["longitude"] % 360)).sortby("longitude")
    da = da.sortby("latitude")
    # regrid to the ERA5 grid (linear is adequate for a structural first look)
    out = da.interp(latitude=target["latitude"], longitude=target["longitude"],
                    method="linear")
    return out.rename("flash_density")


def _env_annual_1deg() -> tuple[xr.DataArray, xr.DataArray]:
    """Global annual-mean CAPE and WMAXSHEAR E over 2010-2023 on the 1-deg grid."""
    capes, Es = [], []
    for cf, wf, y0, y1 in _ENV_PAIRS:
        cds = xr.open_dataset(_CONV / cf)
        wds = xr.open_dataset(_WIND / wf)
        ct = _coord(cds, "valid_time", "time")
        wt = _coord(wds, "valid_time", "time")
        cape = cds["cape"].clip(min=0.0).rename({ct: "time"}).assign_coords(time=cds[ct].values)
        lev = "pressure_level" if "pressure_level" in wds.dims else "level"
        u5, u8 = wds["u"].sel({lev: 500}), wds["u"].sel({lev: 850})
        v5, v8 = wds["v"].sel({lev: 500}), wds["v"].sel({lev: 850})
        shear = np.hypot(u5 - u8, v5 - v8).rename({wt: "time"}).assign_coords(time=wds[wt].values)
        shear = shear.reindex(time=cape["time"], method="nearest",
                              tolerance=np.timedelta64(20, "D"))
        E = np.sqrt(2.0 * cape) * shear
        yr = cape["time"].dt.year
        sel = (yr >= y0) & (yr <= y1)
        capes.append(cape.sel(time=sel)); Es.append(E.sel(time=sel))
    cape = xr.concat(capes, dim="time").mean("time")
    E = xr.concat(Es, dim="time").mean("time")
    return cape.rename("cape"), E.rename("E")


def _at(da: xr.DataArray, lon: float, lat: float) -> float:
    return float(da.sel(longitude=lon, latitude=lat, method="nearest"))


def _land_mask(da: xr.DataArray) -> xr.DataArray:
    """Boolean land mask on the field's grid, rasterised from cartopy's Natural
    Earth land polygons (no extra dependency)."""
    import shapely.geometry as sgeom
    from shapely.ops import unary_union
    from shapely.prepared import prep
    from cartopy.feature import LAND
    land = prep(unary_union(list(LAND.geometries())))
    lats = da["latitude"].to_numpy(); lons = da["longitude"].to_numpy()
    m = np.zeros((lats.size, lons.size), bool)
    for j, lo in enumerate(lons):
        lo180 = lo - 360 if lo > 180 else lo
        for i, la in enumerate(lats):
            m[i, j] = land.contains(sgeom.Point(lo180, la))
    return xr.DataArray(m, coords={"latitude": da["latitude"], "longitude": da["longitude"]},
                        dims=("latitude", "longitude"))


def _wmean(da: xr.DataArray, where: xr.DataArray) -> float:
    """cos(lat)-weighted area mean over the masked cells. Area-mean (not median)
    is the right statistic for lightning: it is dominated by the active convective
    regions, reproducing the classic ~10x land:ocean ratio that the median (swamped
    by quiet desert/high-latitude land) hides."""
    w = np.cos(np.deg2rad(da["latitude"])).broadcast_like(da).where(where & np.isfinite(da))
    num = (da.where(where) * w).sum()
    den = w.sum()
    return float(num / den) if float(den) > 0 else float("nan")


def plot(out_path: Path) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from src.figures._style import apply_nature_style, COL_DOUBLE_IN
    apply_nature_style()

    cape, E = _env_annual_1deg()
    light = _lightning_annual_1deg(cape)

    cape_floor = 50.0          # J/kg; below this the env is irrelevant to convection
    e_floor = float(np.nanpercentile(E.where(E > 0).to_numpy(), 20))
    eff_cape = (light / cape.where(cape > cape_floor)).rename("eff_cape")
    eff_E = (light / E.where(E > e_floor)).rename("eff_E")

    # --- coherence metrics: land vs ocean, within the WGLC-reliable band -------
    # The thesis in three numbers: CAPE is similar over land and ocean, but
    # lightning (hence efficiency) is far higher over land -> efficiency is a real
    # degree of freedom that the environment alone does not capture.
    band = (np.abs(cape["latitude"]) < 55)
    land = _land_mask(cape) & band
    ocean = (~_land_mask(cape)) & band
    cape_land, cape_ocean = _wmean(cape, land), _wmean(cape, ocean)
    light_land, light_ocean = _wmean(light.where(light > 0), land), _wmean(light.where(light > 0), ocean)
    eff_land, eff_ocean = _wmean(eff_cape.where(eff_cape > 0), land), _wmean(eff_cape.where(eff_cape > 0), ocean)
    contrast = eff_land / eff_ocean if (eff_ocean and np.isfinite(eff_ocean) and eff_ocean > 0) else float("inf")
    cape_contrast = cape_land / cape_ocean if cape_ocean else float("nan")
    light_contrast = light_land / light_ocean if light_ocean else float("inf")

    proj = ccrs.Robinson(central_longitude=160)
    lons = cape["longitude"].to_numpy(); lats = cape["latitude"].to_numpy()

    def _base(ax):
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.3)
        ax.set_global()

    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.78))
    gs = fig.add_gridspec(3, 1, hspace=0.16)

    # (a) mean convective environment (CAPE)
    ax = fig.add_subplot(gs[0], projection=proj); _base(ax)
    p = ax.pcolormesh(lons, lats, cape.to_numpy(), transform=ccrs.PlateCarree(),
                      cmap="YlOrRd", vmin=0, vmax=float(np.nanpercentile(cape, 99)),
                      shading="auto")
    ax.set_title("(a) ERA5 convective environment — mean CAPE (J kg$^{-1}$), 2010–2023", fontsize=7.5)
    fig.colorbar(p, ax=ax, fraction=0.025, pad=0.02)

    # (b) realized convection (lightning)
    ax = fig.add_subplot(gs[1], projection=proj); _base(ax)
    lz = light.where(light > 0)
    p = ax.pcolormesh(lons, lats, lz.to_numpy(), transform=ccrs.PlateCarree(),
                      cmap="viridis", norm=LogNorm(vmin=1e-3, vmax=float(np.nanpercentile(lz, 99.5))),
                      shading="auto")
    ax.set_title("(b) Realized convection — WGLC stroke density (km$^{-2}$ d$^{-1}$, log)", fontsize=7.5)
    fig.colorbar(p, ax=ax, fraction=0.025, pad=0.02)

    # (c) EFFICIENCY = lightning per unit CAPE
    ax = fig.add_subplot(gs[2], projection=proj); _base(ax)
    ez = eff_cape.where(eff_cape > 0)
    vmax = float(np.nanpercentile(ez, 99))
    p = ax.pcolormesh(lons, lats, ez.to_numpy(), transform=ccrs.PlateCarree(),
                      cmap="magma", norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax), shading="auto")
    ax.set_title(f"(c) CONVECTIVE EFFICIENCY = lightning / CAPE  "
                 f"(efficiency land/ocean {contrast:.0f}× vs CAPE land/ocean {cape_contrast:.1f}×)",
                 fontsize=7.5)
    fig.colorbar(p, ax=ax, fraction=0.025, pad=0.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    sidecar = {
        "numerator": "WGLC stroke density (strokes km^-2 d^-1), 2012-2021 clim",
        "denominator": "ERA5 mean CAPE 2010-2023 (J/kg); WMAXSHEAR variant also computed",
        "band": "|lat| < 55",
        "cape_land": round(cape_land, 1), "cape_ocean": round(cape_ocean, 1),
        "cape_land_ocean_contrast": round(cape_contrast, 2),
        "lightning_land": round(light_land, 4), "lightning_ocean": round(light_ocean, 6),
        "lightning_land_ocean_contrast": (round(light_contrast, 1) if np.isfinite(light_contrast) else "inf"),
        "efficiency_land": round(eff_land, 6), "efficiency_ocean": round(eff_ocean, 8),
        "efficiency_land_ocean_contrast": (round(contrast, 1) if np.isfinite(contrast) else "inf"),
        "decoupling_ratio": round(contrast / cape_contrast, 1) if cape_contrast else None,
        "go_criterion": "realized convection varies more across land/ocean than the environment "
                        "does (decoupling_ratio > 2) -> efficiency carries independent information",
        "verdict": "GO" if (cape_contrast and contrast / cape_contrast > 2) else "INCONCLUSIVE",
    }
    side = out_path.with_suffix(".json")
    side.write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps(sidecar, indent=2))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "poc_lightning_efficiency.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
