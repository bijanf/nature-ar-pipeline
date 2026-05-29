"""
Holton-dynamics feature pipeline for Atmospheric River modeling.

This module lazily streams ARCO-ERA5 from a public Zarr store on GCS, applies
the configured US-West-Coast bounding box at the load boundary, and then
constructs four physically motivated features used downstream by the LightGBM
emulator:

    1. Integrated Vapor Transport (IVT) — magnitude + zonal/meridional components.
    2. Equivalent potential temperature (theta_e) at 850 hPa — moist static energy.
    3. Quasi-geostrophic / Ertel potential vorticity at 250 hPa — upper-level
       synoptic forcing.
    4. Eady maximum growth rate (850-250 hPa) — baroclinic instability proxy.

Engineering invariants (per the project directive):

  * The pipeline is **lazy end-to-end**. Every operation produces a Dask graph,
    never a materialized array. Callers must call `.compute()` themselves at the
    final aggregation step.
  * All derivatives, Coriolis terms, and Brunt-Vaisala frequencies are computed
    via `MetPy`, which already handles spherical metric terms and unit tracking.
  * Pint quantities are stripped with `.metpy.dequantify()` before the dataset is
    handed to the ML stage — LightGBM cannot ingest quantified arrays.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import metpy.calc as mpcalc
import numpy as np
import xarray as xr

from src import config

# Mapping CDS GRIB shortNames -> ARCO-ERA5 long names. The local-NetCDF cache
# (src.data.fetch_cds_era5) writes variables under the shortNames; the rest of
# the pipeline addresses them through `config.ERA5_VARS` / `ERA5_SURFACE_VARS`,
# which use the ARCO long names. We rename at the load boundary so downstream
# code is source-agnostic.
_CDS_PL_RENAME = {
    "u": config.ERA5_VARS["u"],
    "v": config.ERA5_VARS["v"],
    "t": config.ERA5_VARS["t"],
    "q": config.ERA5_VARS["q"],
    "z": config.ERA5_VARS["z"],
}
_CDS_SFC_RENAME = {
    "sp": config.ERA5_VARS["sp"],
}
_CDS_TP_RENAME = {
    "tp": config.ERA5_SURFACE_VARS["tp"],
}
_CDS_STATIC_RENAME = {
    "lsm": config.ERA5_SURFACE_VARS["lsm"],
    "z": config.ERA5_SURFACE_VARS["z_sfc"],
}

# =============================================================================
# Cloud ingestion
# =============================================================================


def open_arco_era5(zarr_url: str = config.ERA5_ZARR_URL) -> xr.Dataset:
    """Open the ARCO-ERA5 Zarr store and apply the regional bounding box.
    Returns a fully lazy `xr.Dataset` backed by Dask.

    Three ingest modes, dispatched on the shape of ``zarr_url``:

    * remote (``"://" in zarr_url``) — anonymous Zarr open on GCS / S3 / HTTPS.
      Native Zarr chunks are preserved (``chunks={}``); Dask then layers its
      task graph on top of those chunk boundaries.
    * local Zarr — a directory ending in ``.zarr`` (or a single-Zarr layout
      produced by ``src.data.stage_arco_to_local``) — opened with no
      storage_options, otherwise identical.
    * local NetCDF cache — a directory containing CDS-fetched per-year files
      (``{YYYY}_pl.nc``, ``{YYYY}_sfc.nc``, ``static.nc`` from
      ``src.data.fetch_cds_era5``). Files are mfopened, GRIB shortNames are
      renamed to ARCO long names, and longitude is shifted to the 0..360
      convention so ``_apply_regional_bbox`` works unchanged.
    """
    # If the caller didn't override and ERA5_LOCAL_CACHE points at a populated
    # directory, prefer the local cache. Lets SLURM scripts opt in without
    # touching call sites in period_contrast / stage / inference modules.
    if zarr_url == config.ERA5_ZARR_URL:
        env_override = os.environ.get("ERA5_LOCAL_CACHE")
        if env_override:
            zarr_url = env_override

    if "://" in zarr_url:
        ds = xr.open_zarr(
            zarr_url,
            consolidated=True,
            storage_options={"token": "anon"},
            chunks={},
        )
    else:
        path = Path(zarr_url)
        if path.is_dir() and any(path.glob("*.nc")):
            ds = _open_local_netcdf_cache(path)
        else:
            ds = xr.open_zarr(zarr_url, consolidated=True, chunks={})
    ds = _apply_regional_bbox(ds)
    return _ensure_chunks(ds)


def _open_local_netcdf_cache(root: Path) -> xr.Dataset:
    """Load the per-year CDS NetCDF layout written by ``fetch_cds_era5``."""
    pl_files = sorted(root.glob("*_pl.nc"))
    sfc_files = sorted(root.glob("*_sfc.nc"))
    static_file = root / "static.nc"
    if not pl_files:
        raise FileNotFoundError(f"No *_pl.nc files under {root}")

    pl = xr.open_mfdataset(
        [str(p) for p in pl_files],
        combine="by_coords",
        chunks={},
        engine="netcdf4",
    ).rename({k: v for k, v in _CDS_PL_RENAME.items() if k in xr.open_dataset(pl_files[0]).data_vars})

    if sfc_files:
        sfc = xr.open_mfdataset(
            [str(p) for p in sfc_files],
            combine="by_coords",
            chunks={},
            engine="netcdf4",
        ).rename({k: v for k, v in _CDS_SFC_RENAME.items() if k in xr.open_dataset(sfc_files[0]).data_vars})
        merged = xr.merge([pl, sfc], compat="override")
    else:
        merged = pl

    # Shift CDS pl/sfc longitudes to 0..360 BEFORE merging in PIK tp.
    # CDS writes lon in -180..180; PIK is 0..360. The mod-360 transform below
    # is idempotent for already-0..360 grids, so it's safe to run unconditionally.
    lon_name_in = "longitude" if "longitude" in merged.coords else "lon"
    merged = merged.assign_coords({lon_name_in: merged[lon_name_in] % 360.0})
    merged = merged.sortby(lon_name_in)

    tp_files = sorted(root.glob("*_tp.nc"))
    pik_dir = os.environ.get("ERA5_TP_PIK_DIR", config.ERA5_TP_PIK_DIR)
    tp_from_pik = False
    if not tp_files and pik_dir and Path(pik_dir).is_dir():
        # PIK climate_data_central layout: total_precipitation_{YYYY}{MM}.nc.
        # Limit to years already in the local pl cache so we don't mfopen the
        # whole 1940-2024 archive when only a subset is on disk.
        years = sorted({Path(p).name.split("_")[0] for p in pl_files})
        pik_root = Path(pik_dir)
        for y in years:
            tp_files.extend(sorted(pik_root.glob(f"total_precipitation_{y}??.nc")))
        tp_from_pik = bool(tp_files)
    if tp_files:
        # PIK tp NetCDFs are GLOBAL 0.25° (721×1440); a naive
        # `combine="by_coords"` mfopen of 200+ such files blows RAM during
        # coord alignment. Bbox-subset each file in a `preprocess` callback so
        # only a ~140×160 slab is concat'd, and use `combine="nested"` with an
        # explicit time concat_dim to skip alignment. The local-cache CDS
        # `*_tp.nc` are already server-side bbox-sliced, so they stream
        # cheaply through the same path.
        time_dim = "valid_time"  # PIK and CDS both use valid_time
        if tp_from_pik:
            tp = xr.open_mfdataset(
                [str(p) for p in tp_files],
                combine="nested",
                concat_dim=time_dim,
                chunks={},
                engine="netcdf4",
                preprocess=_apply_regional_bbox,
                parallel=False,
            )
        else:
            tp = xr.open_mfdataset(
                [str(p) for p in tp_files],
                combine="by_coords",
                chunks={},
                engine="netcdf4",
            )
        tp = tp.rename({k: v for k, v in _CDS_TP_RENAME.items() if k in tp.data_vars})
        # tp is hourly while pl/sfc are 6-hourly. Resample tp to the
        # pl-time grid by summing each 6-hour window — total precip in
        # the preceding 6 h, ready for per-event Stage 2 aggregation.
        if time_dim not in tp.dims:
            time_dim = "time"
        tp = tp.resample({time_dim: "6h"}).sum()
        merged = xr.merge([merged, tp], compat="override", join="inner")

    if static_file.exists():
        static = xr.open_dataset(str(static_file), engine="netcdf4")
        static = static.rename({k: v for k, v in _CDS_STATIC_RENAME.items() if k in static.data_vars})
        if "valid_time" in static.dims:
            static = static.isel(valid_time=0).drop_vars("valid_time", errors="ignore")
        if "time" in static.dims:
            static = static.isel(time=0).drop_vars("time", errors="ignore")
        if "number" in static.dims:
            static = static.squeeze("number", drop=True)
        static_lon = "longitude" if "longitude" in static.coords else "lon"
        static = static.assign_coords({static_lon: static[static_lon] % 360.0}).sortby(static_lon)
        merged = xr.merge([merged, static], compat="override")

    if "valid_time" in merged.dims:
        merged = merged.rename({"valid_time": "time"})
    if "number" in merged.dims:
        merged = merged.squeeze("number", drop=True)
    if "pressure_level" in merged.dims:
        merged = merged.rename({"pressure_level": "level"})

    return merged


def _apply_regional_bbox(ds: xr.Dataset) -> xr.Dataset:
    """Slice to `config.BBOX`. Handles either ascending or descending latitude
    coordinates (ARCO-ERA5 stores latitude 90 -> -90)."""
    bbox = config.BBOX
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    lat_descending = float(ds[lat_name][0]) > float(ds[lat_name][-1])
    lat_slice = (
        slice(bbox["lat_max"], bbox["lat_min"])
        if lat_descending
        else slice(bbox["lat_min"], bbox["lat_max"])
    )
    lon_slice = slice(bbox["lon_min"], bbox["lon_max"])

    return ds.sel({lat_name: lat_slice, lon_name: lon_slice})


def _ensure_chunks(ds: xr.Dataset) -> xr.Dataset:
    """Apply project-default chunking only along dims that actually exist."""
    chunks = {d: c for d, c in config.DEFAULT_CHUNKS.items() if d in ds.dims}
    return ds.chunk(chunks) if chunks else ds


# =============================================================================
# Helpers
# =============================================================================

_G = 9.80665  # m s^-2, WMO standard gravity
_RD = 287.05  # J kg^-1 K^-1, dry-air gas constant


def _level_name(arr: xr.DataArray) -> str:
    for cand in ("level", "plev", "pressure_level"):
        if cand in arr.dims:
            return cand
    raise ValueError(f"No pressure-level dimension found on {arr.name!r}")


def _lat_name(arr: xr.DataArray) -> str:
    return "latitude" if "latitude" in arr.coords else "lat"


# =============================================================================
# Feature 1 — Integrated Vapor Transport (IVT)
# =============================================================================


def integrated_vapor_transport(ds: xr.Dataset) -> xr.Dataset:
    """Vertically integrate q*u and q*v with the standard -dp/g weighting to
    produce the IVT vector and its magnitude.

    Units: kg m^-1 s^-1. The integral is computed with `xarray.integrate`,
    which dispatches to a Dask-friendly trapezoidal rule along the pressure
    axis without breaking chunk boundaries.
    """
    q = ds[config.ERA5_VARS["q"]]
    u = ds[config.ERA5_VARS["u"]]
    v = ds[config.ERA5_VARS["v"]]
    level = _level_name(q)

    # Sort so pressure increases with index (surface to top -> integration sign
    # convention works out with the -1/g factor for column-integrated transport).
    q = q.sortby(level)
    u = u.sortby(level)
    v = v.sortby(level)

    # Convert pressure coord hPa -> Pa for SI consistency.
    p_pa = q[level].astype("float32") * 100.0
    q = q.assign_coords({level: p_pa})
    u = u.assign_coords({level: p_pa})
    v = v.assign_coords({level: p_pa})

    ivt_u = -(q * u).integrate(level) / _G
    ivt_v = -(q * v).integrate(level) / _G
    ivt = xr.apply_ufunc(np.hypot, ivt_u, ivt_v, dask="allowed")

    return xr.Dataset(
        {
            "ivt_u": ivt_u.astype("float32"),
            "ivt_v": ivt_v.astype("float32"),
            "ivt": ivt.astype("float32"),
        }
    )


# =============================================================================
# Feature 2 — Equivalent potential temperature at 850 hPa
# =============================================================================


def equivalent_potential_temperature_850(ds: xr.Dataset) -> xr.DataArray:
    """MetPy theta_e on the 850 hPa surface. The computation goes:

        q  -> dewpoint  ->  theta_e

    All MetPy calls preserve Dask backing because they vectorize through
    xarray's `apply_ufunc`. Pint units are stripped on return.
    """
    level = _level_name(ds[config.ERA5_VARS["t"]])
    t = ds[config.ERA5_VARS["t"]].sel({level: 850})
    q = ds[config.ERA5_VARS["q"]].sel({level: 850})

    p = xr.full_like(t, 850.0).assign_attrs(units="hPa").metpy.quantify()
    t_q = t.assign_attrs(units="K").metpy.quantify()
    q_q = q.assign_attrs(units="kg/kg").metpy.quantify()

    td = mpcalc.dewpoint_from_specific_humidity(p, t_q, q_q)
    theta_e = mpcalc.equivalent_potential_temperature(p, t_q, td)

    return theta_e.metpy.dequantify().astype("float32").rename("theta_e_850")


# =============================================================================
# Feature 3 — Potential vorticity at 250 hPa
# =============================================================================


def potential_vorticity_250(ds: xr.Dataset) -> xr.DataArray:
    """Ertel baroclinic PV at 250 hPa, as a strong proxy for QG PV / upper-level
    synoptic forcing. MetPy's `potential_vorticity_baroclinic` already handles
    spherical metric terms, Coriolis, and the d(theta)/dp factor.
    """
    level = _level_name(ds[config.ERA5_VARS["t"]])
    t = ds[config.ERA5_VARS["t"]].sortby(level)
    u = ds[config.ERA5_VARS["u"]].sortby(level)
    v = ds[config.ERA5_VARS["v"]].sortby(level)

    p = (t[level].astype("float32")).assign_attrs(units="hPa").metpy.quantify()
    t_q = t.assign_attrs(units="K").metpy.quantify()
    u_q = u.assign_attrs(units="m/s").metpy.quantify()
    v_q = v.assign_attrs(units="m/s").metpy.quantify()

    theta = mpcalc.potential_temperature(p, t_q)
    pv = mpcalc.potential_vorticity_baroclinic(theta, p, u_q, v_q)

    return pv.sel({level: 250}).metpy.dequantify().astype("float32").rename("pv_250")


# =============================================================================
# Feature 4 — Eady maximum growth rate (850-250 hPa)
# =============================================================================


def eady_growth_rate(ds: xr.Dataset) -> xr.DataArray:
    """Eady maximum growth rate:

        sigma_BI = 0.31 * |f / N| * |dU/dz|

    Here:
      * f is the Coriolis parameter (MetPy, latitude-dependent).
      * N  is the column-mean Brunt-Vaisala frequency (MetPy, on heights
        derived from geopotential).
      * |dU/dz| is the bulk vector wind shear between 850 hPa and 250 hPa.

    The bulk shear divided by layer thickness is a two-point finite difference
    of the velocity field along the vertical, which is the standard Eady-layer
    formulation; the spherical derivatives are not invoked here.
    """
    level = _level_name(ds[config.ERA5_VARS["t"]])
    t = ds[config.ERA5_VARS["t"]].sortby(level)
    u = ds[config.ERA5_VARS["u"]].sortby(level)
    v = ds[config.ERA5_VARS["v"]].sortby(level)
    z_geop = ds[config.ERA5_VARS["z"]].sortby(level)  # geopotential m^2/s^2

    # Heights from geopotential.
    z = (z_geop / _G).assign_attrs(units="m")

    p = (t[level].astype("float32")).assign_attrs(units="hPa").metpy.quantify()
    t_q = t.assign_attrs(units="K").metpy.quantify()
    theta = mpcalc.potential_temperature(p, t_q)

    z_q = z.metpy.quantify()
    n_sq = mpcalc.brunt_vaisala_frequency_squared(z_q, theta, vertical_dim=-3)
    n_bar = xr.apply_ufunc(np.sqrt, n_sq.metpy.dequantify().mean(dim=level), dask="allowed")

    # Bulk vector shear between 850 and 250 hPa (m/s per m).
    u850 = u.sel({level: 850})
    u250 = u.sel({level: 250})
    v850 = v.sel({level: 850})
    v250 = v.sel({level: 250})
    z850 = z.sel({level: 850})
    z250 = z.sel({level: 250})
    dz = z250 - z850
    du_dz = (u250 - u850) / dz
    dv_dz = (v250 - v850) / dz
    shear_mag = xr.apply_ufunc(np.hypot, du_dz, dv_dz, dask="allowed")

    # Coriolis from MetPy — latitude in degrees.
    lat = t[_lat_name(t)]
    f = mpcalc.coriolis_parameter(lat.assign_attrs(units="degrees").metpy.quantify())
    f_arr = f.metpy.dequantify()

    eady = 0.31 * np.abs(f_arr) / n_bar * shear_mag
    return eady.astype("float32").rename("eady_growth_rate")


# =============================================================================
# Public entry point
# =============================================================================


def calculate_dynamics(ds: xr.Dataset) -> xr.Dataset:
    """Build the full Holton-dynamics feature dataset on top of a lazily
    loaded ERA5 (or CMIP6) cube.

    Returns a single Dask-backed `xr.Dataset` with:

        ivt, ivt_u, ivt_v, theta_e_850, pv_250, eady_growth_rate

    All variables are float32, unitless (pint stripped), and chunked along the
    time axis. No `.compute()` is invoked.
    """
    ivt = integrated_vapor_transport(ds)
    theta_e = equivalent_potential_temperature_850(ds)
    pv = potential_vorticity_250(ds)
    eady = eady_growth_rate(ds)

    features = xr.merge([ivt, theta_e, pv, eady], compat="override")
    for v in features.data_vars:
        features[v].attrs.pop("units", None)
    return features


def required_era5_vars() -> Iterable[str]:
    """Names of ARCO-ERA5 variables this module touches — useful for the
    downstream Stage-1 loader when subsetting to keep the Dask graph small."""
    return (
        config.ERA5_VARS["u"],
        config.ERA5_VARS["v"],
        config.ERA5_VARS["t"],
        config.ERA5_VARS["q"],
        config.ERA5_VARS["z"],
    )


if __name__ == "__main__":
    # Smoke test only. Does NOT trigger computation — prints the lazy graph.
    ds = open_arco_era5()[list(required_era5_vars())]
    feats = calculate_dynamics(ds)
    print(feats)
