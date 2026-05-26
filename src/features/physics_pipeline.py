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

from collections.abc import Iterable

import metpy.calc as mpcalc
import numpy as np
import xarray as xr

from src import config

# =============================================================================
# Cloud ingestion
# =============================================================================


def open_arco_era5(zarr_url: str = config.ERA5_ZARR_URL) -> xr.Dataset:
    """Open the ARCO-ERA5 Zarr store on GCS with anonymous access and apply the
    regional bounding box. Returns a fully lazy `xr.Dataset` backed by Dask.

    The store is `consolidated=True` so we get a single metadata round-trip.
    Native Zarr chunks are preserved (`chunks={}`); Dask then layers its task
    graph on top of those chunk boundaries.
    """
    ds = xr.open_zarr(
        zarr_url,
        consolidated=True,
        storage_options={"token": "anon"},
        chunks={},
    )
    ds = _apply_regional_bbox(ds)
    return _ensure_chunks(ds)


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
