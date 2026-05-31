"""Per-window time-mean spatial fields for the driver-attribution story.

The period-contrast analysis collapses the 4-D ERA5 fields (u, v, t, q, z at
850/500/250 hPa, 6-hourly) into per-event scalars. That throws away *where* and
*why* the moisture transport changed. This module keeps the spatial structure:
for each observational window it computes the time-mean of

    q, u, v          (per pressure level)            -> mean state
    q*u, q*v         (per pressure level)             -> full (incl. eddy) flux
    ivt, ivt_u, ivt_v                                 -> column IVT vector
    theta_e_850                                       -> low-level moist energy

from which Fig. "drivers" builds (a) the spatial IVT-change map and (b) the
thermodynamic-vs-dynamic decomposition of the inter-window IVT change:

    Δ<q v> ≈ Δ<q>·<v>   (thermodynamic, moisture change at fixed circulation)
           + <q>·Δ<v>   (dynamic, circulation change at fixed moisture)
           + residual   (nonlinear + transient-eddy change)

Memory: identical envelope to the climatology — each year's fields are computed
and reduced (summed over time) independently, so the peak working set is one
year of pressure-level data (~6 GB), never the full window.

Output: ``data/cache/spatial_composite_<short>.nc`` (one per window), tiny
(level × lat × lon), plus a combined ``spatial_composites.nc``.
"""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

from src import config
from src.analysis.period_contrast import _open_raw_era5
from src.features import physics_pipeline

_G = 9.80665

# Map the config period names to short tags used in the cache filenames.
_SHORT = {
    "pre_sat_1940_1959": "presat",
    "modern_1980_1999": "modern",
    "recent_2015_2024": "recent",
}


def _level_name(da: xr.DataArray) -> str:
    for cand in ("level", "pressure_level", "plev", "isobaricInhPa"):
        if cand in da.dims:
            return cand
    raise KeyError(f"no pressure-level dim in {da.dims}")


def _window_time_means(period: tuple[str, str]) -> xr.Dataset:
    """Time-mean q, u, v, q*u, q*v (per level) + IVT vector + theta_e_850 over
    ``period``, accumulated one year at a time so memory stays bounded."""
    ds = _open_raw_era5(period)
    q = ds[config.ERA5_VARS["q"]]
    u = ds[config.ERA5_VARS["u"]]
    v = ds[config.ERA5_VARS["v"]]
    t = ds[config.ERA5_VARS["t"]]
    z = ds[config.ERA5_VARS["z"]]
    lev = _level_name(q)

    ivt_ds = physics_pipeline.integrated_vapor_transport(ds)
    theta_e = physics_pipeline.equivalent_potential_temperature_850(ds)

    # Lazy per-timestep fields to be time-averaged.
    fields = {
        "qbar": q,
        "ubar": u,
        "vbar": v,
        "tbar": t,  # mean temperature per level -> Clausius-Clapeyron ΔT
        "zbar": z,  # geopotential per level -> large-scale circulation
        "qu_bar": q * u,
        "qv_bar": q * v,
        "ivt": ivt_ds["ivt"],
        "ivt_u": ivt_ds["ivt_u"],
        "ivt_v": ivt_ds["ivt_v"],
        "theta_e_850": theta_e,
    }

    y0, y1 = int(period[0][:4]), int(period[1][:4])
    acc: dict[str, xr.DataArray] = {}
    n = 0
    ivt_years: list[xr.DataArray] = []  # per-year annual-mean IVT for significance
    for y in range(y0, y1 + 1):
        yr = slice(f"{y}-01-01", f"{y}-12-31")
        nt = int(ds.sel(time=yr).sizes.get("time", 0))
        if not nt:
            continue
        for name, da in fields.items():
            s = da.sel(time=yr).sum("time").compute()
            acc[name] = s if name not in acc else acc[name] + s
        ivt_years.append(
            ivt_ds["ivt"]
            .sel(time=yr)
            .mean("time")
            .compute()
            .expand_dims(year=[y])
            .astype("float32")
        )
        n += nt

    out = {name: (acc[name] / n).astype("float32") for name in acc}
    ds_out = xr.Dataset(out)
    ds_out["ivt_mag_from_mean"] = np.hypot(ds_out["ivt_u"], ds_out["ivt_v"]).astype("float32")
    ds_out.attrs["period"] = f"{period[0]}:{period[1]}"
    ds_out.attrs["n_timesteps"] = n
    ds_out.attrs["pressure_level_name"] = lev
    # Per-year annual-mean IVT (year, lat, lon) for the significance test. Kept
    # separate because the year dimension differs per window (20/20/10).
    peryear = xr.concat(ivt_years, dim="year").rename("ivt_annual")
    return ds_out, peryear


def run(periods=None, out_dir=None) -> list:
    periods = periods or list(config.OBSERVATIONAL_PERIODS)
    out_dir = out_dir or config.CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    per_window = {}
    for name, period in periods:
        short = _SHORT.get(name, name)
        comp, peryear = _window_time_means(tuple(period))
        comp = comp.expand_dims(window=[short])
        path = out_dir / f"spatial_composite_{short}.nc"
        comp.to_netcdf(path)
        peryear.to_netcdf(out_dir / f"spatial_peryear_{short}.nc")
        written.append(path)
        per_window[short] = comp
        print(f"wrote {path}  (n_timesteps={int(comp.attrs['n_timesteps'])})")
    if len(per_window) > 1:
        combined = xr.concat(list(per_window.values()), dim="window")
        cpath = out_dir / "spatial_composites.nc"
        combined.to_netcdf(cpath)
        written.append(cpath)
        print(f"wrote {cpath}  (windows={list(per_window)})")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--periods",
        nargs="+",
        default=None,
        help="Subset of period names (default: all observational windows).",
    )
    args = ap.parse_args()
    if args.periods:
        periods = [(n, p) for n, p in config.OBSERVATIONAL_PERIODS if n in args.periods]
    else:
        periods = list(config.OBSERVATIONAL_PERIODS)
    run(periods=periods)


if __name__ == "__main__":
    main()
