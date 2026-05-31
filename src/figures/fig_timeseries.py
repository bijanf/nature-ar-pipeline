"""Figure — Continuous 1940–2024 corridor IVT trajectory.

The three windows are samples; this is the whole record. From the global
monthly-mean files (windows + gap decades) we build the annual-mean vertically
integrated moisture transport averaged over the US West Coast corridor box, fit
a Theil–Sen trend with a Mann–Kendall significance test, and estimate a
time-of-emergence (the year after which the smoothed series stays above the
pre-satellite mean + 1 s.d.). The three analysis windows are shaded for
reference, so the reader sees that the discrete contrast is a faithful sample of
a monotonic continuous rise.

Caveat (as for Fig. 5): monthly means resolve the time-mean flow, not the
synoptic transient transport — this is the large-scale moisture-transport
trajectory, complementary to the event-based windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src import config
from src.figures._style import COL_DOUBLE_IN, apply_nature_style

_G = 9.80665
_DIR = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")
_FILES = [
    "presat_monthly_global.nc",
    "gap1_1960_1979_monthly_global.nc",
    "modern_monthly_global.nc",
    "gap2_2000_2014_monthly_global.nc",
    "recent_monthly_global.nc",
]
_WINDOWS = [(1940, 1959), (1980, 1999), (2015, 2024)]


def _corridor_ivt_monthly(path: Path) -> xr.DataArray:
    """Monthly corridor-box-mean IVT magnitude (kg m^-1 s^-1) from one file."""
    ds = xr.open_dataset(path)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    lev = "pressure_level" if "pressure_level" in ds.dims else "level"
    # Corridor box (config.BBOX is 0..360 longitude; files are 0..359).
    lon0, lon1 = config.BBOX["lon_min"], config.BBOX["lon_max"]
    lat0, lat1 = config.BBOX["lat_min"], config.BBOX["lat_max"]
    sub = ds.sel(longitude=slice(lon0, lon1), latitude=slice(lat1, lat0))  # lat descending in ERA5
    q = sub["q"].sortby(lev)
    u = sub["u"].sortby(lev)
    v = sub["v"].sortby(lev)
    q = q.assign_coords({lev: q[lev].astype("float64") * 100.0})
    u = u.assign_coords({lev: u[lev].astype("float64") * 100.0})
    v = v.assign_coords({lev: v[lev].astype("float64") * 100.0})
    ivt_u = -(q * u).integrate(lev) / _G
    ivt_v = -(q * v).integrate(lev) / _G
    ivt = np.hypot(ivt_u, ivt_v)
    w = np.cos(np.deg2rad(sub["latitude"]))
    corridor = ivt.weighted(w).mean(("latitude", "longitude"))
    return corridor.rename("ivt").assign_coords({tdim: ds[tdim]}).rename({tdim: "time"})


def _annual_series() -> pd.Series:
    parts = []
    for f in _FILES:
        p = _DIR / f
        if p.exists():
            parts.append(_corridor_ivt_monthly(p))
    if not parts:
        raise FileNotFoundError("no monthly global files found")
    monthly = xr.concat(parts, dim="time").sortby("time")
    ann = monthly.groupby("time.year").mean("time")
    return pd.Series(ann.to_numpy(), index=ann["year"].to_numpy(), name="ivt")


def _theil_sen(x, y):
    from scipy.stats import kendalltau, theilslopes

    slope, intercept, lo, hi = theilslopes(y, x)
    _tau, p = kendalltau(x, y)
    return slope, intercept, lo, hi, p


def _emergence(years, y, base_mask, intercept, slope):
    """Time-of-emergence: the first year the Theil–Sen trend line rises above the
    pre-satellite baseline mean + 1 s.d. of interannual variability (i.e. the
    forced signal exceeds the natural noise)."""
    base = y[base_mask]
    thr = base.mean() + base.std()
    trend = intercept + slope * years
    above = np.where(trend > thr)[0]
    return (int(years[above[0]]) if above.size else None), thr


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    s = _annual_series()
    years = s.index.to_numpy().astype(int)
    y = s.to_numpy()

    slope, intercept, _lo, _hi, p = _theil_sen(years, y)
    per_decade = slope * 10.0
    base_mask = years <= 1959
    emerge_year, thr = _emergence(years, y, base_mask, intercept, slope)

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.42))
    for a, b in _WINDOWS:
        ax.axvspan(a, b, color="0.88", zorder=0)
    ax.plot(years, y, color="#2c3e50", lw=0.8, marker="o", ms=2, label="annual corridor IVT")
    sm = pd.Series(y, index=years).rolling(7, center=True, min_periods=4).mean()
    ax.plot(years, sm.to_numpy(), color="#c0392b", lw=1.6, label="7-yr running mean")
    ax.plot(
        years,
        intercept + slope * years,
        color="#2980b9",
        lw=1.3,
        ls="--",
        label=f"Theil–Sen {per_decade:+.2f} kg m$^{{-1}}$ s$^{{-1}}$/decade (p={p:.1e})",
    )
    ax.axhline(thr, color="0.5", lw=0.6, ls=":")
    if emerge_year:
        ax.axvline(emerge_year, color="#16a085", lw=1.0)
        ax.annotate(
            f"emergence ≈ {emerge_year}",
            xy=(emerge_year, ax.get_ylim()[1]),
            xytext=(emerge_year + 1, y.min() + 0.6 * (y.max() - y.min())),
            fontsize=7,
            color="#16a085",
        )
    ax.set_xlabel("year", fontsize=8.5)
    ax.set_ylabel("corridor mean IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=8.5)
    ax.set_xlim(years.min(), years.max())
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    # (no in-figure title; described in the caption)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    import json

    (config.CACHE_DIR / "timeseries.json").write_text(
        json.dumps(
            {
                "trend_per_decade": round(per_decade, 2),
                "mk_p": float(f"{p:.2e}"),
                "emergence_year": emerge_year,
                "n_years": len(years),
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    print(
        f"Theil-Sen {per_decade:+.3f}/decade, MK p={p:.2e}, emergence={emerge_year}, "
        f"years={years.min()}-{years.max()} ({len(years)})"
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_timeseries.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
