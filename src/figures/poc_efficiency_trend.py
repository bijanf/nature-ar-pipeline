"""NOVEL ARTIFACT — trend in convective EFFICIENCY, 2012-2023, network-drift-aware.

The climatology (poc_lightning_efficiency) reproduces known physics; the novel,
unpublished question is whether realized convection per unit favorable environment
is *trending* — and whether that trend agrees with or departs from the trend in
the environment itself (a thermodynamic-vs-efficiency decoupling in time).

Two honesty constraints drive the design:
  1. WWLLN detection efficiency grew over time. Global-mean WGLC density jumps
     +31% from 2010->2012 (network build-out), so 2010-2011 are DROPPED and the
     analysis runs 2012-2023.
  2. A residual, broadly spatially-uniform DE drift remains after 2012. We remove
     it with a COMMON-MODE normalization: divide each year's lightning by the
     global-tropical-land mean lightning of that year (renormalised to its own
     mean). A spatially-uniform DE inflation cancels; what survives is the
     region-SPECIFIC efficiency change that the network drift cannot fake.

Reported per region: the ENVIRONMENT trend (ERA5 WMAXSHEAR E, clean, no DE issue)
vs the COMMON-MODE-REMOVED efficiency trend (Theil-Sen %/decade + Mann-Kendall p).
The verdict is deliberately allowed to come back "confounded / too short" — that
is itself the finding that would steer the paper toward a climatology + ENSO +
CMIP6-projection framing rather than an observed-trend headline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.figures.poc_lightning_efficiency import _land_mask, _coord

_WGLC_TS = Path("/p/projects/poem/fallah/nature_ar_data/wglc/wglc_timeseries_30m_monthly.nc")
_CONV = Path("/p/projects/poem/fallah/nature_ar_data/era5_convective_monthly")
_WIND = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")
# (conv, wind, year-range that this file should contribute to 2012-2023)
_ENV_PAIRS = [
    ("gap2_2000_2014_conv_monthly_global.nc", "gap2_2000_2014_monthly_global.nc", 2012, 2014),
    ("recent_conv_monthly_global.nc", "recent_monthly_global.nc", 2015, 2023),
]
_Y0, _Y1 = 2012, 2023

# Convective regions: (name, lon0, lon1, lat0, lat1) in 0..360 longitude.
_REGIONS = [
    ("CONUS", 255, 285, 30, 48),
    ("South America", 290, 315, -35, -5),
    ("Central Africa", 10, 35, -10, 10),
    ("South Asia", 70, 90, 10, 30),
    ("Maritime Cont.", 95, 150, -10, 10),
]


def _env_annual_cube() -> xr.DataArray:
    """Annual-mean WMAXSHEAR E(year, lat, lon) on the ERA5 1-deg grid, 2012-2023."""
    Es = []
    for cf, wf, y0, y1 in _ENV_PAIRS:
        cds = xr.open_dataset(_CONV / cf); wds = xr.open_dataset(_WIND / wf)
        ct = _coord(cds, "valid_time", "time"); wt = _coord(wds, "valid_time", "time")
        cape = cds["cape"].clip(min=0.0).rename({ct: "time"}).assign_coords(time=cds[ct].values)
        lev = "pressure_level" if "pressure_level" in wds.dims else "level"
        u5, u8 = wds["u"].sel({lev: 500}), wds["u"].sel({lev: 850})
        v5, v8 = wds["v"].sel({lev: 500}), wds["v"].sel({lev: 850})
        shear = np.hypot(u5 - u8, v5 - v8).rename({wt: "time"}).assign_coords(time=wds[wt].values)
        shear = shear.reindex(time=cape["time"], method="nearest", tolerance=np.timedelta64(20, "D"))
        E = (np.sqrt(2.0 * cape) * shear).rename("E")
        yr = E["time"].dt.year
        E = E.sel(time=(yr >= y0) & (yr <= y1))
        Es.append(E.groupby("time.year").mean("time"))
    return xr.concat(Es, dim="year").sortby("year")


def _lightning_annual_cube(target: xr.DataArray) -> xr.DataArray:
    """Annual-mean WGLC density(year, lat, lon) regridded to the ERA5 grid, 2012-2023."""
    ds = xr.open_dataset(_WGLC_TS)
    latn = _coord(ds, "lat", "latitude"); lonn = _coord(ds, "lon", "longitude")
    da = ds["density"].rename({latn: "latitude", lonn: "longitude"})
    if float(da["longitude"].min()) < 0:
        da = da.assign_coords(longitude=(da["longitude"] % 360)).sortby("longitude")
    da = da.sortby("latitude")
    ann = da.groupby("time.year").mean("time")
    ann = ann.sel(year=(ann["year"] >= _Y0) & (ann["year"] <= _Y1))
    return ann.interp(latitude=target["latitude"], longitude=target["longitude"], method="linear")


def _region_series(da: xr.DataArray, box, landm: xr.DataArray) -> np.ndarray:
    lon0, lon1, lat0, lat1 = box
    sub = da.sel(longitude=slice(lon0, lon1), latitude=slice(lat1, lat0))
    lm = landm.sel(longitude=slice(lon0, lon1), latitude=slice(lat1, lat0))
    w = np.cos(np.deg2rad(sub["latitude"])).broadcast_like(sub.isel(year=0)).where(lm)
    num = (sub.where(lm) * w).sum(("latitude", "longitude"))
    den = w.sum(("latitude", "longitude"))
    return (num / den).to_numpy()


def _trend(years, y):
    from scipy.stats import theilslopes, kendalltau
    m = np.isfinite(y)
    if m.sum() < 5:
        return float("nan"), float("nan")
    slope, intercept, _, _ = theilslopes(y[m], years[m])
    _, p = kendalltau(years[m], y[m])
    pct_decade = 100.0 * slope * 10.0 / np.nanmean(y[m])   # %/decade relative to its own mean
    return pct_decade, p


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    from src.figures._style import apply_nature_style, COL_DOUBLE_IN
    apply_nature_style()

    E = _env_annual_cube()
    L = _lightning_annual_cube(E)
    E, L = xr.align(E, L, join="inner")          # common years
    years = E["year"].to_numpy().astype(int)
    landm = _land_mask(E.isel(year=0))

    # global tropical-land common mode (the DE-drift + global-climate index)
    trop = landm & (np.abs(E["latitude"]) <= 30)
    wL = np.cos(np.deg2rad(L["latitude"])).broadcast_like(L.isel(year=0)).where(trop)
    common = ((L.where(trop) * wL).sum(("latitude", "longitude")) / wL.sum(("latitude", "longitude"))).to_numpy()
    common_norm = common / np.nanmean(common)    # renormalised multiplicative drift factor
    common_trend, common_p = _trend(years, common)

    rows = []
    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.42))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(_REGIONS)))

    for (name, *box), c in zip(_REGIONS, colors):
        e_ser = _region_series(E, box, landm)
        l_ser = _region_series(L, box, landm)
        eff_raw = l_ser / e_ser
        eff_norm = eff_raw / common_norm          # remove the network/global common mode
        env_tr, env_p = _trend(years, e_ser)
        eff_raw_tr, eff_raw_p = _trend(years, eff_raw)
        eff_tr, eff_p = _trend(years, eff_norm)
        rows.append(dict(region=name, env_trend=env_tr, env_p=env_p,
                         eff_raw_trend=eff_raw_tr, eff_norm_trend=eff_tr, eff_norm_p=eff_p))
        ax.plot(years, eff_norm / np.nanmean(eff_norm), color=c, lw=1.1, marker="o", ms=2.5, label=name)

    ax.axhline(1.0, color="0.6", lw=0.5, ls=":")
    ax.set_xlabel("year", fontsize=8)
    ax.set_ylabel("normalised efficiency\n(common-mode removed)", fontsize=7.5)
    ax.set_title(f"(a) Regional convective-efficiency trajectories, {_Y0}–{_Y1}", fontsize=7.5)
    ax.legend(fontsize=5.5, frameon=False, ncol=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # (b) environment trend vs common-mode-removed efficiency trend, per region
    ax2 = fig.add_subplot(gs[0, 1])
    names = [r["region"] for r in rows]
    yy = np.arange(len(names))
    ax2.barh(yy - 0.2, [r["env_trend"] for r in rows], height=0.4, color="#e67e22", label="environment (E)")
    ax2.barh(yy + 0.2, [r["eff_norm_trend"] for r in rows], height=0.4, color="#8e44ad", label="efficiency (norm.)")
    for i, r in enumerate(rows):           # mark MK-significant efficiency trends
        if np.isfinite(r["eff_norm_p"]) and r["eff_norm_p"] < 0.1:
            ax2.text(r["eff_norm_trend"], i + 0.2, "*", fontsize=9, va="center")
    ax2.axvline(0, color="0.4", lw=0.6)
    ax2.set_yticks(yy); ax2.set_yticklabels(names, fontsize=6.5)
    ax2.set_xlabel("trend (%/decade)", fontsize=7.5)
    ax2.set_title("(b) environment vs efficiency trend", fontsize=7.5)
    ax2.legend(fontsize=5.5, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # verdict: is any common-mode-removed efficiency trend robust (MK p<0.1)?
    n_sig = sum(1 for r in rows if np.isfinite(r["eff_norm_p"]) and r["eff_norm_p"] < 0.1)
    sidecar = {
        "period": f"{_Y0}-{_Y1}",
        "n_years": int(len(years)),
        "de_build_out_dropped": "2010-2011 (global mean +31% 2010->2012)",
        "common_mode_trend_pct_decade": round(common_trend, 1),
        "common_mode_mk_p": round(float(common_p), 3),
        "regions": [{k: (round(v, 1) if isinstance(v, float) and np.isfinite(v) else v)
                     for k, v in r.items()} for r in rows],
        "n_regions_sig_efficiency_trend": n_sig,
        "verdict": ("ROBUST differential efficiency trend" if n_sig >= 2
                    else "CONFOUNDED/TOO-SHORT — efficiency trend not separable from network "
                         "drift over 12 yr; paper headline should be climatology + ENSO + CMIP6"),
    }
    side = out_path.with_suffix(".json")
    side.write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps(sidecar, indent=2))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "poc_efficiency_trend.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
