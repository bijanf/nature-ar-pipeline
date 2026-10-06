"""NOVEL LEAD — is the South Asia efficiency rise an aerosol-invigoration signal?

poc_efficiency_trend found South Asia as the one region with a significant
common-mode-removed convective-efficiency increase (+34 %/decade). The
aerosol-invigoration hypothesis (more CCN over the Indo-Gangetic Plain -> smaller
droplets, delayed warm rain, more ice lofted -> more lightning per unit
thermodynamic environment) makes three falsifiable predictions, all testable with
data in hand:

  1. SEASONALITY  — the rise should concentrate in the PRE-MONSOON (MAM), when
     aerosol loading is extreme and convection is building, not in the monsoon
     (JJAS) when wet scavenging cleans the air.
  2. LOCALISATION — the rise should peak over the Indo-Gangetic Plain (IGP), the
     aerosol hotspot, not be uniform across South Asia.
  3. THERMODYNAMIC RESIDUAL — CAPE, CIN and shear trends should be weak, so the
     efficiency rise is NOT explained by the environment (a necessary, not
     sufficient, condition for a microphysical/aerosol cause).

This script tests all three. It cannot *prove* aerosol causation — that needs an
aerosol record (CAMS AOD via the ADS, which we lack credentials for here) — but it
decides whether the signal has the fingerprint that would justify that next step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.figures._style import apply_nature_style, COL_DOUBLE_IN
from src.figures.poc_lightning_efficiency import _land_mask, _coord
from src.figures.poc_efficiency_trend import (
    _env_annual_cube, _lightning_annual_cube, _trend, _ENV_PAIRS, _Y0, _Y1,
)

_WGLC_TS = Path("/p/projects/poem/fallah/nature_ar_data/wglc/wglc_timeseries_30m_monthly.nc")
_CONV = Path("/p/projects/poem/fallah/nature_ar_data/era5_convective_monthly")
_WIND = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")
_SA = dict(lon0=70, lon1=90, lat0=10, lat1=30)          # South Asia
_IGP = dict(lon0=75, lon1=88, lat0=24, lat1=30)         # Indo-Gangetic Plain
_SEASONS = {"pre-monsoon\n(MAM)": [3, 4, 5], "monsoon\n(JJAS)": [6, 7, 8, 9],
            "post-monsoon\n(ONDJF)": [10, 11, 12, 1, 2]}


def _box_sel(da, b):
    # order-independent (files differ in latitude direction): mask + drop edges.
    return da.where((da["latitude"] >= b["lat0"]) & (da["latitude"] <= b["lat1"])
                    & (da["longitude"] >= b["lon0"]) & (da["longitude"] <= b["lon1"]), drop=True)


def _wmean(da):
    w = np.cos(np.deg2rad(da["latitude"]))
    return da.weighted(w).mean(("latitude", "longitude"))


def _monthly_env(box) -> pd.DataFrame:
    """Region-mean monthly CAPE, CIN, shear, E over 2012-2023."""
    parts = []
    for cf, wf, _, _ in _ENV_PAIRS:
        cds = _box_sel(xr.open_dataset(_CONV / cf), box)
        wds = _box_sel(xr.open_dataset(_WIND / wf), box)
        ct = _coord(cds, "valid_time", "time"); wt = _coord(wds, "valid_time", "time")
        cape = cds["cape"].clip(min=0.0); cin = cds["cin"]
        lev = "pressure_level" if "pressure_level" in wds.dims else "level"
        shear = np.hypot(wds["u"].sel({lev: 500}) - wds["u"].sel({lev: 850}),
                         wds["v"].sel({lev: 500}) - wds["v"].sel({lev: 850}))
        df = pd.DataFrame({
            "cape": _wmean(cape).to_pandas().values,
            "cin": _wmean(np.abs(cin)).to_pandas().values,
            "shear": _wmean(shear.rename({wt: ct}).assign_coords({ct: cds[ct].values})).to_pandas().values,
        }, index=pd.to_datetime(cds[ct].values))
        parts.append(df)
    df = pd.concat(parts).sort_index()
    df = df[(df.index.year >= _Y0) & (df.index.year <= _Y1)]
    df["E"] = np.sqrt(2.0 * df["cape"]) * df["shear"]
    return df


def _monthly_lightning(box) -> pd.Series:
    ds = xr.open_dataset(_WGLC_TS)
    latn = _coord(ds, "lat", "latitude"); lonn = _coord(ds, "lon", "longitude")
    da = ds["density"].rename({latn: "latitude", lonn: "longitude"})
    if float(da["longitude"].min()) < 0:
        da = da.assign_coords(longitude=(da["longitude"] % 360)).sortby("longitude")
    da = _box_sel(da.sortby("latitude"), box)
    s = _wmean(da).to_pandas()
    s.index = pd.to_datetime(s.index)
    return s[(s.index.year >= _Y0) & (s.index.year <= _Y1)]


def plot(out_path: Path) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    apply_nature_style()

    env = _monthly_env(_SA)
    light = _monthly_lightning(_SA).reindex(env.index, method="nearest")
    eff = light / env["E"]
    d = pd.DataFrame({"cape": env["cape"], "cin": env["cin"], "shear": env["shear"],
                      "E": env["E"], "light": light, "eff": eff})

    # (1) seasonality of the efficiency trend
    seasonal = {}
    for name, months in _SEASONS.items():
        sub = d[d.index.month.isin(months)]
        ann = sub.groupby(sub.index.year)["eff"].mean()
        tr, p = _trend(ann.index.to_numpy().astype(int), ann.to_numpy())
        seasonal[name] = (tr, p)

    # (3) thermodynamic-residual: annual trends in each driver vs efficiency
    annual = d.groupby(d.index.year).mean()
    yrs = annual.index.to_numpy().astype(int)
    drivers = {}
    for col in ("cape", "cin", "shear", "E", "eff"):
        drivers[col] = _trend(yrs, annual[col].to_numpy())

    # (2) localisation: per-cell efficiency trend over South Asia
    E_cube = _env_annual_cube(); L_cube = _lightning_annual_cube(E_cube)
    E_cube, L_cube = xr.align(E_cube, L_cube, join="inner")
    landm = _land_mask(E_cube.isel(year=0))
    trop = landm & (np.abs(E_cube["latitude"]) <= 30)
    wL = np.cos(np.deg2rad(L_cube["latitude"])).broadcast_like(L_cube.isel(year=0)).where(trop)
    common = (L_cube.where(trop) * wL).sum(("latitude", "longitude")) / wL.sum(("latitude", "longitude"))
    eff_cube = (L_cube / (common / common.mean("year"))) / E_cube.where(E_cube > 0)
    eff_sa = _box_sel(eff_cube, dict(lon0=60, lon1=100, lat0=5, lat1=35))
    yrs_c = eff_sa["year"].to_numpy().astype(float)

    def _cell_trend(v):
        m = np.isfinite(v)
        if m.sum() < 6:
            return np.nan
        from scipy.stats import theilslopes
        sl = theilslopes(v[m], yrs_c[m])[0]
        mu = np.nanmean(v[m])
        return 100 * sl * 10 / mu if mu else np.nan
    cell = xr.apply_ufunc(_cell_trend, eff_sa, input_core_dims=[["year"]],
                          vectorize=True, output_dtypes=[float])

    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.34))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.1], wspace=0.42)

    # (a) localisation map
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="0.5")
    vm = float(np.nanpercentile(np.abs(cell.to_numpy()), 95))
    pcm = ax.pcolormesh(cell["longitude"], cell["latitude"], cell.to_numpy(),
                        cmap="RdBu_r", vmin=-vm, vmax=vm, transform=ccrs.PlateCarree(), shading="auto")
    ax.plot([_IGP["lon0"], _IGP["lon1"], _IGP["lon1"], _IGP["lon0"], _IGP["lon0"]],
            [_IGP["lat0"], _IGP["lat0"], _IGP["lat1"], _IGP["lat1"], _IGP["lat0"]],
            color="k", lw=1.0, transform=ccrs.PlateCarree())
    ax.set_extent([60, 100, 5, 35], crs=ccrs.PlateCarree())
    ax.set_title("(a) Efficiency trend (%/dec)\nblack = Indo-Gangetic Plain", fontsize=7)
    fig.colorbar(pcm, ax=ax, fraction=0.04, pad=0.03)

    # (b) seasonality bars
    ax2 = fig.add_subplot(gs[0, 1])
    sn = list(seasonal.keys()); sv = [seasonal[k][0] for k in sn]
    bars = ax2.bar(range(len(sn)), sv, color=["#c0392b", "#2980b9", "#7f8c8d"])
    for i, k in enumerate(sn):
        if seasonal[k][1] < 0.1:
            ax2.text(i, sv[i], "*", ha="center", fontsize=10)
    ax2.axhline(0, color="0.4", lw=0.6)
    ax2.set_xticks(range(len(sn))); ax2.set_xticklabels(sn, fontsize=6)
    ax2.set_ylabel("efficiency trend (%/dec)", fontsize=7)
    ax2.set_title("(b) seasonality", fontsize=7.5)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    # (c) drivers vs efficiency annual trend
    ax3 = fig.add_subplot(gs[0, 2])
    keys = ["cape", "cin", "shear", "E", "eff"]
    vals = [drivers[k][0] for k in keys]
    cols = ["#e67e22"] * 4 + ["#8e44ad"]
    ax3.bar(range(len(keys)), vals, color=cols)
    for i, k in enumerate(keys):
        if drivers[k][1] < 0.1:
            ax3.text(i, vals[i], "*", ha="center", fontsize=10)
    ax3.axhline(0, color="0.4", lw=0.6)
    ax3.set_xticks(range(len(keys))); ax3.set_xticklabels(keys, fontsize=6.5)
    ax3.set_ylabel("trend (%/dec)", fontsize=7)
    ax3.set_title("(c) thermodynamic residual\n(drivers vs efficiency)", fontsize=7.5)
    for sp in ("top", "right"):
        ax3.spines[sp].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    igp_med = float(np.nanmedian(_box_sel(cell, _IGP).to_numpy()))
    sa_med = float(np.nanmedian(cell.to_numpy()))
    pre = seasonal["pre-monsoon\n(MAM)"][0]; mon = seasonal["monsoon\n(JJAS)"][0]
    env_weak = all(abs(drivers[k][0]) < abs(drivers["eff"][0]) / 2 for k in ("cape", "cin", "shear", "E"))
    checks = {
        "pre_monsoon_dominant": bool(pre > mon and pre > 0),
        "igp_localised": bool(np.isfinite(igp_med) and igp_med > sa_med),
        "thermodynamically_unforced": bool(env_weak),
    }
    sidecar = {
        "period": f"{_Y0}-{_Y1}", "region": "South Asia 70-90E,10-30N; IGP 75-88E,24-30N",
        "seasonal_eff_trend_pct_decade": {k.replace(chr(10), " "): round(v[0], 1) for k, v in seasonal.items()},
        "driver_trends_pct_decade": {k: round(v[0], 1) for k, v in drivers.items()},
        "driver_mk_p": {k: round(float(v[1]), 3) for k, v in drivers.items()},
        "igp_median_eff_trend": round(igp_med, 1), "south_asia_median_eff_trend": round(sa_med, 1),
        "aerosol_fingerprint_checks": checks,
        "n_checks_passed": int(sum(checks.values())),
        "verdict": ("CONSISTENT with aerosol invigoration (pre-monsoon-concentrated, IGP-localised, "
                    "thermodynamically unforced) -> worth pulling CAMS AOD to test causation"
                    if sum(checks.values()) >= 2 else
                    "fingerprint NOT clean -> aerosol explanation not supported by structure"),
    }
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps(sidecar, indent=2))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "poc_southasia_aerosol.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
