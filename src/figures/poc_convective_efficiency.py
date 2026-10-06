"""PROOF OF CONCEPT — severe-convective-storm environment, CAPE-vs-shear split.

De-risks the convective-efficiency direction by reusing the AR pipeline's
machinery on a new variable, with NO new heavy download:

  * thermodynamic half  : ERA5 monthly CAPE (era5_convective_monthly/)
  * dynamic half        : deep-layer bulk shear S = |V(500) - V(850)| derived
                          from the existing monthly winds (era5_global_monthly/)
  * severe-environment proxy : WMAXSHEAR  E = sqrt(2 CAPE) * S   (Taszarek/Brooks);
                          sqrt(2 CAPE) is the thermodynamic max-updraft speed (m/s),
                          S the deep-layer shear (m/s), so E (m^2 s^-2) factorises
                          cleanly into a thermodynamic and a dynamic term.

Outputs over a CONUS severe-weather box (Great Plains -> Southeast):
  (a) continuous 1940-2024 annual-E trajectory + Theil-Sen trend + emergence
      (reuses the fig_timeseries trend math), and
  (b) the recent-minus-presat decomposition  dE = dE_thermo(CAPE) + dE_dyn(shear)
      following the SAME multiplicative split as fig_drivers:
          dE_thermo = E0 * (W1/W0 - 1),   dE_dyn = dE_total - dE_thermo
      with W = sqrt(2 CAPE).

CAVEAT (load-bearing): monthly-mean CAPE understates intermittent convective
potential; this is a first-look trend + infrastructure-transfer test, not the
production estimate (which needs sub-daily CAPE). The point is to confirm the
signal exists and that ~all of the pipeline transfers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.figures._style import apply_nature_style, COL_DOUBLE_IN

_CONV = Path("/p/projects/poem/fallah/nature_ar_data/era5_convective_monthly")
_WIND = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")
# (conv-file stem, wind-file stem) per window, in time order -> continuous record.
_PAIRS = [
    ("presat_conv_monthly_global.nc", "presat_monthly_global.nc"),
    ("gap1_1960_1979_conv_monthly_global.nc", "gap1_1960_1979_monthly_global.nc"),
    ("modern_conv_monthly_global.nc", "modern_monthly_global.nc"),
    ("gap2_2000_2014_conv_monthly_global.nc", "gap2_2000_2014_monthly_global.nc"),
    ("recent_conv_monthly_global.nc", "recent_monthly_global.nc"),
]
_WINDOWS = [(1940, 1959), (1980, 1999), (2015, 2024)]
# CONUS severe-weather box (Great Plains + Midwest + Southeast). 0..360 lon.
_BOX = dict(lon_min=255.0, lon_max=285.0, lat_min=30.0, lat_max=48.0)


def _tname(ds):
    return "valid_time" if "valid_time" in ds.dims else "time"


def _lev(da):
    return "pressure_level" if "pressure_level" in da.dims else "level"


def _box(ds):
    return ds.sel(longitude=slice(_BOX["lon_min"], _BOX["lon_max"]),
                  latitude=slice(_BOX["lat_max"], _BOX["lat_min"]))  # lat descending


def _shear(windpath: Path) -> xr.DataArray:
    """Monthly deep-layer bulk shear |V(500) - V(850)| (m/s) over the box."""
    ds = _box(xr.open_dataset(windpath))
    lev = _lev(ds["u"])
    u5 = ds["u"].sel({lev: 500}); u8 = ds["u"].sel({lev: 850})
    v5 = ds["v"].sel({lev: 500}); v8 = ds["v"].sel({lev: 850})
    s = np.hypot(u5 - u8, v5 - v8).rename("shear")
    t = _tname(ds)
    return s.rename({t: "time"}).assign_coords(time=ds[t].values)


def _cape(convpath: Path) -> xr.DataArray:
    ds = _box(xr.open_dataset(convpath))
    cape = ds["cape"].clip(min=0.0)
    t = _tname(ds)
    return cape.rename({t: "time"}).assign_coords(time=ds[t].values).rename("cape")


def _wmaxshear(cape: xr.DataArray, shear: xr.DataArray) -> xr.DataArray:
    """E = sqrt(2 CAPE) * S, the WMAXSHEAR severe-environment proxy (m^2 s^-2)."""
    return (np.sqrt(2.0 * cape) * shear).rename("E")


def _wmean(da: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(da["latitude"]))
    return da.weighted(w).mean(("latitude", "longitude"))


def _active_mean(da: xr.DataArray, e_ref: xr.DataArray) -> float:
    """Mean weighted to the active severe-weather region (high mean E), mirroring
    fig_drivers._corridor_mean, so the headline reflects where storms live."""
    w = e_ref.where(e_ref > float(e_ref.quantile(0.6)), 0.0)
    return float((da * w).sum() / w.sum())


def _annual_series() -> pd.DataFrame:
    """Continuous annual box-mean CAPE, shear, and E across all windows."""
    capes, shears, Es = [], [], []
    for cf, wf in _PAIRS:
        cp, wp = _CONV / cf, _WIND / wf
        if not (cp.exists() and wp.exists()):
            continue
        cape = _cape(cp)
        shear = _shear(wp)
        # align on common monthly timestamps
        shear = shear.reindex(time=cape["time"], method="nearest", tolerance=np.timedelta64(20, "D"))
        E = _wmaxshear(cape, shear)
        capes.append(_wmean(cape)); shears.append(_wmean(shear)); Es.append(_wmean(E))
    cape = xr.concat(capes, dim="time").sortby("time")
    shear = xr.concat(shears, dim="time").sortby("time")
    E = xr.concat(Es, dim="time").sortby("time")
    df = pd.DataFrame({
        "cape": cape.groupby("time.year").mean("time").to_pandas(),
        "shear": shear.groupby("time.year").mean("time").to_pandas(),
        "E": E.groupby("time.year").mean("time").to_pandas(),
    })
    return df


def _theil_sen(x, y):
    from scipy.stats import theilslopes, kendalltau
    slope, intercept, lo, hi = theilslopes(y, x)
    tau, p = kendalltau(x, y)
    return slope, intercept, p


def _emergence(years, y, base_mask, intercept, slope):
    base = y[base_mask]
    thr = base.mean() + base.std()
    trend = intercept + slope * years
    above = np.where(trend > thr)[0]
    return (int(years[above[0]]) if above.size else None), thr


def _window_field(stem_conv: str, stem_wind: str) -> tuple[xr.DataArray, xr.DataArray]:
    """Time-mean CAPE-updraft W=sqrt(2 CAPE) and shear S over a window, per pixel."""
    cape = _cape(_CONV / stem_conv).mean("time")
    shear = _shear(_WIND / stem_wind).mean("time")
    shear = shear.reindex_like(cape, method="nearest")
    W = np.sqrt(2.0 * cape)
    return W, shear


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    df = _annual_series()
    years = df.index.to_numpy().astype(int)
    E = df["E"].to_numpy()
    slope, intercept, p = _theil_sen(years, E)
    per_decade = slope * 10.0
    base_mask = years <= 1959
    emerge, thr = _emergence(years, E, base_mask, intercept, slope)

    # --- decomposition: recent (2015-24) minus pre-sat (1940-59), per pixel -----
    W0, S0 = _window_field(_PAIRS[0][0], _PAIRS[0][1])
    W1, S1 = _window_field(_PAIRS[-1][0], _PAIRS[-1][1])
    E0 = (W0 * S0).rename("E0")
    E1 = (W1 * S1).rename("E1")
    d_total = (E1 - E0)
    d_thermo = E0 * (W1 / W0 - 1.0)        # CAPE change at fixed shear
    d_dyn = d_total - d_thermo             # residual = shear/circulation change
    am_total = _active_mean(d_total, E0)
    am_thermo = _active_mean(d_thermo, E0)
    am_dyn = _active_mean(d_dyn, E0)
    e0 = _active_mean(E0, E0)
    pct = 100.0 * am_total / e0 if e0 else float("nan")
    thermo_pct = 100.0 * am_thermo / am_total if am_total else float("nan")

    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.40))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.0, 1.4], wspace=0.42)

    # (a) continuous trajectory
    ax = fig.add_subplot(gs[0, 0])
    for (a, b) in _WINDOWS:
        ax.axvspan(a, b, color="0.88", zorder=0)
    ax.plot(years, E, color="#2c3e50", lw=0.8, marker="o", ms=2, label="annual E")
    sm = pd.Series(E, index=years).rolling(7, center=True, min_periods=4).mean()
    ax.plot(years, sm.to_numpy(), color="#c0392b", lw=1.6, label="7-yr mean")
    ax.plot(years, intercept + slope * years, color="#2980b9", lw=1.3, ls="--",
            label=f"Theil–Sen {per_decade:+.1f}/dec (p={p:.1e})")
    ax.axhline(thr, color="0.5", lw=0.6, ls=":")
    if emerge:
        ax.axvline(emerge, color="#16a085", lw=1.0)
        ax.annotate(f"emergence ≈ {emerge}", xy=(emerge, thr), fontsize=6,
                    color="#16a085", xytext=(emerge + 1, E.min() + 0.55 * (E.max() - E.min())))
    ax.set_xlabel("year", fontsize=8)
    ax.set_ylabel("WMAXSHEAR E (m$^2$ s$^{-2}$)", fontsize=8)
    ax.set_xlim(years.min(), years.max())
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    ax.set_title("(a) CONUS severe-environment trajectory, 1940–2024", fontsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # (b) thermo/dynamic split bars
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar([0, 1], [am_thermo, am_dyn], color=["#c0392b", "#2980b9"], width=0.7)
    ax2.axhline(0, color="0.4", lw=0.6)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["CAPE\n(thermo)", "shear\n(dyn)"], fontsize=6.5)
    ax2.set_ylabel("ΔE contribution", fontsize=7)
    ax2.set_title(f"(b) split\n{thermo_pct:.0f}% / {100-thermo_pct:.0f}%", fontsize=7.5)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    # (c) ΔE map over the CONUS box
    ax3 = fig.add_subplot(gs[0, 2])
    dmax = float(np.nanpercentile(np.abs(d_total.to_numpy()), 98))
    lons = d_total["longitude"].to_numpy(); lons_m = np.where(lons > 180, lons - 360, lons)
    lats = d_total["latitude"].to_numpy()
    pcm = ax3.pcolormesh(lons_m, lats, d_total.to_numpy(), cmap="RdBu_r",
                         vmin=-dmax, vmax=dmax, shading="auto")
    ax3.set_title(f"(c) ΔE recent−presat\nactive {am_total:+.0f} ({pct:+.0f}%)", fontsize=7.5)
    ax3.set_xlabel("lon", fontsize=7); ax3.set_ylabel("lat", fontsize=7)
    fig.colorbar(pcm, ax=ax3, fraction=0.046, pad=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    sidecar = {
        "proxy": "WMAXSHEAR sqrt(2*CAPE)*|V500-V850|",
        "box": _BOX,
        "trend_per_decade": round(per_decade, 2),
        "mk_p": float(f"{p:.2e}"),
        "emergence_year": emerge,
        "delta_E_active": round(am_total, 1),
        "delta_E_pct": round(pct, 1),
        "thermo_pct": round(thermo_pct),
        "dyn_pct": round(100 - thermo_pct),
        "cape_presat_mean": round(float(df["cape"][df.index <= 1959].mean()), 1),
        "cape_recent_mean": round(float(df["cape"][df.index >= 2015].mean()), 1),
        "shear_presat_mean": round(float(df["shear"][df.index <= 1959].mean()), 2),
        "shear_recent_mean": round(float(df["shear"][df.index >= 2015].mean()), 2),
        "n_years": int(len(years)),
    }
    side = out_path.with_suffix(".json")
    side.write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps(sidecar, indent=2))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "poc_convective_efficiency.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
