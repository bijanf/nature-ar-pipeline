"""Preview of the level-resolved decomposition on the existing three-level ERA5 data.

The rejected manuscript used ERA5 monthly means on 850, 500 and 250 hPa only. This
script (i) reproduces its V-hat numbers as a check of the data handling, and
(ii) applies the new moisture/wind/covariance/transient split to the same data,
for the 30-year headline periods and the two alternative period pairs, and
(iii) compares with the CMIP6 members extracted so far on the same three levels.
The final analysis uses the 23-level data and the hourly flux integrals.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import xarray as xr

from src.analysis import fullcolumn_decomp as fd
from src.figures.fig_global_ar_regions import _REGIONS

OLD = fd.DATA / "era5_global_monthly"
FILES = ["presat_monthly_global.nc", "gap1_1960_1979_monthly_global.nc", "modern_monthly_global.nc",
         "gap2_2000_2014_monthly_global.nc", "recent_monthly_global.nc"]
LEVELS_PA = np.array([25000.0, 50000.0, 85000.0])
# Supplementary Table S1 of the rejected manuscript: dIVT(%), thermo, dynamic, covariance
TABLE_S1 = {
    "Western Europe": (17.8, 5.03, 6.61, 0.49), "Amazon outflow": (10.6, 5.77, 5.76, 0.30),
    "East Asia": (12.6, 7.11, 5.14, 0.37), "SE South America": (5.2, -1.54, 4.86, -0.12),
    "South Africa": (5.4, 1.30, 1.21, 0.03), "SE US / Gulf": (11.0, 7.18, 1.04, 0.10),
    "US West Coast": (6.4, 3.92, 0.30, 0.02), "SE Australia / NZ": (3.9, 5.17, -2.59, -0.22),
}


def _drop(ds):
    return ds.drop_vars([v for v in ("expver", "number") if v in ds.variables])


def load_old() -> xr.Dataset:
    ds = xr.concat([_drop(xr.open_dataset(OLD / f)[["q", "u", "v", "t"]]) for f in FILES], "valid_time")
    ds = ds.sortby("valid_time")
    ds = ds.isel(valid_time=~ds.get_index("valid_time").duplicated())
    assert ds.sizes["valid_time"] == 1020, ds.sizes
    return ds.sortby("pressure_level")


def ps_clim() -> xr.DataArray:
    files = sorted(glob.glob(str(fd.ERA5_DIR / "sl" / "era5_sl_monthly_*.nc")))
    return xr.open_mfdataset(files, combine="by_coords", preprocess=_drop)["sp"].mean("valid_time").load()


def old_weights(shape):
    """Trapezoid over 250-850 hPa, as xarray.integrate did in the rejected manuscript."""
    w = np.array([12500.0, 30000.0, 17500.0]) / fd.G
    return np.broadcast_to(w[:, None, None], (3,) + shape).copy()


def era5_corridor(ds, name, box, mode, ps):
    lon0, lon1, lat0, lat1 = box
    sub = ds.sel(longitude=slice(lon0, min(lon1, 359.0)), latitude=slice(lat1, lat0))
    lat, lon = sub.latitude.values, sub.longitude.values
    p = sub.pressure_level.values.astype(float) * 100.0
    if mode == "old":
        wint = old_weights((len(lat), len(lon)))
    else:
        wint = fd.layer_weights(p, ps.sel(latitude=lat, longitude=lon).values)
    arr = lambda v: fd._yr_month(sub[v], fd.YEARS).astype("float64")  # noqa: E731
    return fd.Corridor(name, fd.YEARS, p, lat, arr("q"), arr("u"), arr("v"), wint, t=arr("t"))


def cmip_corridor(path, name, ps):
    k = fd.corridor_key(name)
    ds = xr.open_dataset(path)
    q, u, v = (fd.merge_dup_plev(ds[f"{x}_{k}"]).sel(plev=LEVELS_PA) for x in ("hus", "ua", "va"))
    lat, lon = q[f"lat_{k}"].values, q[f"lon_{k}"].values
    wint = fd.layer_weights(LEVELS_PA, ps.interp(latitude=lat, longitude=lon, method="nearest", kwargs={"fill_value": "extrapolate"}).values)
    arrs = [np.nan_to_num(fd._yr_month(x, fd.YEARS).astype("float64")) for x in (q, u, v)]
    return fd.Corridor(name, fd.YEARS, LEVELS_PA, lat, *arrs, wint)


def old_style_vhat(c, w0, w1):
    """Rejected-manuscript V-hat split: annual means of monthly |F| and W."""
    mag = np.hypot(c.fmon[:, :, 0], c.fmon[:, :, 1])
    I = np.einsum("tmyx,yx->tm", mag, c.aw).mean(1)
    W = np.einsum("tmyx,yx->tm", c.iwv, c.aw).mean(1)
    i0 = (fd.YEARS >= w0[0]) & (fd.YEARS <= w0[1])
    i1 = (fd.YEARS >= w1[0]) & (fd.YEARS <= w1[1])
    I0, I1, W0, W1 = I[i0].mean(), I[i1].mean(), W[i0].mean(), W[i1].mean()
    V0, V1 = I0 / W0, I1 / W1
    return (round(100 * (I1 - I0) / I0, 1), round(V0 * (W1 - W0), 2), round(W0 * (V1 - V0), 2),
            round((W1 - W0) * (V1 - V0), 2))


def main_sensitivity():
    """Period-sensitivity matrix and seasonal split on the three-level data."""
    ds = load_old()
    ps = ps_clim()
    cors = [era5_corridor(ds, n, b, "sfc", ps) for n, *b in _REGIONS]
    out = {"matrix": fd.sensitivity_matrix(cors, n_boot=300)}
    for s, months in fd.SEASONS.items():
        out[s] = fd.window_summary(cors, *fd.HEADLINE, n_boot=500, block=3, months=months)
    fd.OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = fd.OUT_DIR / "preview_3level_sensitivity.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


def main():
    import sys
    if "--part" in sys.argv and sys.argv[sys.argv.index("--part") + 1] == "sensitivity":
        return main_sensitivity()
    ds = load_old()
    ps = ps_clim()
    out = {"validation_vs_tableS1": {}}
    old = [era5_corridor(ds, n, b, "old", ps) for n, *b in _REGIONS]
    for c in old:
        out["validation_vs_tableS1"][c.name] = {"recomputed": old_style_vhat(c, *fd.ORIGINAL),
                                                "table_S1": TABLE_S1[c.name]}
    cors = [era5_corridor(ds, n, b, "sfc", ps) for n, *b in _REGIONS]
    pairs = {"headline": fd.HEADLINE, "original": fd.ORIGINAL, "satellite": fd.SATELLITE}
    for lab, (w0, w1) in pairs.items():
        out[lab] = fd.window_summary(cors, w0, w1, n_boot=1000, block=3)
        out[lab + "_block5"] = fd.window_summary(cors, w0, w1, n_boot=1000, block=5)
    ys = {c.name: fd.yearly_contributions(c) for c in cors}
    out["trends_full"] = {k: fd.trend_table(v, fd.YEARS) for k, v in ys.items()}
    out["trends_sat"] = {k: fd.trend_table(v, fd.YEARS, 1979, 2024) for k, v in ys.items()}
    # CMIP6 members available so far, same three levels and weights
    det = {}
    for model_dir in sorted((fd.DATA / "cmip6_corridors").glob("*")):
        files = sorted(model_dir.glob("*.nc"))
        if len(files) < 5:
            continue
        vals = {n: {lab: [] for lab in pairs} for n, *_ in _REGIONS}
        for f in files:
            for n, *_ in _REGIONS:
                c = cmip_corridor(f, n, ps)
                for lab, (w0, w1) in pairs.items():
                    i0 = np.where((fd.YEARS >= w0[0]) & (fd.YEARS <= w0[1]))[0]
                    i1 = np.where((fd.YEARS >= w1[0]) & (fd.YEARS <= w1[1]))[0]
                    r = fd.reduce_months(fd.window_terms(c, i0, i1, cc_split=False))
                    vals[n][lab].append((r["moist"], r["wind"], r["total"]))
        det[model_dir.name] = {"n": len(files)}
        for n in vals:
            det[model_dir.name][n] = {}
            for lab in pairs:
                x = np.array(vals[n][lab])
                obs = next(r for r in out[lab] if r["corridor"] == n)
                row = {}
                for j, term in enumerate(("moist", "wind", "total")):
                    mu, sd, o = x[:, j].mean(), x[:, j].std(ddof=1), obs[term]
                    row[term] = {"obs": round(o, 2), "mean": round(float(mu), 2), "sd": round(float(sd), 2),
                                 "z": round(float((o - mu) / sd), 2),
                                 "range": [round(float(x[:, j].min()), 2), round(float(x[:, j].max()), 2)]}
                det[model_dir.name][n][lab] = row
    out["cmip6_preview"] = det
    fd.OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = fd.OUT_DIR / "preview_3level.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
