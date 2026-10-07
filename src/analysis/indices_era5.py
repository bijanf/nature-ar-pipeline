"""Standard climate indices computed from ERA5 over 1940-2024.

All monthly anomalies are taken relative to the 1940-2024 calendar-month
climatology.

* Relative Nino3.4: SST anomaly over 5S-5N, 170W-120W minus the tropical-mean
  (20S-20N) SST anomaly, which removes the warming common to the tropics.
* Pacific Decadal Oscillation: leading EOF of North Pacific (20N-70N,
  110E-100W) SST anomalies after removal of the 60S-60N mean SST anomaly.
* Atlantic Multidecadal Variability: North Atlantic (0-60N, 80W-0) SST anomaly
  minus the 60S-60N mean SST anomaly.
* North Atlantic Oscillation: leading EOF of sea-level pressure anomalies over
  20N-80N, 90W-40E, signed so that low pressure near Iceland is positive.
* Southern Annular Mode: difference of normalised zonal-mean sea-level pressure
  at 40S and 65S.
* Pacific-North American pattern: four-point combination of standardised 500 hPa
  height anomalies at (20N,160W), (45N,165W), (55N,115W) and (30N,85W).

Output: results_esd/indices_era5.nc (monthly) and a validation table against
the published series.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DATA = Path("/p/projects/poem/fallah/nature_ar_data")
OUT = DATA / "results_esd"
PUB = DATA / "indices_published"
G = 9.80665


def _anom(da):
    t = "valid_time"
    return da.groupby(f"{t}.month") - da.groupby(f"{t}.month").mean(t)


def _wmean(da, lat0, lat1, lon0=None, lon1=None):
    s = da.sel(latitude=slice(lat1, lat0))
    if lon0 is not None:
        s = s.sel(longitude=slice(lon0, lon1))
    w = np.cos(np.deg2rad(s.latitude))
    return s.weighted(w).mean(("latitude", "longitude"))


def _eof1(da):
    w = np.sqrt(np.cos(np.deg2rad(da.latitude)))
    x = (da * w).stack(s=("latitude", "longitude")).dropna("s", how="any")
    a = x.values - x.values.mean(0)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    pc = u[:, 0] * s[0]
    frac = s[0] ** 2 / np.sum(s ** 2)
    pat = xr.DataArray(vt[0], coords={"s": x.s}, dims="s").unstack("s")
    return xr.DataArray(pc / pc.std(), coords={"valid_time": da.valid_time}, dims="valid_time"), pat, frac


def compute():
    sl = xr.open_mfdataset(sorted(glob.glob(str(DATA / "era5_fullcolumn/sl/era5_sl_monthly_*.nc"))),
                           combine="by_coords",
                           preprocess=lambda d: d.drop_vars([v for v in ("expver", "number") if v in d.variables]))
    sst = _anom(sl["sst"].load())
    msl = _anom(sl["msl"].load())
    sst_trop = _wmean(sst, -20, 20)
    sst_glob = _wmean(sst, -60, 60)
    out = {}
    out["rnino34"] = _wmean(sst, -5, 5, 190, 240) - sst_trop
    np_ = sst.sel(latitude=slice(70, 20), longitude=slice(110, 260)) - sst_glob
    out["pdo"], pat_pdo, f_pdo = _eof1(np_)
    # sign convention: positive PDO has cool central North Pacific
    if float(pat_pdo.sortby("latitude").sortby("longitude").sel(latitude=40, longitude=180, method="nearest")) > 0:
        out["pdo"] = -out["pdo"]
    out["amv"] = _wmean(sst, 0, 60, 280, 360) - sst_glob
    # Atlantic sector on a monotonic longitude axis from -90 to 40
    msl_w = msl.assign_coords(longitude=((msl.longitude + 180) % 360) - 180).sortby("longitude")
    na = msl_w.sel(longitude=slice(-90, 40), latitude=slice(80, 20))
    nao, pat, f_nao = _eof1(na)
    if float(pat.sortby("latitude").sortby("longitude").sel(latitude=65, longitude=-20, method="nearest")) > 0:
        nao = -nao
    out["nao"] = nao
    zm = msl.mean("longitude")
    norm = lambda x: (x.groupby("valid_time.month") / x.groupby("valid_time.month").std())  # noqa: E731
    out["sam"] = norm(zm.sel(latitude=-40)) - norm(zm.sel(latitude=-65))
    zfiles = sorted(glob.glob(str(DATA / "era5_global_monthly/*_monthly_global.nc")))
    z = xr.open_mfdataset(zfiles, combine="by_coords")["z"].sel(pressure_level=500).load() / G
    z = z.isel(valid_time=~z.get_index("valid_time").duplicated()).sortby("valid_time")
    za = _anom(z)
    zs = za.groupby("valid_time.month") / za.groupby("valid_time.month").std()
    pt = lambda la, lo: zs.sel(latitude=la, longitude=lo, method="nearest").drop_vars(["latitude", "longitude"])  # noqa: E731
    out["pna"] = 0.25 * (pt(20, 200) - pt(45, 195) + pt(55, 245) - pt(30, 275))
    ds = xr.Dataset({k: v.drop_vars([c for c in v.coords if c not in ("valid_time",)]) for k, v in out.items()})
    ds = ds.sel(valid_time=slice("1940-01-01", "2024-12-31"))
    ds.attrs.update(pdo_eof1_variance=float(f_pdo), nao_eof1_variance=float(f_nao))
    OUT.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT / "indices_era5.nc")
    return ds


# ----------------------------------------------------------------------------- validation
def _read_rows(path, first_year_col=True, skip=0, missing=(-99.99, -99.9, -9.9, -999)):
    rec = {}
    for line in Path(path).read_text().splitlines()[skip:]:
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            y = int(parts[0]); vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        for m, v in enumerate(vals, 1):
            if v not in missing and abs(v) < 90:
                rec[pd.Timestamp(y, m, 1)] = v
    return pd.Series(rec).sort_index()


def _read_cpc(path):
    rec = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) == 3:
            rec[pd.Timestamp(int(p[0]), int(p[1]), 1)] = float(p[2])
    return pd.Series(rec)


def validate(ds):
    pub = {
        "rnino34": _read_rows(PUB / "nina34.anom.data"),
        "nao": _read_cpc(PUB / "norm.nao.monthly.b5001.current.ascii"),
        "pna": _read_cpc(PUB / "norm.pna.monthly.b5001.current.ascii"),
        "pdo": _read_rows(PUB / "ersst.v5.pdo.dat"),
        "amv": _read_rows(PUB / "amon.us.long.data"),
        "sam": _read_rows(PUB / "newsam.1957.2007.txt"),
    }
    res = {}
    for k, s in pub.items():
        e = ds[k].to_series()
        e.index = pd.to_datetime(e.index).to_period("M").to_timestamp()
        j = pd.concat([e, s], axis=1, join="inner").dropna()
        ann = j.groupby(j.index.year).mean()
        res[k] = {"overlap": f"{j.index[0].year}-{j.index[-1].year}",
                  "r_monthly": round(float(j.corr().iloc[0, 1]), 3),
                  "r_annual": round(float(ann.corr().iloc[0, 1]), 3)}
    (OUT / "indices_validation.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    d = compute()
    print(validate(d))
