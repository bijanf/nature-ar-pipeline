"""Extract monthly humidity and wind over the eight corridor boxes from CMIP6.

For every model and member, the historical run (1940-2014) is joined with its
scenario continuation (2015-2024, ssp585, or ssp245 where ssp585 is missing for
that member) and specific humidity, zonal and meridional wind are saved on the
standard pressure levels from 1000 to 200 hPa over each corridor box. The
global-mean near-surface temperature of the same member is saved alongside, so
the warming of each member over the analysis windows is known.

Two sources are supported:
* ``pangeo``: the Pangeo CMIP6 cloud archive (large single-model ensembles);
* ``pik``: the institutional CMIP6 archive at PIK (one member per model).

Output: ROOT/<model>/<member>.nc with variables hus_<k>, ua_<k>, va_<k>
(dimensions time, plev, lat_<k>, lon_<k>) for corridor key k, and tas_gm(time).

Usage:
  python -m src.data.extract_cmip6_corridors --source pangeo --model CanESM5 [--max-members N]
  python -m src.data.extract_cmip6_corridors --source pik --model MIROC6
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

from src.figures.fig_global_ar_regions import _REGIONS

ROOT = Path("/p/projects/poem/fallah/nature_ar_data/cmip6_corridors")
CATALOG = Path("/p/projects/poem/fallah/nature_ar_data/pangeo-cmip6.csv")
PIK = "/p/projects/climate_data_central/CMIP/CMIP6"
VARS = ("hus", "ua", "va")
PMIN = 20000.0  # Pa
T0, T1 = "1940-01-01", "2024-12-31"


def corridor_key(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _subset_box(da: xr.DataArray, box) -> xr.DataArray:
    lon0, lon1, lat0, lat1 = box
    lon = da["lon"] % 360
    da = da.assign_coords(lon=lon).sortby("lon").sortby("lat")
    return da.sel(lon=slice(lon0, lon1), lat=slice(lat0, lat1))


def _tidy(ds: xr.Dataset) -> xr.Dataset:
    ren = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.dims}
    ds = ds.rename(ren)
    drop = [v for v in ds.coords if v not in ("time", "plev", "lat", "lon")]
    return ds.drop_vars(drop + [v for v in ds.data_vars if v.endswith("_bnds")], errors="ignore")


def _monthly_index(da: xr.DataArray) -> xr.DataArray:
    t = da["time"].to_index()
    stamps = pd.to_datetime([f"{x.year:04d}-{x.month:02d}-01" for x in t])
    return da.assign_coords(time=stamps)


# ----------------------------------------------------------------------------- pangeo
def _pangeo_table() -> pd.DataFrame:
    if not CATALOG.exists():
        pd.read_csv("https://storage.googleapis.com/cmip6/pangeo-cmip6.csv").to_csv(CATALOG, index=False)
    df = pd.read_csv(CATALOG)
    return df[(df.table_id == "Amon") & df.variable_id.isin(VARS + ("tas",))]


def _pangeo_members(df: pd.DataFrame, model: str) -> list[tuple[str, str]]:
    sub = df[df.source_id == model]

    def have(exp):
        s = sub[sub.experiment_id == exp]
        sets = [set(s[s.variable_id == v].member_id) for v in VARS + ("tas",)]
        return set.intersection(*sets) if sets else set()

    hist, s585, s245 = have("historical"), have("ssp585"), have("ssp245")
    out = [(m, "ssp585") for m in sorted(hist & s585)]
    out += [(m, "ssp245") for m in sorted((hist & s245) - s585)]
    return out


def _pangeo_open(df, model, exp, member, var):
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    s = df[(df.source_id == model) & (df.experiment_id == exp) &
           (df.member_id == member) & (df.variable_id == var)]
    s = s.sort_values("version")
    z = s.zstore.iloc[-1]
    return _tidy(xr.open_zarr(fs.get_mapper(z), consolidated=True))[var]


# ----------------------------------------------------------------------------- pik
def _pik_open(model, exp, member, var):
    act = "CMIP" if exp == "historical" else "ScenarioMIP"
    files = sorted(glob.glob(f"{PIK}/{act}/*/{model}/{exp}/{member}/Amon/{var}/*/*/*.nc"))
    if not files:
        raise FileNotFoundError(f"{model} {exp} {member} {var}")
    vers = sorted({f.split("/")[-2] for f in files})
    files = [f for f in files if f.split("/")[-2] == vers[-1]]
    ds = xr.open_mfdataset(files, combine="by_coords", use_cftime=True, chunks={"time": 120})
    return _tidy(ds)[var]


def _pik_members(model: str) -> list[tuple[str, str]]:
    out = []
    for d in sorted(glob.glob(f"{PIK}/CMIP/*/{model}/historical/*/Amon/hus")):
        m = d.split("/")[-3]
        for exp in ("ssp585", "ssp245"):
            if glob.glob(f"{PIK}/ScenarioMIP/*/{model}/{exp}/{m}/Amon/hus"):
                out.append((m, exp))
                break
    return out[:1]


# ----------------------------------------------------------------------------- core
def extract_member(opener, model: str, member: str, scen: str, t0=T0, t1=T1, root=ROOT) -> str:
    out = root / model / f"{member}.nc"
    if out.exists() and out.stat().st_size > 4096:
        return f"skip {model} {member}"
    out.parent.mkdir(parents=True, exist_ok=True)
    pieces = {}
    for var in VARS + ("tas",):
        parts = []
        for exp in ("historical", scen):
            da = _monthly_index(opener(model, exp, member, var))
            if "plev" in da.coords:  # historical and scenario files may differ by float noise
                da = da.assign_coords(plev=np.round(da["plev"].values.astype("float64")))
            parts.append(da)
        da = xr.concat(parts, "time").sortby("time")
        da = da.isel(time=~da.get_index("time").duplicated()).sel(time=slice(t0, t1))
        pieces[var] = da
    lazy = {}
    for name, *box in _REGIONS:
        k = corridor_key(name)
        for var in VARS:
            da = pieces[var]
            da = da.sel(plev=da.plev[da.plev >= PMIN - 1])
            sub = _subset_box(da, box).astype("float32")
            lazy[f"{var}_{k}"] = sub.rename(lat=f"lat_{k}", lon=f"lon_{k}")
    tas = pieces["tas"]
    w = np.cos(np.deg2rad(tas["lat"]))
    lazy["tas_gm"] = tas.weighted(w).mean(("lat", "lon")).astype("float32")
    names = list(lazy)
    vals = dask.compute(*[lazy[n] for n in names])
    ds = xr.Dataset({n: v for n, v in zip(names, vals)})
    ds.attrs.update(model=model, member=member, scenario_after_2014=scen)
    tmp = out.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp)
    os.replace(tmp, out)
    nt = ds.sizes["time"]
    return f"wrote {model} {member} ({scen}) nt={nt} {out.stat().st_size / 1e6:.0f} MB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pangeo", "pik"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-members", type=int, default=0)
    ap.add_argument("--projection", action="store_true",
                    help="1995-2100 with ssp585 into cmip6_corridors_proj (PIK source)")
    a = ap.parse_args()
    dask.config.set(scheduler="threads", num_workers=4)
    if a.source == "pangeo":
        df = _pangeo_table()
        members = _pangeo_members(df, a.model)
        opener = lambda mo, ex, me, v: _pangeo_open(df, mo, ex, me, v)  # noqa: E731
    else:
        members = _pik_members(a.model)
        opener = _pik_open
    if a.max_members:
        members = members[: a.max_members]
    print(f"{a.model}: {len(members)} members", flush=True)
    for m, scen in members:
        try:
            if a.projection:
                if scen != "ssp585":
                    continue
                print(extract_member(opener, a.model, m, scen, "1995-01-01", "2100-12-31",
                                     ROOT.with_name("cmip6_corridors_proj")), flush=True)
            else:
                print(extract_member(opener, a.model, m, scen), flush=True)
        except Exception as e:  # keep going; rerun fills gaps
            print(f"FAILED {a.model} {m}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
