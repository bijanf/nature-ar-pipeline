"""CMIP6 projection anchor: does a model reproduce the OBSERVED thermodynamic/
dynamic split per corridor?

We apply the identical three-term decomposition (thermodynamic / dynamic /
covariance) used on ERA5 to a CMIP6 model, contrasting a recent-historical baseline
(1995-2014) with an end-of-century ssp585 window (2081-2100), for the same eight
corridors. Comparing the model's projected dynamic fraction with the observed one
tests whether the model's circulation response carries the regionally organised
dynamic contribution seen in the observations -- the gap the manuscript argues
matters for projection confidence.

Model: MPI-ESM1-2-HR (r1i1p1f1), Amon monthly hus/ua/va on standard pressure levels,
read from the PIK shared CMIP6 archive. No network.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import xarray as xr

from src import config
from src.analysis.decomposition import _three_term
from src.figures.fig_global_ar_regions import _REGIONS

_G = 9.80665
_C6 = "/p/projects/climate_data_central/CMIP/CMIP6"
_SRC = "MPI-ESM1-2-HR"
_MEM = "r1i1p1f1"
_LEVELS = [85000.0, 50000.0, 25000.0]   # Pa, to match the ERA5 three-level integral
_BASE = (1995, 2014)        # recent-historical baseline
_FUT = (2081, 2100)         # end-of-century ssp585


def _files(var, experiment):
    if experiment == "historical":
        pat = f"{_C6}/CMIP/*/{_SRC}/historical/{_MEM}/Amon/{var}/*/*/*.nc"
    else:
        pat = f"{_C6}/ScenarioMIP/*/{_SRC}/{experiment}/{_MEM}/Amon/{var}/*/*/*.nc"
    return sorted(glob.glob(pat))


def _open(var, experiment, yr0, yr1):
    ds = xr.open_mfdataset(_files(var, experiment), combine="by_coords",
                           chunks={"time": 60}, decode_times=True)
    ds = ds.sel(plev=_LEVELS).sel(time=slice(f"{yr0}-01", f"{yr1}-12"))
    return ds[var]


def _corridor_annual(qa, ua, va, box):
    lon0, lon1, lat0, lat1 = box
    sl = dict(lon=slice(lon0, lon1))
    lat_asc = float(qa.lat[0]) < float(qa.lat[-1])
    latsl = slice(lat0, lat1) if lat_asc else slice(lat1, lat0)
    q = qa.sel(lon=sl["lon"], lat=latsl)
    u = ua.sel(lon=sl["lon"], lat=latsl)
    v = va.sel(lon=sl["lon"], lat=latsl)
    ivt = np.hypot((q * u).integrate("plev") / _G, (q * v).integrate("plev") / _G)
    iwv = q.integrate("plev") / _G
    w = np.cos(np.deg2rad(q["lat"]))
    ivt_m = ivt.weighted(w).mean(("lat", "lon"))
    iwv_m = iwv.weighted(w).mean(("lat", "lon"))
    ivt_y = ivt_m.groupby("time.year").mean("time").to_numpy()
    iwv_y = iwv_m.groupby("time.year").mean("time").to_numpy()
    return ivt_y, iwv_y


def compute(experiment="ssp585"):
    qb = _open("hus", "historical", *_BASE).load()
    ub = _open("ua", "historical", *_BASE).load()
    vb = _open("va", "historical", *_BASE).load()
    qf = _open("hus", experiment, *_FUT).load()
    uf = _open("ua", experiment, *_FUT).load()
    vf = _open("va", experiment, *_FUT).load()

    rows = []
    for name, *box in _REGIONS:
        i0, w0 = _corridor_annual(qb, ub, vb, box)
        i1, w1 = _corridor_annual(qf, uf, vf, box)
        d_tot, d_th, d_dy, d_cv = _three_term(i0, w0, i1, w1)
        base = i0.mean()
        rows.append(dict(
            region=name, base_ivt=round(float(base), 1),
            pct_change=round(100 * d_tot / base, 1),
            d_total=round(float(d_tot), 2), d_thermo=round(float(d_th), 2),
            d_dyn=round(float(d_dy), 2), d_cov=round(float(d_cv), 2),
            dyn_frac=round(float(d_dy / d_tot), 3) if d_tot else None,
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="ssp585")
    args = ap.parse_args()
    rows = compute(args.experiment)
    out = config.CACHE_DIR / "cmip6_decomp.json"
    out.write_text(json.dumps(dict(model=_SRC, experiment=args.experiment,
                                   base=_BASE, future=_FUT, corridors=rows), indent=2))
    print(f"wrote {out}  ({_SRC} {args.experiment}, {_BASE[0]}-{_BASE[1]} -> {_FUT[0]}-{_FUT[1]})")
    print(f"{'corridor':20s} {'tot':>6s} {'thermo':>7s} {'dyn':>6s} {'cov':>6s} {'dyn%':>6s}")
    for r in sorted(rows, key=lambda r: -r["d_dyn"]):
        df = 100 * r["dyn_frac"] if r["dyn_frac"] is not None else float("nan")
        print(f"{r['region']:20s} {r['d_total']:6.2f} {r['d_thermo']:7.2f} "
              f"{r['d_dyn']:6.2f} {r['d_cov']:6.2f} {df:6.0f}")


if __name__ == "__main__":
    main()
