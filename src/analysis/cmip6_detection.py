"""Apply the level-resolved decomposition to every CMIP6 member and compare with ERA5.

For each member and corridor the moisture-change, wind-change, covariance and
transient terms are computed for the same windows as the observations (monthly
products on the CMIP6 standard levels, so the transient term holds month-to-month
departures only; ERA5 is recomputed the same way for a like-for-like comparison).

Within each single-model large ensemble, the ensemble mean estimates the forced
response and the spread across members estimates internal variability. The
observed term is compared with that distribution as a standardised departure
(obs - ensemble mean) / ensemble standard deviation and as the fraction of members
whose term is at least as large as the observed one.

Output: results_esd/cmip6_members.json and results_esd/cmip6_detection.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from src.analysis import fullcolumn_decomp as fd
from src.figures.fig_global_ar_regions import _REGIONS

ROOT = Path("/p/projects/poem/fallah/nature_ar_data/cmip6_corridors")
LARGE = ("CanESM5", "MIROC6", "ACCESS-ESM1-5", "EC-Earth3")
WINDOWS = {"headline": fd.HEADLINE, "original": fd.ORIGINAL, "satellite": fd.SATELLITE}


def _ps_clim():
    f = fd.ERA5_DIR / "era5_sl_monthly_1940_2024.nc"
    return xr.open_dataset(f)["sp"].mean("valid_time").load()


def member_terms(path: Path, ps) -> dict:
    out = {}
    for name, *_ in _REGIONS:
        c = fd.load_cmip_corridor(path, name, ps)
        res = {}
        for lab, (w0, w1) in WINDOWS.items():
            i0 = np.where((fd.YEARS >= w0[0]) & (fd.YEARS <= w0[1]))[0]
            i1 = np.where((fd.YEARS >= w1[0]) & (fd.YEARS <= w1[1]))[0]
            r = fd.reduce_months(fd.window_terms(c, i0, i1, cc_split=False))
            r.pop("vhat")
            res[lab] = {k: round(v, 4) for k, v in r.items()}
        ys = fd.yearly_contributions(c)
        res["trend_full"] = fd.trend_table(ys, fd.YEARS)
        out[name] = res
    ds = xr.open_dataset(path)
    out["_valid_frac"] = {n: round(float(fd.load_cmip_corridor(path, n, ps).valid_frac), 3) for n, *_ in _REGIONS[:1]}
    tg = ds["tas_gm"].groupby("time.year").mean().to_series()
    w0, w1 = fd.HEADLINE
    out["_dT_headline"] = round(float(tg.loc[w1[0]:w1[1]].mean() - tg.loc[w0[0]:w0[1]].mean()), 3)
    return out


def era5_like_for_like() -> dict:
    out = {}
    for name, *_ in _REGIONS:
        c = fd.load_era5_corridor(name, levels_hpa=fd.CMIP_LEVELS_HPA, use_integrals=False)
        res = {}
        for lab, (w0, w1) in WINDOWS.items():
            i0 = np.where((fd.YEARS >= w0[0]) & (fd.YEARS <= w0[1]))[0]
            i1 = np.where((fd.YEARS >= w1[0]) & (fd.YEARS <= w1[1]))[0]
            r = fd.reduce_months(fd.window_terms(c, i0, i1, cc_split=False))
            r.pop("vhat")
            res[lab] = {k: round(v, 4) for k, v in r.items()}
        out[name] = res
    return out


def detection(members: dict, obs: dict) -> dict:
    out = {}
    for model in LARGE:
        mem = {k: v for k, v in members.items() if k.startswith(model + "/")}
        if len(mem) < 5:
            continue
        out[model] = {"n_members": len(mem)}
        for name, *_ in _REGIONS:
            row = {}
            for lab in WINDOWS:
                for term in ("wind", "moist", "total"):
                    x = np.array([m[name][lab][term] for m in mem.values()])
                    o = obs[name][lab][term]
                    mu, sd = float(x.mean()), float(x.std(ddof=1))
                    frac = float(np.mean(x >= o)) if o >= mu else float(np.mean(x <= o))
                    row[f"{lab}_{term}"] = {"obs": round(o, 3), "ens_mean": round(mu, 3),
                                            "ens_sd": round(sd, 3), "z": round((o - mu) / sd, 2),
                                            "frac_as_extreme": round(frac, 3),
                                            "min": round(float(x.min()), 3), "max": round(float(x.max()), 3)}
            out[model][name] = row
    return out


def main():
    ps = _ps_clim()
    members = {}
    for path in sorted(ROOT.glob("*/*.nc")):
        key = f"{path.parent.name}/{path.stem}"
        try:
            members[key] = member_terms(path, ps)
        except Exception as e:  # report and continue
            print("FAILED", key, e, flush=True)
    obs = era5_like_for_like()
    fd.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (fd.OUT_DIR / "cmip6_members.json").write_text(json.dumps({"obs_cmiplevels": obs, "members": members}, indent=1))
    det = detection(members, obs)
    (fd.OUT_DIR / "cmip6_detection.json").write_text(json.dumps(det, indent=1))
    print("members", len(members), "| large ensembles:", {k: v["n_members"] for k, v in det.items()})


if __name__ == "__main__":
    main()
