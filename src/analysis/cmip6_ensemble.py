"""CMIP6 multi-model ensemble projection anchor.

We apply the identical three-term decomposition (thermodynamic / dynamic /
covariance) used on ERA5 to a CMIP6 ensemble, contrasting a recent-historical
baseline (1995-2014) with an end-of-century ssp585 window (2081-2100), for the
same eight corridors. For each model we obtain a projected dynamic fraction per
corridor; the ensemble distribution (median and inter-model spread) is compared
with the observed dynamic fraction. This replaces the earlier single-model anchor
and tests whether the under-representation of the observed dynamic contribution is
a robust feature of the CMIP6 circulation response rather than one model's quirk.

All models use Amon monthly hus/ua/va on standard pressure levels, read from the
PIK shared CMIP6 archive. No network. To stay within memory limits we never
materialise a global field: each corridor box is selected lazily and only the
reduced (box-mean, annual) series is computed.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json

import numpy as np
import xarray as xr

from src import config
from src.analysis.decomposition import _three_term
from src.figures.fig_global_ar_regions import _REGIONS

_G = 9.80665
_C6 = "/p/projects/climate_data_central/CMIP/CMIP6"
_LEVELS = [85000.0, 50000.0, 25000.0]  # Pa, to match the ERA5 three-level integral
_BASE = (1995, 2014)  # recent-historical baseline
_FUT = (2081, 2100)  # end-of-century ssp585

# (model, member) pairs with hus+ua+va present in BOTH historical and ssp585
# (a shared member), discovered from the PIK archive. r1i1p1f1 unless the model
# only provides an f2 forcing variant (CNRM, UKESM).
_MODELS = [
    ("BCC-CSM2-MR", "r1i1p1f1"),
    ("CAMS-CSM1-0", "r1i1p1f1"),
    ("CNRM-ESM2-1", "r1i1p1f2"),
    ("CanESM5", "r1i1p1f1"),
    ("EC-Earth3", "r1i1p1f1"),
    ("EC-Earth3-Veg", "r1i1p1f1"),
    ("GFDL-CM4", "r1i1p1f1"),
    ("GFDL-ESM4", "r1i1p1f1"),
    ("IPSL-CM6A-LR", "r1i1p1f1"),
    ("MIROC6", "r1i1p1f1"),
    ("MPI-ESM1-2-HR", "r1i1p1f1"),
    ("MRI-ESM2-0", "r1i1p1f1"),
    ("NESM3", "r1i1p1f1"),
    ("UKESM1-0-LL", "r1i1p1f2"),
]


def _files(model, member, var, experiment):
    if experiment == "historical":
        pat = f"{_C6}/CMIP/*/{model}/historical/{member}/Amon/{var}/*/*/*.nc"
    else:
        pat = f"{_C6}/ScenarioMIP/*/{model}/{experiment}/{member}/Amon/{var}/*/*/*.nc"
    return sorted(glob.glob(pat))


def _open(model, member, var, experiment, yr0, yr1):
    files = _files(model, member, var, experiment)
    if not files:
        raise FileNotFoundError(f"{model} {member} {var} {experiment}")
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 120}, decode_times=True)
    da = ds[var]
    # nearest-match the three target levels (handles plev8/plev19/plev27 sets);
    # tolerance keeps us from silently grabbing a wrong level if one is absent.
    da = da.sel(plev=_LEVELS, method="nearest", tolerance=2500.0)
    da = da.sel(time=slice(f"{yr0}-01", f"{yr1}-12"))
    return da


def _corridor_annual(qa, ua, va, box):
    """Lazy: select the corridor box, integrate, box-mean, annual-mean, then compute."""
    lon0, lon1, lat0, lat1 = box
    lat_asc = float(qa.lat[0]) < float(qa.lat[-1])
    latsl = slice(lat0, lat1) if lat_asc else slice(lat1, lat0)
    q = qa.sel(lon=slice(lon0, lon1), lat=latsl)
    u = ua.sel(lon=slice(lon0, lon1), lat=latsl)
    v = va.sel(lon=slice(lon0, lon1), lat=latsl)
    ivt = np.hypot((q * u).integrate("plev") / _G, (q * v).integrate("plev") / _G)
    iwv = q.integrate("plev") / _G
    w = np.cos(np.deg2rad(q["lat"]))
    ivt_y = ivt.weighted(w).mean(("lat", "lon")).groupby("time.year").mean("time")
    iwv_y = iwv.weighted(w).mean(("lat", "lon")).groupby("time.year").mean("time")
    return ivt_y.to_numpy(), iwv_y.to_numpy()


def _window_series(model, member, experiment, yr0, yr1):
    """Load one window's q/u/v (3 levels only) and return per-corridor (ivt, iwv)."""
    q = _open(model, member, "hus", experiment, yr0, yr1).load()
    u = _open(model, member, "ua", experiment, yr0, yr1).load()
    v = _open(model, member, "va", experiment, yr0, yr1).load()
    out = {name: _corridor_annual(q, u, v, box) for name, *box in _REGIONS}
    del q, u, v
    gc.collect()
    return out


def _one_model(model, member, experiment="ssp585"):
    base = _window_series(model, member, "historical", *_BASE)
    fut = _window_series(model, member, experiment, *_FUT)
    rows = []
    for name, *_ in _REGIONS:
        i0, w0 = base[name]
        i1, w1 = fut[name]
        d_tot, d_th, d_dy, d_cv = _three_term(i0, w0, i1, w1)
        rows.append(
            dict(
                region=name,
                base_ivt=round(float(i0.mean()), 1),
                d_total=round(float(d_tot), 2),
                d_thermo=round(float(d_th), 2),
                d_dyn=round(float(d_dy), 2),
                d_cov=round(float(d_cv), 2),
                dyn_frac=round(float(d_dy / d_tot), 3) if d_tot else None,
            )
        )
    return rows


def compute(experiment="ssp585"):
    per_model = {}
    partial = config.CACHE_DIR / "cmip6_ensemble.partial.json"
    for k, (model, member) in enumerate(_MODELS, 1):
        try:
            per_model[model] = _one_model(model, member, experiment)
            fracs = [r["dyn_frac"] for r in per_model[model]]
            print(
                f"  ok  [{k:2d}/{len(_MODELS)}] {model:18s} "
                + " ".join(f"{100 * v:+.0f}%" if v is not None else "  nan" for v in fracs),
                flush=True,
            )
        except Exception as exc:  # one bad model must not kill the ensemble
            print(
                f"  SKIP [{k:2d}/{len(_MODELS)}] {model:18s} ({type(exc).__name__}: {exc})",
                flush=True,
            )
        # crash-safe progress: rewrite the partial cache after every model
        partial.write_text(json.dumps({"done": list(per_model), "per_model": per_model}))
        gc.collect()

    return _aggregate(per_model)


def _aggregate(per_model):
    """Per-corridor ensemble statistics over the models that produced a finite
    dynamic fraction (drops None and NaN, e.g. a model whose grid/calendar
    yields an empty corridor selection)."""
    names = [n for n, *_ in _REGIONS]
    ensemble = []
    for name in names:
        fr = [
            r["dyn_frac"]
            for m in per_model.values()
            for r in m
            if r["region"] == name and r["dyn_frac"] is not None and np.isfinite(r["dyn_frac"])
        ]
        fr = np.array(fr, dtype=float)
        ensemble.append(
            dict(
                region=name,
                n_models=int(fr.size),
                dyn_frac_median=round(float(np.median(fr)), 3),
                dyn_frac_q25=round(float(np.percentile(fr, 25)), 3),
                dyn_frac_q75=round(float(np.percentile(fr, 75)), 3),
                dyn_frac_min=round(float(fr.min()), 3),
                dyn_frac_max=round(float(fr.max()), 3),
                dyn_frac_models=[round(float(x), 3) for x in fr],
            )
        )
    return per_model, ensemble


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="ssp585")
    args = ap.parse_args()
    per_model, ensemble = compute(args.experiment)
    out = config.CACHE_DIR / "cmip6_ensemble.json"
    out.write_text(
        json.dumps(
            dict(
                experiment=args.experiment,
                base=_BASE,
                future=_FUT,
                models=[m for m, _ in _MODELS],
                members={m: mem for m, mem in _MODELS},
                per_model=per_model,
                ensemble=ensemble,
            ),
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    print(f"{'corridor':20s} {'n':>3s} {'median':>7s} {'q25':>6s} {'q75':>6s}")
    for e in sorted(ensemble, key=lambda e: -e["dyn_frac_median"]):
        print(
            f"{e['region']:20s} {e['n_models']:3d} "
            f"{100 * e['dyn_frac_median']:6.0f}% {100 * e['dyn_frac_q25']:5.0f}% "
            f"{100 * e['dyn_frac_q75']:5.0f}%"
        )


if __name__ == "__main__":
    main()
