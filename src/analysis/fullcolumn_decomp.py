"""Level-resolved decomposition of corridor moisture-transport change.

The vertically integrated water-vapour transport vector at a grid point,

    F = (1/g) * integral_{p_top}^{p_s} q V dp ,

is split between two periods (0 = base, 1 = later) using window climatologies
of each calendar month (overbar) of specific humidity q and horizontal wind V
on pressure levels:

    dF =  (1/g) int  V0 dq   dp     moisture-change term
        + (1/g) int  q0 dV   dp     wind-change term
        + (1/g) int  dq dV   dp     covariance term
        + d( F_total - (1/g) int q V dp )   transient term.

The first three terms close the change of the stationary (climatological)
transport exactly; the transient term holds every change carried by
departures from the window climatology (sub-monthly weather systems when the
total comes from ERA5 vertical integrals of hourly fluxes, and month-to-month
departures in every case). Each vector term is projected, month by month, on
the direction of the base-window climatological transport of that calendar
month at the same grid point, area weighted over the corridor box and averaged
over calendar months (annual) or over a season. The change of the mean monthly
transport magnitude is kept alongside, so that a rotation of the flow is visible.

The moisture-change term is further split into the part expected from the local
temperature change at fixed relative humidity (Clausius-Clapeyron scaling of the
saturation vapour pressure, Bolton 1980) and a residual from changes in relative
humidity, including those from shifted moisture pathways.

The earlier three-term split with the moisture-weighted transport speed
V-hat = |F| / W (W = column water vapour) is kept as a cross-check.

Uncertainty comes from a moving-block bootstrap of whole years within each
window; two-sided p-values are adjusted with Benjamini-Hochberg across corridors.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from src.analysis.trend_stats import hamed_rao_mk
from src.figures.fig_global_ar_regions import _REGIONS

G = 9.80665
DATA = Path("/p/projects/poem/fallah/nature_ar_data")
ERA5_DIR = DATA / "era5_fullcolumn"
CORR_DIR = DATA / "era5_corridors"
OUT_DIR = DATA / "results_esd"
YEARS = np.arange(1940, 2025)
CMIP_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]
SEASONS = {"DJF": [11, 0, 1], "MAM": [2, 3, 4], "JJA": [5, 6, 7], "SON": [8, 9, 10]}
TERMS = ("moist", "wind", "cov", "trans")


def corridor_key(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


# ----------------------------------------------------------------------------- data
@dataclass
class Corridor:
    """Monthly fields over one corridor box, arranged (year, month, ...)."""

    name: str
    years: np.ndarray            # (Y,)
    p: np.ndarray                # (P,) Pa, ascending
    lat: np.ndarray              # (ny,)
    q: np.ndarray                # (Y, 12, P, ny, nx)
    u: np.ndarray
    v: np.ndarray
    wint: np.ndarray             # (P, ny, nx) vertical integration weights incl. 1/g
    t: np.ndarray | None = None  # (Y, 12, P, ny, nx)
    ftot: np.ndarray | None = None  # (Y, 12, 2, ny, nx) total transport (ERA5 integrals)
    tcwv: np.ndarray | None = None  # (Y, 12, ny, nx)

    valid: np.ndarray | None = None  # (ny, nx) mask of grid points with data

    def __post_init__(self):
        self.aw = np.cos(np.deg2rad(self.lat))[:, None] * np.ones(self.q.shape[-1])[None, :]
        if self.valid is not None:
            self.aw = self.aw * self.valid
        self.aw = self.aw / self.aw.sum()
        # monthly product transport, used as the total where no integral exists
        self.fmon = np.stack([np.einsum("pyx,tmpyx->tmyx", self.wint, self.q * self.u),
                              np.einsum("pyx,tmpyx->tmyx", self.wint, self.q * self.v)], axis=2)
        if self.ftot is None:
            self.ftot = self.fmon
        self.iwv = (np.einsum("pyx,tmpyx->tmyx", self.wint, self.q)
                    if self.tcwv is None else self.tcwv)


def layer_weights(p: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """Weights w(p, y, x) with sum_p w x ~ (1/g) int_{p_top}^{p_s} x dp.

    Trapezoid rule over the levels above the climatological surface pressure;
    the lowest level above ground is extended down to the surface.
    """
    P = len(p)
    w = np.zeros((P,) + ps.shape)
    for j in np.ndindex(ps.shape):
        ok = np.where(p <= ps[j])[0]
        if len(ok) == 0:
            continue
        pk = p[ok]
        wk = np.zeros(len(ok))
        if len(ok) > 1:
            dp = np.diff(pk)
            wk[:-1] += dp / 2
            wk[1:] += dp / 2
        wk[-1] += ps[j] - pk[-1]
        w[ok, j[0], j[1]] = wk
    return w / G


def _bolton_es(t):
    tc = t - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def _yr_month(da: xr.DataArray, years) -> np.ndarray:
    tdim = [d for d in da.dims if "time" in d][0]
    da = da.sel({tdim: slice(f"{years[0]}-01-01", f"{years[-1]}-12-31")}).transpose(tdim, ...)
    arr = da.values
    return arr.reshape((len(years), 12) + arr.shape[1:])


def load_era5_corridor(name: str, levels_hpa=None, use_integrals=True) -> Corridor:
    ds = xr.open_dataset(CORR_DIR / f"{corridor_key(name)}.nc")
    if levels_hpa is not None:
        ds = ds.sel(pressure_level=levels_hpa)
    ds = ds.sortby("pressure_level")
    p = ds["pressure_level"].values.astype(float) * 100.0
    ps = ds["sp"].mean("valid_time").values
    wint = layer_weights(p, ps)
    f = lambda v: _yr_month(ds[v], YEARS).astype("float64")  # noqa: E731
    ftot = np.stack([f("viwve"), f("viwvn")], axis=2) if use_integrals else None
    return Corridor(name, YEARS, p, ds["latitude"].values, f("q"), f("u"), f("v"), wint,
                    t=f("t"), ftot=ftot, tcwv=f("tcwv") if use_integrals else None)


def load_cmip_corridor(path: Path, name: str, ps_clim: xr.DataArray, years=YEARS) -> Corridor:
    k = corridor_key(name)
    ds = xr.open_dataset(path)
    q, u, v = (merge_dup_plev(ds[f"{x}_{k}"]) for x in ("hus", "ua", "va"))
    p = q["plev"].values.astype(float)
    lat, lon = q[f"lat_{k}"].values, q[f"lon_{k}"].values
    ps = ps_clim.interp(latitude=lat, longitude=lon, method="nearest",
                       kwargs={"fill_value": "extrapolate"}).values
    wint = layer_weights(p, ps)
    t = q.get_index("time")
    if t[0].year > years[0] or t[-1].year < years[-1]:
        raise ValueError(f"{path.name}: covers {t[0].year}-{t[-1].year}, needs {years[0]}-{years[-1]}")
    raw = [_yr_month(x, years).astype("float64") for x in (q, u, v)]
    # grid points masked at every level and time (e.g. a land or ocean mask in the store) are dropped
    # from the area mean; missing values elsewhere (below the model ground) carry no weight
    allnan = np.isnan(raw[0]).all(axis=(0, 1, 2))
    valid = (~allnan).astype("float64")
    if valid.sum() < 4:
        raise ValueError(f"{path.name} {name}: fewer than four valid grid points")
    nan_w = float(np.einsum("pyx,tmpyx->", wint, np.isnan(raw[0]) * valid) / (wint.sum() * raw[0].shape[0] * 12 * valid.mean()))
    if nan_w > 0.25:
        raise ValueError(f"{path.name} {name}: {nan_w:.0%} of the column weight is missing on valid points")
    c = Corridor(name, np.asarray(years), p, lat, *[np.nan_to_num(a) for a in raw], wint, valid=valid)
    c.nan_weight_frac = nan_w
    c.valid_frac = float(valid.mean())
    return c


# ----------------------------------------------------------------------------- core terms
def _clim(x, idx):
    return x[idx].mean(0)


def _proj_boxmean(vec, e, aw):
    """vec (..., 12, 2, ny, nx) projected on the unit vector e (12, 2, ny, nx) of the
    same calendar month, then area mean -> (..., 12)."""
    s = vec[..., 0, :, :] * e[:, 0] + vec[..., 1, :, :] * e[:, 1]
    return np.einsum("...yx,yx->...", s, aw)


def _unit_monthly(fclim):
    """Unit vectors (12, 2, ny, nx) of a monthly climatological transport field."""
    mag = np.hypot(fclim[:, 0], fclim[:, 1])
    return fclim / np.where(mag > 0, mag, 1.0)[:, None]


def merge_dup_plev(da: xr.DataArray) -> xr.DataArray:
    """Merge pressure levels that differ only by floating-point noise.

    Some models store slightly different plev values in their historical and
    scenario files; joining them gives duplicated, half-empty levels.
    """
    p = np.round(da["plev"].values.astype("float64"))
    da = da.assign_coords(plev=p)
    uniq = np.unique(p)
    if len(uniq) == len(p):
        return da.sortby("plev")
    parts = [da.isel(plev=np.where(p == u)[0]).mean("plev", skipna=True).expand_dims(plev=[u])
             for u in uniq]
    return xr.concat(parts, "plev").transpose(*da.dims).sortby("plev")


def window_terms(c: Corridor, idx0, idx1, cc_split=True) -> dict:
    """All decomposition terms, per calendar month (12,), for windows idx0 -> idx1."""
    W = c.wint
    q0, u0, v0 = _clim(c.q, idx0), _clim(c.u, idx0), _clim(c.v, idx0)
    q1, u1, v1 = _clim(c.q, idx1), _clim(c.u, idx1), _clim(c.v, idx1)
    dq, du, dv = q1 - q0, u1 - u0, v1 - v0
    integ = lambda a, b: np.einsum("pyx,mpyx->myx", W, a * b)  # noqa: E731
    vec = lambda a_u, a_v: np.stack([a_u, a_v], axis=1)  # noqa: E731
    moist = vec(integ(u0, dq), integ(v0, dq))
    wind = vec(integ(q0, du), integ(q0, dv))
    cov = vec(integ(dq, du), integ(dq, dv))
    fst0 = vec(integ(q0, u0), integ(q0, v0))
    fst1 = vec(integ(q1, u1), integ(q1, v1))
    fb0, fb1 = _clim(c.ftot, idx0), _clim(c.ftot, idx1)
    trans = (fb1 - fst1) - (fb0 - fst0)
    e = _unit_monthly(fb0)
    out = {
        "moist": _proj_boxmean(moist, e, c.aw),
        "wind": _proj_boxmean(wind, e, c.aw),
        "cov": _proj_boxmean(cov, e, c.aw),
        "trans": _proj_boxmean(trans, e, c.aw),
        "total": _proj_boxmean(fb1 - fb0, e, c.aw),
        "base": _proj_boxmean(fb0, e, c.aw),
        "dmag": np.einsum("myx,yx->m", np.hypot(fb1[:, 0], fb1[:, 1]) - np.hypot(fb0[:, 0], fb0[:, 1]), c.aw),
    }
    if cc_split and c.t is not None:
        t0, t1 = _clim(c.t, idx0), _clim(c.t, idx1)
        dq_cc = q0 * (_bolton_es(t1) / _bolton_es(t0) - 1.0)
        moist_cc = vec(integ(u0, dq_cc), integ(v0, dq_cc))
        out["moist_cc"] = _proj_boxmean(moist_cc, e, c.aw)
        out["moist_rh"] = out["moist"] - out["moist_cc"]
    # V-hat cross-check on annual box means of |F| and column water vapour
    I0 = np.einsum("myx,yx->", np.hypot(fb0[:, 0], fb0[:, 1]), c.aw) / 12
    I1 = np.einsum("myx,yx->", np.hypot(fb1[:, 0], fb1[:, 1]), c.aw) / 12
    W0 = np.einsum("myx,yx->", _clim(c.iwv, idx0), c.aw) / 12
    W1 = np.einsum("myx,yx->", _clim(c.iwv, idx1), c.aw) / 12
    V0, V1 = I0 / W0, I1 / W1
    out["vhat"] = np.array([V0 * (W1 - W0), W0 * (V1 - V0), (W1 - W0) * (V1 - V0), I1 - I0, I0])
    return out


def reduce_months(terms: dict, months=None) -> dict:
    sel = slice(None) if months is None else months
    return {k: (float(np.mean(v[sel])) if k != "vhat" else v.tolist()) for k, v in terms.items()}


# ----------------------------------------------------------------------------- bootstrap
def _block_idx(idx, rng, block):
    n = len(idx)
    if block <= 1:
        return idx[rng.integers(0, n, n)]
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return idx[np.concatenate([np.arange(s, s + block) for s in starts])[:n]]


def bootstrap(c: Corridor, idx0, idx1, n_boot=1000, block=3, seed=0, months=None):
    rng = np.random.default_rng(seed)
    keys = None
    reps = []
    for _ in range(n_boot):
        r = reduce_months(window_terms(c, _block_idx(idx0, rng, block),
                                       _block_idx(idx1, rng, block)), months)
        r.pop("vhat")
        keys = keys or list(r)
        reps.append([r[k] for k in keys])
    return keys, np.array(reps)


def two_sided_p(reps_col, point):
    if point >= 0:
        return float(min(1.0, 2 * np.mean(reps_col <= 0)))
    return float(min(1.0, 2 * np.mean(reps_col >= 0)))


def bh_fdr(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    r = p[o] * m / (np.arange(m) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.clip(r, 0, 1)
    return out


def window_summary(cors, w0, w1, n_boot=1000, block=3, months=None, fdr=True):
    """Point estimates, 95% intervals and FDR-adjusted p-values for every corridor."""
    i0 = np.where((YEARS >= w0[0]) & (YEARS <= w0[1]))[0]
    i1 = np.where((YEARS >= w1[0]) & (YEARS <= w1[1]))[0]
    rows = []
    for c in cors:
        pt = reduce_months(window_terms(c, i0, i1), months)
        keys, reps = bootstrap(c, i0, i1, n_boot, block, months=months)
        row = {"corridor": c.name, **{k: round(v, 3) for k, v in pt.items() if k != "vhat"},
               "vhat_thermo_dyn_cov_total_base": [round(x, 3) for x in pt["vhat"]]}
        for j, k in enumerate(keys):
            row[f"{k}_ci95"] = [round(float(np.percentile(reps[:, j], 2.5)), 3),
                                round(float(np.percentile(reps[:, j], 97.5)), 3)]
            row[f"{k}_p"] = round(two_sided_p(reps[:, j], pt[k]), 4)
        rows.append(row)
    if fdr:
        for k in TERMS + ("moist_cc", "moist_rh"):
            if f"{k}_p" in rows[0]:
                adj = bh_fdr([r[f"{k}_p"] for r in rows])
                for r, a in zip(rows, adj):
                    r[f"{k}_pfdr"] = round(float(a), 4)
    return rows


# ----------------------------------------------------------------------------- yearly series
def yearly_contributions(c: Corridor, months=None) -> dict:
    """Per-year anomaly contributions relative to the 1940-2024 climatology.

    F_y - Fbar = (1/g) int [Vbar q'_y + qbar V'_y + (q'V')_y - mean(q'V')] dp + E'_y,
    projected on the full-period mean transport direction. Exact closure.
    """
    allidx = np.arange(len(c.years))
    qb, ub, vb = _clim(c.q, allidx), _clim(c.u, allidx), _clim(c.v, allidx)
    qa, ua, va = c.q - qb, c.u - ub, c.v - vb
    W = c.wint
    integ = lambda a, b: np.einsum("pyx,tmpyx->tmyx", W, a * b)  # noqa: E731
    integb = lambda a, b: np.einsum("pyx,mpyx,tmpyx->tmyx", W, a, b)  # noqa: E731
    moist = np.stack([integb(ub, qa), integb(vb, qa)], axis=2)
    wind = np.stack([integb(qb, ua), integb(qb, va)], axis=2)
    cov = np.stack([integ(qa, ua), integ(qa, va)], axis=2)
    cov = cov - cov.mean(0, keepdims=True)
    fst = c.fmon  # sum_p W q_y V_y
    edd = c.ftot - fst
    edd = edd - edd.mean(0, keepdims=True)
    e = _unit_monthly(c.ftot.mean(0))
    sel = slice(None) if months is None else months
    red = lambda x: _proj_boxmean(x, e, c.aw)[:, sel].mean(1)  # noqa: E731
    total = c.ftot - c.ftot.mean(0, keepdims=True)
    return {"moist": red(moist), "wind": red(wind), "cov": red(cov), "trans": red(edd),
            "total": red(total), "base": float(_proj_boxmean(c.ftot.mean(0), e, c.aw).mean())}


def trend_table(series: dict, years, y0=1940, y1=2024) -> dict:
    m = (years >= y0) & (years <= y1)
    out = {}
    for k in TERMS + ("total",):
        r = hamed_rao_mk(series[k][m])
        out[k] = {"slope_per_decade": round(10 * r.slope, 3),
                  "pct_per_decade": round(1000 * r.slope / series["base"], 3),
                  "p": round(r.p, 4)}
    return out


def index_regression(series, indices: dict, years, w0, w1):
    """Regress the yearly wind term on standardised indices plus a linear time term.

    Returns explained variance, coefficients and the window difference of the wind
    term after the index-congruent part is removed.
    """
    y = series["wind"]
    names = list(indices)
    X = np.column_stack([np.ones_like(y), (years - years.mean()) / 10.0] +
                        [(indices[n] - indices[n].mean()) / indices[n].std() for n in names])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    idx_part = X[:, 2:] @ beta[2:]
    resid = y - idx_part
    r2_idx = 1 - np.var(y - idx_part - beta[0]) / np.var(y)
    m0 = (years >= w0[0]) & (years <= w0[1])
    m1 = (years >= w1[0]) & (years <= w1[1])
    return {"coef": {n: round(float(b), 3) for n, b in zip(["const", "trend_per_decade"] + names, beta)},
            "r2_indices": round(float(r2_idx), 3),
            "wind_diff_raw": round(float(y[m1].mean() - y[m0].mean()), 3),
            "wind_diff_index_removed": round(float(resid[m1].mean() - resid[m0].mean()), 3)}


# ----------------------------------------------------------------------------- driver
HEADLINE = ((1940, 1969), (1995, 2024))
SATELLITE = ((1979, 2001), (2002, 2024))
ORIGINAL = ((1940, 1959), (2015, 2024))


def sensitivity_matrix(cors, n_boot=300):
    rows = []
    for L in (10, 15, 20, 25, 30, 35, 40):
        for shift in (0, 5, 10):
            w0 = (1940 + shift, 1940 + shift + L - 1)
            w1 = (2024 - L + 1, 2024)
            if w0[1] >= w1[0]:
                continue
            for r in window_summary(cors, w0, w1, n_boot=n_boot, block=3):
                rows.append({"L": L, "base_start": w0[0], "corridor": r["corridor"],
                             "wind": r["wind"], "wind_p": r["wind_p"], "total": r["total"],
                             "wind_pfdr": r["wind_pfdr"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["headline", "sensitivity", "seasons", "levels"], required=True)
    ap.add_argument("--nboot", type=int, default=1000)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.part == "levels":
        cors = [load_era5_corridor(n, levels_hpa=CMIP_LEVELS_HPA, use_integrals=False) for n, *_ in _REGIONS]
    else:
        cors = [load_era5_corridor(n) for n, *_ in _REGIONS]
    res = {}
    if a.part in ("headline", "levels"):
        for lab, (w0, w1) in {"headline": HEADLINE, "satellite": SATELLITE, "original": ORIGINAL}.items():
            res[lab] = window_summary(cors, w0, w1, n_boot=a.nboot, block=3)
            res[lab + "_block5"] = window_summary(cors, w0, w1, n_boot=a.nboot, block=5)
        ys = {c.name: yearly_contributions(c) for c in cors}
        res["trends_full"] = {k: trend_table(v, YEARS) for k, v in ys.items()}
        res["trends_sat"] = {k: trend_table(v, YEARS, 1979, 2024) for k, v in ys.items()}
        np.savez(OUT_DIR / f"yearly_{a.part}.npz",
                 **{f"{corridor_key(k)}__{t}": v[t] for k, v in ys.items() for t in TERMS + ("total",)})
    elif a.part == "sensitivity":
        res["matrix"] = sensitivity_matrix(cors, n_boot=min(a.nboot, 300))
    elif a.part == "seasons":
        for s, months in SEASONS.items():
            res[s] = window_summary(cors, *HEADLINE, n_boot=min(a.nboot, 500), block=3, months=months)
            ys = {c.name: yearly_contributions(c, months) for c in cors}
            np.savez(OUT_DIR / f"yearly_{s}.npz",
                     **{f"{corridor_key(k)}__{t}": v[t] for k, v in ys.items() for t in TERMS + ("total",)})
    p = OUT_DIR / f"fullcolumn_{a.part}.json"
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
