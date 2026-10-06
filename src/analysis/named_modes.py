"""Named-mode regression: link each corridor's dynamic term to recognised
teleconnections.

The EOF/machine-learning attribution (mode_attribution.py) shows the dynamic term
projects onto data-driven circulation modes but does not name them. Here we compute
classical teleconnection indices directly from the annual 500 hPa height field --
the North Atlantic Oscillation (NAO), the Southern Annular Mode (SAM), the
Pacific-North American pattern (PNA) and a tropical-mean upper-height ENSO proxy --
and correlate each corridor's transport-efficiency anomaly with them. Sea-surface
temperature is not on disk, so the ENSO proxy is height-based and the SST
confirmation is left to future work. All indices are standardised annual values
computed from the same ERA5 windows; no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import pearsonr

from src import config
from src.analysis.decomposition import _DIR, _FILES
from src.analysis.mode_attribution import corridor_records
from src.figures.fig_global_ar_regions import _REGIONS

_G = 9.80665


def _annual_z500():
    fields, years = [], []
    lat = lon = None
    for w in ["presat", "gap1", "modern", "gap2", "recent"]:
        ds = xr.open_dataset(_DIR / _FILES[w])
        lev = "pressure_level" if "pressure_level" in ds.dims else "level"
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        z = (ds["z"].sel({lev: 500}) / _G)
        zy = z.groupby(z[tdim].dt.year).mean(tdim)
        if lat is None:
            lat = ds["latitude"].to_numpy(); lon = ds["longitude"].to_numpy()
        fields.append(zy.transpose("year", "latitude", "longitude").to_numpy())
        years.append(zy["year"].to_numpy())
        ds.close()
    Z = np.concatenate(fields, 0); yr = np.concatenate(years)
    o = np.argsort(yr)
    return yr[o], Z[o], lat, lon


def _boxmean(Z, lat, lon, la0, la1, lo0, lo1):
    """Cos-lat box mean of Z[T,ny,nx]; lon in 0..360."""
    lo0 %= 360; lo1 %= 360
    jm = (lat >= min(la0, la1)) & (lat <= max(la0, la1))
    im = (lon >= lo0) & (lon <= lo1) if lo0 <= lo1 else ((lon >= lo0) | (lon <= lo1))
    w = np.cos(np.deg2rad(lat[jm]))[:, None]
    sub = Z[:, jm][:, :, im]
    return (sub * w).sum((1, 2)) / (w.sum() * im.sum())


def _std(x):
    return (x - x.mean()) / x.std()


def indices():
    yr, Z, lat, lon = _annual_z500()
    # NAO: Azores (38N,332E) minus Iceland (65N,338E), z500 anomaly dipole
    nao = _std(_boxmean(Z, lat, lon, 33, 43, 327, 337) - _boxmean(Z, lat, lon, 60, 70, 333, 343))
    # SAM: zonal-mean 40S minus 65S
    sam = _std(_boxmean(Z, lat, lon, -45, -35, 0, 359) - _boxmean(Z, lat, lon, -70, -60, 0, 359))
    # PNA: Wallace-Gutzler 4-point on z500 anomalies
    p1 = _boxmean(Z, lat, lon, 15, 25, 180, 220)
    p2 = _boxmean(Z, lat, lon, 40, 50, 180, 220)
    p3 = _boxmean(Z, lat, lon, 45, 60, 235, 255)
    p4 = _boxmean(Z, lat, lon, 25, 35, 270, 290)
    pna = _std(_std(p1) - _std(p2) + _std(p3) - _std(p4))
    # ENSO proxy: tropical-belt (20S-20N) mean upper height (warm ENSO -> high)
    enso = _std(_boxmean(Z, lat, lon, -20, 20, 0, 359))
    return yr, {"NAO": nao, "SAM": sam, "PNA": pna, "ENSO*": enso}


def regress():
    yr, idx = indices()
    recs = corridor_records()
    names = list(idx)
    rows = []
    for region, *_ in _REGIONS:
        r = recs[region]
        _, ia, ib = np.intersect1d(r["year"], yr, return_indices=True)
        vhat = _std(r["vhat"][ia])
        corr = {}
        for n in names:
            cc, pp = pearsonr(idx[n][ib], vhat)
            corr[n] = (round(float(cc), 3), round(float(pp), 4))
        top = max(names, key=lambda n: abs(corr[n][0]))
        rows.append(dict(region=region, corr=corr, top_mode=top,
                         top_r=corr[top][0], top_p=corr[top][1]))
    return names, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    names, rows = regress()
    (config.CACHE_DIR / "named_modes.json").write_text(
        json.dumps(dict(modes=names, corridors=rows), indent=2))
    hdr = "corridor".ljust(20) + "".join(f"{n:>9s}" for n in names) + "   top"
    print(hdr)
    for r in rows:
        line = r["region"].ljust(20) + "".join(
            f"{r['corr'][n][0]:+9.2f}" for n in names)
        star = "*" if r["top_p"] < 0.05 else ""
        print(f"{line}   {r['top_mode']}{star} (r={r['top_r']:+.2f})")


if __name__ == "__main__":
    main()
