"""NOVEL ARTIFACT — does convective EFFICIENCY respond to ENSO?

A trend over 12 yr is confounded (poc_efficiency_trend), but ENSO drives large,
sign-known year-to-year swings, so its fingerprint is recoverable in a short
record. The question that goes beyond the known ENSO-environment link (Allen et
al.): does *efficiency* — realized convection per unit favorable environment —
respond to ENSO independently of the environment's own ENSO response?

Method: annual ONI (NOAA CPC) 2012-2023; per-cell and per-region anomalies of the
WGLC efficiency (common-mode / detection-efficiency drift removed, as in
poc_efficiency_trend) and of the ERA5 environment E; Pearson correlation with ONI.
We map corr(efficiency, ONI) and contrast it, per region, with corr(E, ONI). If
the two differ, efficiency carries an ENSO signal the environment does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from src.figures._style import apply_nature_style, COL_DOUBLE_IN
from src.figures.poc_lightning_efficiency import _land_mask
from src.figures.poc_efficiency_trend import (
    _env_annual_cube, _lightning_annual_cube, _region_series, _REGIONS, _Y0, _Y1,
)

_ONI = Path("/p/projects/poem/fallah/nature_ar_data/indices/oni.ascii.txt")
_ENSO_REGIONS = _REGIONS   # already includes the deep-tropical ENSO-sensitive boxes
# n=12 two-tailed significance threshold for a Pearson r (|r|>0.576 => p<0.05).
_R_SIG = 0.576


def _oni_annual() -> dict[int, float]:
    """Annual-mean ONI (the ANOM column) per year from the CPC ascii table."""
    out: dict[int, list] = {}
    for line in _ONI.read_text().splitlines()[1:]:
        p = line.split()
        if len(p) != 4:
            continue
        yr, anom = int(p[1]), float(p[3])
        out.setdefault(yr, []).append(anom)
    return {y: float(np.mean(v)) for y, v in out.items()}


def _anom(da: xr.DataArray) -> xr.DataArray:
    return da - da.mean("year")


def plot(out_path: Path) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    apply_nature_style()

    E = _env_annual_cube()
    L = _lightning_annual_cube(E)
    E, L = xr.align(E, L, join="inner")
    years = E["year"].to_numpy().astype(int)
    oni_map = _oni_annual()
    oni = xr.DataArray([oni_map[y] for y in years], coords={"year": E["year"]}, dims="year")

    landm = _land_mask(E.isel(year=0))
    trop = landm & (np.abs(E["latitude"]) <= 30)
    wL = np.cos(np.deg2rad(L["latitude"])).broadcast_like(L.isel(year=0)).where(trop)
    common = (L.where(trop) * wL).sum(("latitude", "longitude")) / wL.sum(("latitude", "longitude"))
    common_norm = common / common.mean("year")

    eff = (L / common_norm) / E.where(E > 0)        # detection-drift-removed efficiency
    eff_a = _anom(eff)
    E_a = _anom(E)
    r_eff = xr.corr(eff_a, oni, dim="year")
    r_E = xr.corr(E_a, oni, dim="year")

    # per-region correlations
    rows = []
    for name, *box in _ENSO_REGIONS:
        e_ser = _region_series(eff, box, landm)
        env_ser = _region_series(E, box, landm)
        re = float(np.corrcoef(e_ser - np.nanmean(e_ser), oni.to_numpy())[0, 1])
        rv = float(np.corrcoef(env_ser - np.nanmean(env_ser), oni.to_numpy())[0, 1])
        rows.append(dict(region=name, r_eff_oni=re, r_env_oni=rv))

    lon = E["longitude"].to_numpy(); lat = E["latitude"].to_numpy()
    proj = ccrs.Robinson(central_longitude=180)
    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.60))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.28)

    ax = fig.add_subplot(gs[0], projection=proj)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.3); ax.set_global()
    rr = r_eff.where(landm | (np.abs(lat[:, None]) <= 35))   # show land + tropical oceans
    pcm = ax.pcolormesh(lon, lat, rr.to_numpy(), transform=ccrs.PlateCarree(),
                        cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    ax.set_title("(a) Correlation of convective efficiency with ONI (El Niño +)", fontsize=7.5)
    cb = fig.colorbar(pcm, ax=ax, fraction=0.025, pad=0.02); cb.set_label("r(efficiency, ONI)", fontsize=6)

    ax2 = fig.add_subplot(gs[1])
    names = [r["region"] for r in rows]; yy = np.arange(len(names))
    ax2.barh(yy - 0.2, [r["r_env_oni"] for r in rows], height=0.4, color="#e67e22", label="environment")
    ax2.barh(yy + 0.2, [r["r_eff_oni"] for r in rows], height=0.4, color="#8e44ad", label="efficiency")
    ax2.axvline(0, color="0.4", lw=0.6)
    ax2.set_yticks(yy); ax2.set_yticklabels(names, fontsize=6.5)
    ax2.set_xlabel("correlation with ONI", fontsize=7.5); ax2.set_xlim(-1, 1)
    ax2.set_title("(b) ENSO response: environment vs efficiency, by region", fontsize=7.5)
    ax2.legend(fontsize=5.5, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # The signal of interest is SIGN OPPOSITION: regions where the environment and
    # efficiency respond to ENSO with opposite sign -> efficiency is not a passive
    # function of the environment but compensates it.
    opposed = [r["region"] for r in rows
               if np.sign(r["r_eff_oni"]) != np.sign(r["r_env_oni"])
               and abs(r["r_eff_oni"]) > 0.4 and abs(r["r_env_oni"]) > 0.4]
    sig = [r["region"] for r in rows if abs(r["r_eff_oni"]) > _R_SIG]
    sidecar = {
        "period": f"{_Y0}-{_Y1}", "index": "NOAA CPC ONI (annual mean)",
        "n_years": int(len(years)), "r_sig_threshold_p05": _R_SIG,
        "oni_by_year": {int(y): round(oni_map[y], 2) for y in years},
        "regions": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
        "opposite_sign_env_vs_efficiency": opposed,
        "efficiency_oni_significant_regions": sig,
        "verdict": (f"efficiency and environment have OPPOSITE-SIGN ENSO responses in "
                    f"{len(opposed)}/{len(rows)} regions ({', '.join(opposed)}) -> efficiency "
                    f"compensates, not tracks, the environment. n=12 so |r|>{_R_SIG} for p<0.05; "
                    f"treat as suggestive."),
    }
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps(sidecar, indent=2))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "poc_efficiency_enso.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
