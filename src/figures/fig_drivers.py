"""Figure — Spatial fingerprint and thermodynamic/dynamic drivers of the
observed US West Coast atmospheric-river intensification.

This is the depth the period-contrast bars cannot show. From the per-window
time-mean 4-D fields (``data/cache/spatial_composites.nc``) it builds:

  Row 1  Mean column IVT magnitude + transport vectors for each window
         (pre-satellite, modern, recent) — *where* the moisture corridor sits
         and how it strengthens.

  Row 2  (d) the IVT change recent - pre-satellite;
         (e) the THERMODYNAMIC expectation (IVT scaled by the column-water-vapour
             ratio IWV_recent/IWV_presat — i.e. Clausius-Clapeyron moistening at
             fixed circulation);
         (f) the DYNAMIC residual (total change minus thermodynamic) — the part
             attributable to a change in the transporting circulation.

Decomposition (per pixel), following the standard IVT = IWV x V_eff split:

    ΔIVT_thermo = IVT_0 (IWV_1/IWV_0 - 1)
    ΔIVT_dyn    = ΔIVT_total - ΔIVT_thermo

with IWV = -(1/g) ∫ q dp over the same three levels used for IVT, so the two
quantities are mutually consistent. A corridor-mean split is annotated so the
reader gets the one-number answer: how much of the intensification is "wetter
air" versus "stronger winds".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from src import config
from src.figures._style import apply_nature_style, COL_DOUBLE_IN

_G = 9.80665
_WINDOW_LABEL = {
    "presat": "Pre-satellite 1940–1959",
    "modern": "Modern 1980–1999",
    "recent": "Recent 2015–2024",
}


def _iwv(qbar: xr.DataArray) -> xr.DataArray:
    """Column water vapour (kg m^-2, positive) from the per-level mean specific
    humidity, integrated over pressure on the same three levels used for IVT so
    the two are mutually consistent. (Only 850/500/250 hPa are available, so this
    under-samples the moist boundary layer and is a consistent *lower bound* on
    true precipitable water — fine for inter-window ratios, which is all the
    decomposition uses.)"""
    lev = "level" if "level" in qbar.dims else "pressure_level"
    q = qbar.sortby(lev)
    q = q.assign_coords({lev: q[lev].astype("float32") * 100.0})  # hPa -> Pa
    # ∫ q dp over ascending pressure is positive precipitable water.
    return (q.integrate(lev) / _G).rename("iwv")


def _corridor_mean(da: xr.DataArray, ivt_ref: xr.DataArray) -> float:
    """Mean of ``da`` weighted to the AR corridor (where mean IVT is strong),
    so the headline number reflects the landfalling-AR region, not quiet ocean."""
    w = ivt_ref.where(ivt_ref > float(ivt_ref.quantile(0.6)), 0.0)
    return float((da * w).sum() / w.sum())


def plot(composites_path: Path, out_path: Path) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    apply_nature_style()
    ds = xr.open_dataset(composites_path)
    wins = [w for w in ("presat", "modern", "recent") if w in ds["window"].values]
    g = {w: ds.sel(window=w) for w in wins}

    iwv = {w: _iwv(g[w]["qbar"]) for w in wins}
    ivt = {w: g[w]["ivt"] for w in wins}

    a, b = wins[0], wins[-1]                 # pre-sat -> recent (longest baseline)
    d_total = ivt[b] - ivt[a]
    d_thermo = ivt[a] * (iwv[b] / iwv[a] - 1.0)
    d_dyn = d_total - d_thermo

    cm_total = _corridor_mean(d_total, ivt[a])
    cm_thermo = _corridor_mean(d_thermo, ivt[a])
    cm_dyn = _corridor_mean(d_dyn, ivt[a])

    # --- Clausius-Clapeyron test ---------------------------------------------
    # Is the thermodynamic moistening consistent with ~7 %/K column scaling?
    ivt0 = _corridor_mean(ivt[a], ivt[a])
    iwv0 = _corridor_mean(iwv[a], ivt[a])
    iwv1 = _corridor_mean(iwv[b], ivt[a])
    moist_pct = 100.0 * (iwv1 / iwv0 - 1.0)            # observed δIWV/IWV (%)
    ivt_pct = 100.0 * cm_total / ivt0                 # observed δIVT/IVT (%)
    dT = None
    if "tbar" in g[a]:
        lev = "level" if "level" in g[a]["tbar"].dims else "pressure_level"
        t0 = _corridor_mean(g[a]["tbar"].sel({lev: 850}), ivt[a])
        t1 = _corridor_mean(g[b]["tbar"].sel({lev: 850}), ivt[a])
        dT = t1 - t0                                   # 850-hPa warming (K)
    cc_rate = (moist_pct / dT) if dT else float("nan")  # implied %/K
    cc_expected = 7.0 * dT if dT else float("nan")      # CC-predicted δIWV/IWV (%)

    proj = ccrs.PlateCarree()
    lon0 = config.BBOX["lon_min"] - 360
    lon1 = config.BBOX["lon_max"] - 360
    lat0, lat1 = config.BBOX["lat_min"], config.BBOX["lat_max"]
    lons = ivt[a]["longitude"].to_numpy()
    lons_m = np.where(lons > 180, lons - 360, lons)
    lats = ivt[a]["latitude"].to_numpy()

    def _basemap(ax):
        ax.set_extent([lon0, lon1, lat0, lat1], crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="0.4")
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.2, edgecolor="0.6")

    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.62))
    gs = fig.add_gridspec(2, 3, hspace=0.18, wspace=0.10)

    vmax = max(float(ivt[w].max()) for w in wins)
    # --- Row 1: mean IVT magnitude + transport vectors -----------------------
    for j, w in enumerate(wins):
        ax = fig.add_subplot(gs[0, j], projection=proj)
        _basemap(ax)
        pcm = ax.pcolormesh(lons_m, lats, ivt[w].to_numpy(), transform=proj,
                            cmap="YlGnBu", vmin=0, vmax=vmax, shading="auto")
        st = 8
        # The pipeline stores IVT components with a flipped sign (harmless for
        # magnitude/detection); negate for physical transport direction (the
        # moist plume points onshore, eastward/northeastward).
        ax.quiver(lons_m[::st], lats[::st],
                  -g[w]["ivt_u"].to_numpy()[::st, ::st],
                  -g[w]["ivt_v"].to_numpy()[::st, ::st],
                  transform=proj, scale=2600, width=0.005, color="0.15")
        ax.set_title(f"({chr(97+j)}) {_WINDOW_LABEL[w]}", fontsize=7)
        if j == len(wins) - 1:
            cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("mean IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=6)

    # --- Row 2: change + thermodynamic + dynamic -----------------------------
    dmax = float(np.nanpercentile(np.abs(d_total.to_numpy()), 99))
    panels = [
        (d_total, f"(d) Total change\n(recent − pre-sat), corridor +{cm_total:.0f}"),
        (d_thermo, f"(e) Thermodynamic\n(moisture/CC), +{cm_thermo:.0f}"),
        (d_dyn, f"(f) Dynamic\n(circulation), {cm_dyn:+.0f}"),
    ]
    for j, (field, title) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j], projection=proj)
        _basemap(ax)
        pcm = ax.pcolormesh(lons_m, lats, field.to_numpy(), transform=proj,
                            cmap="RdBu_r", vmin=-dmax, vmax=dmax, shading="auto")
        ax.set_title(title, fontsize=7)
        if j == len(panels) - 1:
            cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("ΔIVT (kg m$^{-1}$ s$^{-1}$)", fontsize=6)

    thermo_pct = 100 * cm_thermo / cm_total if cm_total else float("nan")
    cc_str = ""
    if dT and np.isfinite(dT):
        cc_str = (f"  |  +{dT:.2f} K warming → moisture +{moist_pct:.1f}% "
                  f"({cc_rate:.1f}%/K vs CC 7%/K)")
    fig.suptitle(
        f"Corridor IVT {cm_total:+.0f} kg m$^{{-1}}$ s$^{{-1}}$ "
        f"({ivt_pct:+.1f}%): thermodynamic {thermo_pct:.0f}%, dynamic {100 - thermo_pct:.0f}%"
        + cc_str,
        fontsize=7.5, y=0.995)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # Sidecar so the manuscript numbers are read from the same computation that
    # made the figure (no hand-copied values).
    import json
    sidecar = {
        "delta_ivt_corridor": round(cm_total, 1),
        "delta_ivt_pct": round(ivt_pct, 1),
        "thermo_pct": round(thermo_pct),
        "dyn_pct": round(100 - thermo_pct),
        "delta_T_850": round(float(dT), 2) if dT else None,
        "moist_pct": round(moist_pct, 1),
        "cc_rate_per_K": round(cc_rate, 1) if dT else None,
    }
    (config.CACHE_DIR / "driver_attribution.json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {out_path}")
    print(f"corridor ΔIVT total={cm_total:.1f} ({ivt_pct:+.1f}%), "
          f"thermo={cm_thermo:.1f} ({thermo_pct:.0f}%), dyn={cm_dyn:.1f}")
    if dT and np.isfinite(dT):
        print(f"CC test: ΔT(850)={dT:.2f}K, δIWV/IWV={moist_pct:.1f}% "
              f"→ {cc_rate:.1f}%/K (CC≈7%/K); CC-expected moisture {cc_expected:.1f}%")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--composites", type=Path,
                    default=config.CACHE_DIR / "spatial_composites.nc")
    ap.add_argument("--out", type=Path,
                    default=Path("figures") / "fig_drivers.pdf")
    args = ap.parse_args()
    plot(args.composites, args.out)


if __name__ == "__main__":
    main()
