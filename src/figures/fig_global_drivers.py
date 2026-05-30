"""Figure — Global context: thermodynamic vs dynamic change in vertically
integrated moisture transport.

The event-based intensification is a US West Coast result. To place it in the
global picture we use ERA5 *monthly-mean* pressure-level fields (global, 1 deg,
850/500/250 hPa) for the same three windows and decompose the change in the
mean-flow IVT exactly as in Fig. 3:

    IVT = IWV · V̂,   ΔIVT_thermo = IVT_0 (IWV_1/IWV_0 − 1),   ΔIVT_dyn = ΔIVT − ΔIVT_thermo

Caveat: monthly means capture the *mean-flow* transport, not the sub-monthly
transient (synoptic) flux that dominates individual ARs, so the magnitudes are a
mean-circulation lower bound; the spatial pattern and the thermodynamic/dynamic
partition are the message. The US West Coast study box is drawn for reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from src import config
from src.figures._style import apply_nature_style, COL_DOUBLE_IN

_G = 9.80665
_GLOBAL_DIR = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")
_WINS = ["presat", "modern", "recent"]


def _mean_fields(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    return ds[["q", "u", "v", "t", "z"]].mean(tdim)


def _ivt_iwv(ds: xr.Dataset):
    lev = "pressure_level" if "pressure_level" in ds.dims else "level"
    q = ds["q"].sortby(lev)
    u = ds["u"].sortby(lev)
    v = ds["v"].sortby(lev)
    q = q.assign_coords({lev: q[lev].astype("float64") * 100.0})
    u = u.assign_coords({lev: u[lev].astype("float64") * 100.0})
    v = v.assign_coords({lev: v[lev].astype("float64") * 100.0})
    ivt_u = -(q * u).integrate(lev) / _G
    ivt_v = -(q * v).integrate(lev) / _G
    ivt = np.hypot(ivt_u, ivt_v)
    iwv = q.integrate(lev) / _G          # positive column water vapour
    return ivt, iwv


def plot(out_path: Path) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    apply_nature_style()
    fields = {w: _mean_fields(_GLOBAL_DIR / f"{w}_monthly_global.nc") for w in _WINS}
    ivt, iwv = {}, {}
    for w in _WINS:
        ivt[w], iwv[w] = _ivt_iwv(fields[w])

    a, b = "presat", "recent"
    d_total = ivt[b] - ivt[a]
    d_thermo = ivt[a] * (iwv[b] / iwv[a] - 1.0)
    d_dyn = d_total - d_thermo

    lon = ivt[a]["longitude"].to_numpy()
    lat = ivt[a]["latitude"].to_numpy()

    proj = ccrs.Robinson(central_longitude=200)
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.62))
    gs = fig.add_gridspec(2, 2, hspace=0.06, wspace=0.06)

    # US West Coast study box (config.BBOX, in -180..180 for drawing).
    bx0 = config.BBOX["lon_min"] - 360
    bx1 = config.BBOX["lon_max"] - 360
    by0, by1 = config.BBOX["lat_min"], config.BBOX["lat_max"]

    def _box(ax):
        ax.plot([bx0, bx1, bx1, bx0, bx0], [by0, by0, by1, by1, by0],
                color="#111", lw=0.9, transform=pc, zorder=6)

    def _base(ax):
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.25, edgecolor="0.3")
        ax.set_global()

    vmax_mean = float(np.nanpercentile(ivt[b].to_numpy(), 99))
    dmax = float(np.nanpercentile(np.abs(d_total.to_numpy()), 98))

    spec = [
        (ivt[b], "(a) Mean IVT, recent (2015–24)", "YlGnBu", 0, vmax_mean,
         "mean IVT (kg m$^{-1}$ s$^{-1}$)"),
        (d_total, "(b) Total change (recent − pre-sat)", "RdBu_r", -dmax, dmax,
         "ΔIVT (kg m$^{-1}$ s$^{-1}$)"),
        (d_thermo, "(c) Thermodynamic (moisture / CC)", "RdBu_r", -dmax, dmax, None),
        (d_dyn, "(d) Dynamic (circulation)", "RdBu_r", -dmax, dmax, None),
    ]
    for k, (field, title, cmap, vmin, vmax, clabel) in enumerate(spec):
        ax = fig.add_subplot(gs[k // 2, k % 2], projection=proj)
        _base(ax)
        pcm = ax.pcolormesh(lon, lat, field.to_numpy(), transform=pc,
                            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        _box(ax)
        ax.set_title(title, fontsize=7)
        cb = fig.colorbar(pcm, ax=ax, orientation="horizontal",
                          fraction=0.05, pad=0.03, shrink=0.85)
        cb.ax.tick_params(labelsize=5)
        if clabel:
            cb.set_label(clabel, fontsize=5.5)
        elif k == 1:
            cb.set_label("ΔIVT (kg m$^{-1}$ s$^{-1}$)", fontsize=5.5)

    # Area-weighted GLOBAL-MEAN change. The thermodynamic term is a coherent
    # moistening (its global mean is the net intensification); the dynamic term
    # mostly redistributes (its global mean ~ 0), so the share of the *mean*
    # increase is the honest statement, not the |magnitude| share.
    wgt = np.cos(np.deg2rad(lat))[:, None]
    wsum = float(wgt.sum() * lon.size)
    gm_total = float((d_total.to_numpy() * wgt).sum()) / wsum
    gm_thermo = float((d_thermo.to_numpy() * wgt).sum()) / wsum
    gm_dyn = float((d_dyn.to_numpy() * wgt).sum()) / wsum
    share = 100 * gm_thermo / gm_total if gm_total else float("nan")
    fig.suptitle(
        f"Global mean-flow IVT change {gm_total:+.1f} kg m$^{{-1}}$ s$^{{-1}}$: "
        f"thermodynamic {gm_thermo:+.1f} ({share:.0f}%), dynamic {gm_dyn:+.1f} "
        f"({100 - share:.0f}%)  ·  US West Coast box", fontsize=7.2, y=0.985)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    import json
    (config.CACHE_DIR / "global_drivers.json").write_text(json.dumps({
        "gm_total": round(gm_total, 1),
        "gm_thermo": round(gm_thermo, 1),
        "gm_dyn": round(gm_dyn, 1),
        "thermo_share": round(share),
    }, indent=2))
    print(f"wrote {out_path}")
    print(f"global-mean ΔIVT={gm_total:+.2f}, thermo={gm_thermo:+.2f} ({share:.0f}%), "
          f"dyn={gm_dyn:+.2f}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_global_drivers.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
