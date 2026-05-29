"""
Figure 5 — Where the AR landfall corridor sits, along the Story-A trajectory.

Seven cartographic panels in a 2x4 grid (one cell empty), each a Gaussian
KDE of per-event landfall coordinates (``landfall_lon``, ``landfall_lat``)
over a US-West-Coast Plate-Carrée basemap.

Row 1 (observed):  Pre-sat | Modern | Recent | (empty)
Row 2 (projected): SSP2-4.5 | SSP3-7.0 | SSP4-6.0 | SSP5-8.5

The figure reveals whether the *position* of the landfall corridor migrates
under warming — complementing fig3 (count / intensity along the trajectory)
and fig4 (driver bucket mix). Same colour ramp on every panel; the same
event count is plotted at the same colour intensity so panels are visually
comparable.

Inputs
------
``data/cache/observational_events.parquet`` — written by
:mod:`src.analysis.period_contrast`. Must carry ``period``,
``landfall_lat``, ``landfall_lon``.

``data/cache/cmip6_events_<source>_<ssp>_direct.parquet`` — written by
:mod:`src.inference.cmip6_apply` (``--mode direct``). Headline protocol;
delta mode goes in the supplement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src import config
from src.figures._style import COL_DOUBLE_IN, apply_nature_style

_OBS_ORDER = ("pre_sat_1940_1959", "modern_1980_1999", "recent_2015_2024")
_OBS_LABEL = {
    "pre_sat_1940_1959": "Pre-sat (1940-59)",
    "modern_1980_1999": "Modern (1980-99)",
    "recent_2015_2024": "Recent (2015-24)",
}
_SSP_ORDER = ("ssp245", "ssp370", "ssp460", "ssp585")
_SSP_LABEL = {
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp460": "SSP4-6.0",
    "ssp585": "SSP5-8.5",
}


def _draw_basemap(ax: plt.Axes) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    lat_min, lat_max = config.BBOX["lat_min"], config.BBOX["lat_max"]
    lon_min, lon_max = config.BBOX["lon_min"], config.BBOX["lon_max"]
    ax.set_extent([lon_min - 360, lon_max - 360, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4, edgecolor="black")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="0.4")
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.2, edgecolor="0.55")
    gl = ax.gridlines(draw_labels=False, linewidth=0.2, color="0.7", alpha=0.5, linestyle=":")
    gl.xlocator = plt.matplotlib.ticker.FixedLocator([-150, -130, -110])
    gl.ylocator = plt.matplotlib.ticker.FixedLocator([30, 40, 50])


def _density(
    ax: plt.Axes,
    events: pd.DataFrame,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    vmax: float,
) -> None:
    """Gaussian-KDE per-event landfall density on the supplied lon/lat grid."""
    df = events.dropna(subset=["landfall_lat", "landfall_lon"])
    if len(df) < 5:
        ax.text(
            0.5,
            0.5,
            "n<5",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6,
            color="0.4",
        )
        return

    lon = df["landfall_lon"].to_numpy(dtype="float64")
    # Cartopy expects -180..180 longitudes; ERA5 events are stored 0..360.
    lon = np.where(lon > 180, lon - 360, lon)
    lat = df["landfall_lat"].to_numpy(dtype="float64")
    sample = np.vstack([lon, lat])

    import cartopy.crs as ccrs

    kde = gaussian_kde(sample, bw_method=0.25)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
    z = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    # Scale to events/year for an intuitive colour ramp (KDE integrates to 1).
    z *= len(df)

    ax.pcolormesh(
        grid_x,
        grid_y,
        z,
        transform=ccrs.PlateCarree(),
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        shading="auto",
    )


def plot(
    observed: dict[str, pd.DataFrame],
    projected: dict[str, pd.DataFrame],
    observed_only: bool = False,
) -> plt.Figure:
    """Landfall density figure. Default: 7 panels (3 observed + 4 projected).

    ``observed_only=True`` renders just the three observational windows in a
    single row — the ERA5-only manuscript form, where the projected CMIP6
    panels would be synthetic and must not appear.
    """
    import cartopy.crs as ccrs

    apply_nature_style()
    if observed_only:
        projected = {}
        nrows, ncols = 1, 3
        fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.40))
    else:
        nrows, ncols = 2, 4
        fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.55))
    proj = ccrs.PlateCarree()

    lat_min, lat_max = config.BBOX["lat_min"], config.BBOX["lat_max"]
    lon_min, lon_max = config.BBOX["lon_min"], config.BBOX["lon_max"]
    grid_lon = np.linspace(lon_min - 360, lon_max - 360, 81)
    grid_lat = np.linspace(lat_min, lat_max, 71)

    # Common colour ceiling across every panel so densities are comparable.
    counts = [len(df) for df in (*observed.values(), *projected.values()) if df is not None]
    vmax = max(counts) / 50.0 if counts else 1.0

    panels = [
        (i, _OBS_LABEL.get(p, p), observed.get(p)) for i, p in enumerate(_OBS_ORDER, start=1)
    ]
    if not observed_only:
        panels += [(5 + i, _SSP_LABEL[s], projected.get(s)) for i, s in enumerate(_SSP_ORDER)]

    for idx, label, df in panels:
        ax = fig.add_subplot(nrows, ncols, idx, projection=proj)
        _draw_basemap(ax)
        if df is not None:
            _density(ax, df, grid_lon, grid_lat, vmax)
        ax.set_title(label, fontsize=6, pad=2.0)

    # Shared colour bar across all panels.
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.70])
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.matplotlib.colors.Normalize(0, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("landfall event density (a.u.)", fontsize=6)
    cb.ax.tick_params(labelsize=5)

    title = (
        "Landfall corridor across three observational windows" if observed_only
        else "Landfall corridor along the observed-then-projected trajectory"
    )
    fig.suptitle(title, fontsize=7)
    fig.subplots_adjust(left=0.02, right=0.90, top=0.92, bottom=0.05, hspace=0.20, wspace=0.05)
    return fig


def _split_observational(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {p: events.loc[events["period"] == p].reset_index(drop=True) for p in _OBS_ORDER}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observed",
        type=str,
        default=str(config.CACHE_DIR / "observational_events.parquet"),
    )
    parser.add_argument("--source", type=str, default=config.CMIP6_QUERY_BASE["source_id"])
    parser.add_argument("--mode", type=str, default="direct", choices=("direct", "delta"))
    parser.add_argument("--out", type=str, default="figures/fig5_landfall_density.pdf")
    parser.add_argument(
        "--observed-only",
        action="store_true",
        help="Render only the three observational windows (ERA5-only manuscript); "
        "omit the synthetic projected CMIP6 panels.",
    )
    args = parser.parse_args()

    observed_events = pd.read_parquet(args.observed)
    observed = _split_observational(observed_events)

    projected = {}
    if not args.observed_only:
        for ssp in _SSP_ORDER:
            path = config.CACHE_DIR / f"cmip6_events_{args.source}_{ssp}_{args.mode}.parquet"
            if path.exists():
                projected[ssp] = pd.read_parquet(path)

    fig = plot(observed, projected, observed_only=args.observed_only)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
