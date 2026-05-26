"""
Guan & Waliser (2015) Atmospheric River detection.

Reference
---------
Guan, B. & D. E. Waliser (2015). Detection of atmospheric rivers: Evaluation
and application of an algorithm for global studies. *J. Geophys. Res. Atmos.*,
120, 12514–12535. doi:10.1002/2015JD024257.

Algorithm
---------
For each 6-hourly time step:

1. **Intensity threshold.** A pixel is flagged candidate if its IVT exceeds
   ``max(GW_IVT_FLOOR, climatology_85th_pctile(month))`` — a hybrid absolute /
   relative threshold (GW §3.2). The relative branch lets the detector pick up
   sub-tropical events that never reach 250 kg m⁻¹ s⁻¹.

2. **Connected components.** 2-D 8-connectivity labelling via
   ``scipy.ndimage.label`` on each time slice.

3. **Geometry filter.** Each labeled object is kept only if:

   * **length** (great-circle Feret diameter of the convex hull) ≥ 2000 km
   * **length / width** ≥ 2, where width = area / length
   * the principal direction of the object's pixel cloud is within
     ``GW_AXIS_TOL_DEG`` of the mean (ivt_u, ivt_v) vector — i.e. the AR
     points roughly along its moisture transport.

Engineering notes
-----------------
* The geometry filter is wrapped in :func:`xarray.apply_ufunc` with
  ``dask="parallelized"`` so the whole pipeline stays lazy. Each call sees one
  ``(lat, lon)`` slice; the time dim is the parallel axis.
* The climatology is the only one-shot ``.compute()`` in the module. It is
  written to ``CACHE_DIR / "ivt_climatology.zarr"`` and re-used across runs.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from scipy.spatial import ConvexHull, QhullError

from src import config

# Earth-surface scale: 1° latitude ≈ 111.0 km. Longitude shrinks by cos(lat).
_KM_PER_DEG_LAT = 111.0


# =============================================================================
# Climatology (per-pixel monthly 85th percentile)
# =============================================================================


def compute_ivt_climatology(
    ivt: xr.DataArray,
    period: tuple[str, str] = config.TRAIN_PERIOD,
    quantile: float = config.GW_CLIM_QUANTILE,
) -> xr.DataArray:
    """Per-pixel monthly 85th percentile of IVT over the training window.

    Returns a DataArray with dims ``(month, lat, lon)`` and units of
    ``kg m⁻¹ s⁻¹``. Materialises and caches at
    ``CACHE_DIR / "ivt_climatology_<period>.zarr"``.
    """
    cache_path = (
        config.CACHE_DIR
        / f"ivt_climatology_{period[0][:4]}_{period[1][:4]}_q{int(quantile * 100):02d}.zarr"
    )
    if cache_path.exists():
        return xr.open_zarr(cache_path)["ivt_climatology"]

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    window = ivt.sel(time=slice(*period))
    # quantile() needs the reduce dim un-chunked.
    clim = window.chunk({"time": -1}).groupby("time.month").quantile(q=quantile, dim="time")
    clim = clim.astype("float32").rename("ivt_climatology")
    clim.to_dataset().to_zarr(cache_path, mode="w", consolidated=True)
    return xr.open_zarr(cache_path)["ivt_climatology"]


# =============================================================================
# Per-timestep geometry filter (the pure NumPy kernel)
# =============================================================================


def _filter_one_timestep(
    candidate: np.ndarray,  # (lat, lon) bool
    ivt_u: np.ndarray,  # (lat, lon) float
    ivt_v: np.ndarray,  # (lat, lon) float
    lat_1d: np.ndarray,  # (lat,)
    lon_1d: np.ndarray,  # (lon,)
) -> np.ndarray:
    """Apply the GW geometry filter to a single time slice. Returns a bool
    mask of the same shape with non-AR objects removed."""
    if not candidate.any():
        return candidate.copy()

    labels, n_obj = ndi.label(candidate, structure=np.ones((3, 3)))
    out = np.zeros_like(candidate)

    cos_tol = np.cos(np.deg2rad(config.GW_AXIS_TOL_DEG))

    for label_id in range(1, n_obj + 1):
        obj = labels == label_id
        if obj.sum() < 5:
            # Too small to host a coherent axis. Skip without invoking ConvexHull.
            continue

        i_idx, j_idx = np.where(obj)
        lats = lat_1d[i_idx]
        lons = lon_1d[j_idx]

        # Planar approximation centred on the object centroid. Adequate at
        # mid-latitudes for the ≤ 2-3000 km objects we care about; we are
        # ranking with hard thresholds, not doing geodesy.
        lat_c = float(lats.mean())
        lon_c = float(lons.mean())
        dy_km = (lats - lat_c) * _KM_PER_DEG_LAT
        dx_km = (lons - lon_c) * _KM_PER_DEG_LAT * np.cos(np.deg2rad(lat_c))
        pts = np.column_stack([dx_km, dy_km])

        try:
            hull = ConvexHull(pts)
        except QhullError:
            # Co-linear points — also too degenerate to be an AR.
            continue

        # Feret diameter: max distance between hull vertices. Vertex count is
        # tiny so the O(V^2) is cheap.
        hv = pts[hull.vertices]
        diffs = hv[:, None, :] - hv[None, :, :]
        length_km = float(np.sqrt((diffs * diffs).sum(axis=-1)).max())
        if length_km < config.GW_MIN_LENGTH_KM:
            continue

        # Width via the GW area-over-length definition.
        area_km2 = float(hull.volume)  # 2-D hull volume == area
        width_km = area_km2 / max(length_km, 1.0)
        if length_km / max(width_km, 1.0) < config.GW_MIN_LW_RATIO:
            continue

        # Principal direction of the pixel cloud (PCA on 2-D positions).
        # eigh returns in ascending order; the last column is the principal axis.
        _, eig_vecs = np.linalg.eigh(np.cov(pts.T))
        major = eig_vecs[:, -1]

        # Mean IVT vector inside the object — the moisture transport direction.
        u_mean = float(ivt_u[obj].mean())
        v_mean = float(ivt_v[obj].mean())
        ivt_mag = np.hypot(u_mean, v_mean)
        if ivt_mag < 1e-6:
            continue
        ivt_dir = np.array([u_mean, v_mean]) / ivt_mag

        # PCA direction is bidirectional; take |·| to compare orientations.
        if abs(major[0] * ivt_dir[0] + major[1] * ivt_dir[1]) < cos_tol:
            continue

        out |= obj

    return out


# =============================================================================
# Lazy public entry point
# =============================================================================


def compute_ar_mask(
    ivt: xr.DataArray,
    ivt_u: xr.DataArray,
    ivt_v: xr.DataArray,
    climatology: xr.DataArray,
) -> xr.DataArray:
    """Lazy Dask-backed Guan-Waliser AR mask aligned to ``ivt``'s time index.

    All inputs must share ``(time, lat, lon)`` dimensions (the lat/lon dim
    names are auto-detected). Returns a ``bool`` DataArray.
    """
    lat_name = "latitude" if "latitude" in ivt.coords else "lat"
    lon_name = "longitude" if "longitude" in ivt.coords else "lon"

    month = ivt["time"].dt.month
    pixel_thresh = climatology.sel(month=month).where(
        climatology.sel(month=month) > config.GW_IVT_FLOOR, config.GW_IVT_FLOOR
    )
    candidate = ivt > pixel_thresh

    lat_1d = ivt[lat_name].values
    lon_1d = ivt[lon_name].values

    mask = xr.apply_ufunc(
        _filter_one_timestep,
        candidate,
        ivt_u,
        ivt_v,
        input_core_dims=[[lat_name, lon_name]] * 3,
        output_core_dims=[[lat_name, lon_name]],
        kwargs={"lat_1d": lat_1d, "lon_1d": lon_1d},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
    )
    return mask.rename("ar_mask")
