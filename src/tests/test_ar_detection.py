"""Synthetic-field unit tests for the Guan-Waliser geometry filter.

These tests build small NumPy arrays representing canonical AR vs. non-AR
shapes and assert that ``_filter_one_timestep`` keeps the AR-shaped objects
and rejects the rest. They are network-free and run in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.ar_detection import _filter_one_timestep

# 0.25° grid covering the project bbox (25–60 N, 210–250 E).
LATS = np.arange(25.0, 60.25, 0.25)
LONS = np.arange(210.0, 250.25, 0.25)


def _empty_field() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (LATS.size, LONS.size)
    return (
        np.zeros(shape, dtype=bool),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
    )


def test_elongated_ar_passes() -> None:
    # A 30°-long, 2°-wide diagonal band — > 3000 km long, length/width >> 2,
    # aligned with a south-westerly IVT vector. Should pass all GW criteria.
    mask, ivt_u, ivt_v = _empty_field()
    for k in range(120):
        i = 30 + k // 2
        j = 20 + k
        if i < mask.shape[0] and j < mask.shape[1]:
            mask[i - 2 : i + 2, j - 4 : j + 4] = True
    # IVT vector points to the north-east, parallel to the band's long axis.
    ivt_u[mask] = 200.0
    ivt_v[mask] = 200.0

    out = _filter_one_timestep(mask, ivt_u, ivt_v, LATS, LONS)
    assert out.any()
    assert out.sum() >= mask.sum() * 0.95  # almost all pixels retained


def test_circular_blob_rejected() -> None:
    # A near-circular blob with length/width ≈ 1. Should be rejected.
    mask, ivt_u, ivt_v = _empty_field()
    cy, cx = 60, 60
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 < 12**2] = True
    ivt_u[mask] = 300.0
    ivt_v[mask] = 50.0

    out = _filter_one_timestep(mask, ivt_u, ivt_v, LATS, LONS)
    assert not out.any()


def test_axis_misaligned_rejected() -> None:
    # NW-SE oriented band but IVT pointing due-north: the angle between the
    # band's major axis and the IVT vector exceeds GW_AXIS_TOL_DEG.
    mask, ivt_u, ivt_v = _empty_field()
    for k in range(80):
        i = 80 - k // 2
        j = 20 + k
        if 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1]:
            mask[i - 1 : i + 1, j - 2 : j + 2] = True
    ivt_v[mask] = 300.0  # purely meridional, ~perpendicular to the band

    out = _filter_one_timestep(mask, ivt_u, ivt_v, LATS, LONS)
    assert not out.any()


def test_too_short_rejected() -> None:
    # A correctly-shaped but short (~500 km) band. Should fail length test.
    mask, ivt_u, ivt_v = _empty_field()
    for k in range(20):
        i = 50 + k // 4
        j = 50 + k
        mask[i, j - 1 : j + 1] = True
    ivt_u[mask] = 200.0
    ivt_v[mask] = 100.0

    out = _filter_one_timestep(mask, ivt_u, ivt_v, LATS, LONS)
    assert not out.any()


def test_empty_field_returns_empty() -> None:
    mask, ivt_u, ivt_v = _empty_field()
    out = _filter_one_timestep(mask, ivt_u, ivt_v, LATS, LONS)
    assert out.shape == mask.shape
    assert not out.any()


@pytest.mark.parametrize("dtype", [bool, np.uint8])
def test_dtype_preserved(dtype: type) -> None:
    mask, ivt_u, ivt_v = _empty_field()
    mask = mask.astype(dtype)
    out = _filter_one_timestep(mask, ivt_u, ivt_v, LATS, LONS)
    assert out.shape == mask.shape


# --- fixed-threshold sensitivity test (constant_climatology) ----------------

import xarray as xr  # noqa: E402

from src.features.ar_detection import compute_ar_mask, constant_climatology  # noqa: E402


def _band_ivt() -> xr.Dataset:
    """A single-timestep IVT field with one elongated AR band (|IVT|=400)
    on a quiescent 100 background, IVT vector aligned to the band's long axis."""
    shape = (LATS.size, LONS.size)
    ivt = np.full(shape, 100.0, dtype=np.float32)
    u = np.zeros(shape, dtype=np.float32)
    v = np.zeros(shape, dtype=np.float32)
    for k in range(120):
        i = 30 + k // 2
        j = 20 + k
        if i < shape[0] and j < shape[1]:
            ivt[i - 2 : i + 2, j - 4 : j + 4] = 400.0
            u[i - 2 : i + 2, j - 4 : j + 4] = 200.0
            v[i - 2 : i + 2, j - 4 : j + 4] = 200.0
    time = np.array(["2020-01-15"], dtype="datetime64[ns]")
    coords = {"time": time, "latitude": LATS, "longitude": LONS}

    def mk(a: np.ndarray) -> xr.DataArray:
        return xr.DataArray(a[None], dims=("time", "latitude", "longitude"), coords=coords)

    return xr.Dataset({"ivt": mk(ivt), "ivt_u": mk(u), "ivt_v": mk(v)})


def test_constant_climatology_shape_and_value() -> None:
    ds = _band_ivt()
    clim = constant_climatology(ds["ivt"], 250.0)
    assert clim.dims == ("month", "latitude", "longitude")
    assert clim.sizes["month"] == 12
    assert float(clim.min()) == 250.0 and float(clim.max()) == 250.0


def test_fixed_threshold_below_band_detects() -> None:
    # Uniform threshold 250 < band's 400 -> the band is a candidate and,
    # being AR-shaped, survives the geometry filter.
    ds = _band_ivt()
    clim = constant_climatology(ds["ivt"], 250.0)
    mask = compute_ar_mask(ds["ivt"], ds["ivt_u"], ds["ivt_v"], clim).compute()
    assert bool(mask.any())


def test_fixed_threshold_above_band_detects_nothing() -> None:
    # Uniform threshold 500 > band's 400 -> no pixel exceeds it, mask is empty.
    ds = _band_ivt()
    clim = constant_climatology(ds["ivt"], 500.0)
    mask = compute_ar_mask(ds["ivt"], ds["ivt_u"], ds["ivt_v"], clim).compute()
    assert not bool(mask.any())
