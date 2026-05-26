"""
Project-wide configuration.

All spatial, vertical, and dataset constants live here so they can be imported
from feature builders, model pipelines, and notebooks without drift.
"""

from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Filesystem layout. Raw ERA5/CMIP6 never lands here — only derived artifacts:
# climatologies, GW masks, training Parquet, trained boosters, event records.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

# -----------------------------------------------------------------------------
# Training / holdout / inference windows.
# -----------------------------------------------------------------------------
TRAIN_PERIOD = ("1980-01-01", "2014-12-31")
HOLDOUT_PERIOD = ("2015-01-01", "2024-12-31")
CMIP6_HIST_PERIOD = ("1980-01-01", "2014-12-31")
CMIP6_FUT_PERIOD = ("2070-01-01", "2099-12-31")

# Sub-daily cadence at which Guan-Waliser is canonically applied and at which
# we sub-sample ARCO-ERA5 for training.
TRAINING_CADENCE_HOURS = 6

# -----------------------------------------------------------------------------
# Regional bounding box — US West Coast atmospheric-river landfall corridor.
# Longitudes are in the 0..360 convention to match ARCO-ERA5 / Pangeo CMIP6.
# -----------------------------------------------------------------------------
BBOX = {
    "lat_min": 25.0,
    "lat_max": 60.0,
    "lon_min": 210.0,
    "lon_max": 250.0,
}

# -----------------------------------------------------------------------------
# Target pressure levels (hPa).
#   850 hPa — lower-troposphere moisture and theta_e
#   500 hPa — mid-troposphere steering flow / static stability anchor
#   250 hPa — upper-troposphere jet, QG PV, baroclinic shear
# -----------------------------------------------------------------------------
PRESSURE_LEVELS = [850, 500, 250]

# -----------------------------------------------------------------------------
# Public ARCO Zarr stores (anonymous read, no auth required).
# ARCO-ERA5: Google Public Datasets — Pangeo / Google Research mirror.
# CMIP6: Pangeo cloud catalog (JSON manifest of ESGF-style queryable Zarrs).
# -----------------------------------------------------------------------------
ERA5_ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

CMIP6_CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"

# Default SSP5-8.5 query against the Pangeo CMIP6 catalog. Resolved at Phase 3.
CMIP6_QUERY = {
    "activity_id": "ScenarioMIP",
    "experiment_id": "ssp585",
    "table_id": "6hrLev",
    "variable_id": ["ua", "va", "ta", "hus", "ps"],
    "source_id": "MPI-ESM1-2-HR",
    "member_id": "r1i1p1f1",
}

# -----------------------------------------------------------------------------
# Dask chunking hints. Time is the parallel axis; spatial dims stay whole so
# MetPy spherical operators don't see partial neighborhoods.
# -----------------------------------------------------------------------------
DEFAULT_CHUNKS = {
    "time": 240,  # ~10 days at hourly cadence
    "level": -1,  # all pressure levels together (needed for integrals)
    "latitude": -1,
    "longitude": -1,
}

# -----------------------------------------------------------------------------
# Variable-name map for ARCO-ERA5 (CF long names used in the store).
# -----------------------------------------------------------------------------
ERA5_VARS = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
    "q": "specific_humidity",
    "z": "geopotential",
    "sp": "surface_pressure",
}

# Precipitation, surface geopotential, and land-sea mask (Stage 2 / topography).
ERA5_SURFACE_VARS = {
    "tp": "total_precipitation",
    "z_sfc": "geopotential_at_surface",
    "lsm": "land_sea_mask",
}

# -----------------------------------------------------------------------------
# Guan & Waliser (2015) AR detection — algorithm thresholds.
# Ref.: Guan, B. & D. E. Waliser (2015), J. Geophys. Res. Atmos., 120, 12514.
# -----------------------------------------------------------------------------
GW_IVT_FLOOR = 250.0  # kg m^-1 s^-1   minimum threshold ceiling
GW_CLIM_QUANTILE = 0.85  #                 percentile of per-pixel monthly climatology
GW_MIN_LENGTH_KM = 2000.0  # km              great-circle Feret diameter
GW_MIN_LW_RATIO = 2.0  #                 length / width
GW_AXIS_TOL_DEG = 45.0  # deg             tolerated angle between PCA major axis and mean IVT vector

# -----------------------------------------------------------------------------
# Event post-processing & ML targets.
# -----------------------------------------------------------------------------
STAGE1_EVENT_THRESHOLD = 100.0  # kg m^-1 s^-1   threshold on predicted intensity field
RANDOM_SEED = 42
