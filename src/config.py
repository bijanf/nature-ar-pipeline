"""
Project-wide configuration.

All spatial, vertical, and dataset constants live here so they can be imported
from feature builders, model pipelines, and notebooks without drift.
"""

from __future__ import annotations

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
