"""End-of-century decomposition for the single-member CMIP6 models under SSP5-8.5.

For each model in cmip6_corridors_proj (1995-2100), the level-resolved split is
computed for 1995-2014 against 2081-2100, per corridor, together with the
global-mean warming of the member between the two periods. Terms are also given
per kelvin of global warming, for comparison with the observed 1940-1969 to
1995-2024 change expressed the same way.

Output: results_esd/cmip6_projection.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from src.analysis import fullcolumn_decomp as fd
from src.figures.fig_global_ar_regions import _REGIONS

ROOT = Path("/p/projects/poem/fallah/nature_ar_data/cmip6_corridors_proj")
YEARS = np.arange(1995, 2101)
W0, W1 = (1995, 2014), (2081, 2100)


def main():
    ps = xr.open_dataset(fd.ERA5_DIR / "era5_sl_monthly_1940_2024.nc")["sp"].mean("valid_time").load()
    out = {}
    for path in sorted(ROOT.glob("*/*.nc")):
        model = path.parent.name
        try:
            ds = xr.open_dataset(path)
            tg = ds["tas_gm"].groupby("time.year").mean().to_series()
            dT = float(tg.loc[W1[0]:W1[1]].mean() - tg.loc[W0[0]:W0[1]].mean())
            res = {"_dT": round(dT, 3), "member": path.stem}
            for name, *_ in _REGIONS:
                c = fd.load_cmip_corridor(path, name, ps, years=YEARS)
                i0 = np.where((YEARS >= W0[0]) & (YEARS <= W0[1]))[0]
                i1 = np.where((YEARS >= W1[0]) & (YEARS <= W1[1]))[0]
                r = fd.reduce_months(fd.window_terms(c, i0, i1, cc_split=False))
                r.pop("vhat")
                res[name] = {k: round(v, 4) for k, v in r.items()}
                res[name]["per_K"] = {k: round(r[k] / dT, 4) for k in ("moist", "wind", "cov", "trans", "total")}
            out[model] = res
            print(model, "dT", round(dT, 2), "| wind per K:", {n.split()[0][:6]: res[n]["per_K"]["wind"] for n, *_ in _REGIONS[:4]}, flush=True)
        except Exception as e:
            print("FAILED", model, type(e).__name__, e, flush=True)
    p = fd.OUT_DIR / "cmip6_projection.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p, "models:", len(out))


if __name__ == "__main__":
    main()
