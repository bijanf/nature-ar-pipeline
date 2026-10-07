"""Supplement data: bootstrap replicate distributions of the wind term and the
block-length comparison, for the headline period pair.

Writes results_esd/bootstrap_wind.npz with, per corridor, the 1000 replicates of
the wind-change term for block lengths 1, 3 and 5 years, and the point estimate.

Usage: python -m src.analysis.bootstrap_distributions
"""

from __future__ import annotations

import numpy as np

from src.analysis import fullcolumn_decomp as fd
from src.figures.fig_global_ar_regions import _REGIONS


def main():
    w0, w1 = fd.HEADLINE
    i0 = np.where((fd.YEARS >= w0[0]) & (fd.YEARS <= w0[1]))[0]
    i1 = np.where((fd.YEARS >= w1[0]) & (fd.YEARS <= w1[1]))[0]
    out = {}
    for name, *_ in _REGIONS:
        c = fd.load_era5_corridor(name)
        k = fd.corridor_key(name)
        pt = fd.reduce_months(fd.window_terms(c, i0, i1))
        out[f"{k}__point"] = np.array([pt["wind"], pt["moist"], pt["trans"]])
        for b in (1, 3, 5):
            keys, reps = fd.bootstrap(c, i0, i1, n_boot=1000, block=b)
            out[f"{k}__b{b}"] = reps[:, [keys.index("wind"), keys.index("moist"), keys.index("trans")]]
        print(name, "done", flush=True)
    np.savez(fd.OUT_DIR / "bootstrap_wind.npz", **out)
    print("wrote", fd.OUT_DIR / "bootstrap_wind.npz")


if __name__ == "__main__":
    main()
