"""Compare the observed trend of the yearly wind contribution with the CMIP6 members.

The observed series is ERA5 on the ten CMIP6 standard levels from monthly fields
(fullcolumn_levels.json, key ``trends_full``); the member series come from
cmip6_members.json (key ``trend_full``). Both use the same yearly decomposition and
the same Theil-Sen slope over 1940-2024. For each corridor the observed slope is
compared with each single-model large ensemble (ensemble mean, standard deviation,
standardised departure, fraction of members with a slope at least as large as the
observed one, member range) and with the model-weighted distribution of all members,
in which every model carries the same total weight.

Usage: python -m src.analysis.cmip6_trend_test RESULTS_DIR
Output: RESULTS_DIR/cmip6_trend_test.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

LARGE = ("CanESM5", "MIROC6", "ACCESS-ESM1-5")
TERMS = ("wind", "moist", "total")
KEYS = ("slope_per_decade", "pct_per_decade")


def _wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(cw, q), len(x) - 1)])


def ensemble_stats(x, o):
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    frac = float(np.mean(x >= o)) if o >= mu else float(np.mean(x <= o))
    return {"n": int(x.size), "mean": round(mu, 3), "sd": round(sd, 3), "z": round((o - mu) / sd, 2),
            "frac_as_extreme": round(frac, 3), "min": round(float(x.min()), 3), "max": round(float(x.max()), 3)}


def main(res_dir: Path) -> dict:
    members = json.load(open(res_dir / "cmip6_members.json"))["members"]
    obs_all = json.load(open(res_dir / "fullcolumn_levels.json"))["trends_full"]
    keys = sorted(members)
    model = {k: k.split("/")[0] for k in keys}
    count = Counter(model.values())
    w = np.array([1.0 / count[model[k]] for k in keys])
    w /= w.sum()
    out = {"n_members": len(keys), "n_models": len(count), "large": list(LARGE)}
    for corr in obs_all:
        row = {}
        for term in TERMS:
            for key in KEYS:
                o = float(obs_all[corr][term][key])
                x = np.array([members[k][corr]["trend_full"][term][key] for k in keys])
                r = {"obs": round(o, 3), "obs_p": obs_all[corr][term]["p"]}
                for mdl in LARGE:
                    sel = np.array([model[k] == mdl for k in keys])
                    if sel.sum() >= 5:
                        r[mdl] = ensemble_stats(x[sel], o)
                mu = float((x * w).sum())
                frac = float(w[x >= o].sum()) if o >= mu else float(w[x <= o].sum())
                r["multi_model"] = {"weighted_mean": round(mu, 3), "p05": round(_wquantile(x, w, 0.05), 3),
                                    "p95": round(_wquantile(x, w, 0.95), 3), "frac_as_extreme": round(frac, 3),
                                    "n_members_as_extreme": int((x >= o).sum() if o >= mu else (x <= o).sum())}
                row[f"{term}_{key}"] = r
        out[corr] = row
    (res_dir / "cmip6_trend_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = main(Path(sys.argv[1]))
    for corr in [c for c in res if isinstance(res[c], dict) and "wind_slope_per_decade" in res[c]]:
        r = res[corr]["wind_slope_per_decade"]
        s = " | ".join(f"{m} {r[m]['mean']:+.2f}+-{r[m]['sd']:.2f} z {r[m]['z']:+.1f} f {r[m]['frac_as_extreme']:.2f} max {r[m]['max']:+.2f}" for m in LARGE if m in r)
        mm = r["multi_model"]
        print(f"{corr:18s} obs {r['obs']:+.2f} (p {r['obs_p']:.3f}) | {s} | multi mean {mm['weighted_mean']:+.2f} p95 {mm['p95']:+.2f} f {mm['frac_as_extreme']:.3f} n {mm['n_members_as_extreme']}")
