"""Autocorrelation-corrected trend statistics.

Corridor moisture-transport series are strongly serially correlated, so the
ordinary Mann-Kendall (MK) test is anticonservative: its variance assumes
independent observations and the inflation of the effective sample size makes
weak trends look highly significant.  We use the Hamed & Rao (1998) variance
correction, which rescales Var(S) by the lag autocorrelation structure of the
detrended ranks, together with the Theil-Sen slope estimator.  This is the test
the manuscript reports (e.g. p ~ 0.05 for the US West Coast trajectory, where the
plain MK gives p ~ 1e-3).

Reference: Hamed, K. H. & Rao, A. R. A modified Mann-Kendall trend test for
autocorrelated data. J. Hydrol. 204, 182-196 (1998).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, rankdata


@dataclass
class TrendResult:
    slope: float          # Theil-Sen slope (units per step)
    intercept: float
    tau: float            # Kendall's tau
    z: float              # corrected standard normal statistic
    p: float              # two-sided p-value (Hamed-Rao corrected)
    p_plain: float        # two-sided p-value, uncorrected MK
    n_eff_ratio: float    # n / n* variance-correction factor (>1 means inflation)


def _mk_S_var(x: np.ndarray) -> tuple[float, float, float]:
    n = len(x)
    s = 0.0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1:] - x[k]))
    # tie-corrected variance of S
    _, counts = np.unique(x, return_counts=True)
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    denom = 0.5 * n * (n - 1)
    tau = s / denom if denom else 0.0
    return s, var, tau


def _theil_sen(t: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    from scipy.stats import theilslopes
    slope, intercept, _, _ = theilslopes(x, t)
    return float(slope), float(intercept)


def hamed_rao_mk(x: np.ndarray, t: np.ndarray | None = None) -> TrendResult:
    """Modified Mann-Kendall trend test with Hamed-Rao variance correction."""
    x = np.asarray(x, float)
    n = len(x)
    if t is None:
        t = np.arange(n, dtype=float)
    t = np.asarray(t, float)

    s, var0, tau = _mk_S_var(x)
    slope, intercept = _theil_sen(t, x)

    # Detrend with the Theil-Sen slope, rank, compute significant lag autocorr.
    detr = x - slope * (t - t.mean())
    ranks = rankdata(detr)
    ranks = ranks - ranks.mean()
    # autocorrelation of the ranks at lags 1..n-1
    acf = np.array([
        np.sum(ranks[:n - k] * ranks[k:]) / np.sum(ranks ** 2)
        for k in range(1, n)
    ])
    # keep only lags whose autocorrelation is significant at 95%
    crit = 1.96 / np.sqrt(n)
    sig = np.abs(acf) > crit
    corr = 0.0
    for k in range(1, n):
        rk = acf[k - 1]
        if sig[k - 1]:
            corr += (n - k) * (n - k - 1) * (n - k - 2) * rk
    n_eff_ratio = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * corr
    n_eff_ratio = max(n_eff_ratio, 1e-6)
    var_corr = var0 * n_eff_ratio

    def _z_p(var):
        if var <= 0:
            return 0.0, 1.0
        if s > 0:
            z = (s - 1) / np.sqrt(var)
        elif s < 0:
            z = (s + 1) / np.sqrt(var)
        else:
            z = 0.0
        return z, 2 * (1 - norm.cdf(abs(z)))

    z, p = _z_p(var_corr)
    _, p_plain = _z_p(var0)
    return TrendResult(slope=slope, intercept=intercept, tau=tau, z=z, p=p,
                       p_plain=p_plain, n_eff_ratio=n_eff_ratio)
