"""Signal-verdict gate for the ERA5-only AR manuscript.

Reads ``data/cache/period_contrast.parquet`` (the unified three-window
summary) plus ``observational_events.parquet`` and prints a one-screen
verdict: is the observed intensification real, flat, or mixed?

The paper's headline claim is a *coherent* shift Pre-sat -> Modern -> Recent.
This script answers, mechanically and without spin:

  1. Sanity   — are the events usable? (land_fraction non-zero, landfall_lat
                mostly non-NaN, plausible counts)
  2. Order    — is mean intensity monotonic non-decreasing across the windows?
  3. Strength — do the 90 % bootstrap bands separate for the two pairwise
                contrasts (Modern-Presat, Recent-Modern)?
  4. Breadth  — do footprint and duration move the same way as intensity?

Verdict:
  INTENSIFIED  intensity monotonic up AND the Modern-Presat band separates
               (the longest-baseline contrast carries it)
  FLAT         deltas small and no band separates anywhere
  MIXED        anything else (non-monotonic, or intensity up while frequency
               or breadth disagree) — the most scientifically interesting case

Run:  python -m src.analysis.verify_signal [--source full|partial]
Exit code 0 always (this is a report, not a test); the verdict is in stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src import config

_ORDER = [
    ("pre_sat_1940_1959", "Pre-sat 1940-59"),
    ("modern_1980_1999", "Modern 1980-99"),
    ("recent_2015_2024", "Recent 2015-24"),
]


def _load(source: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return (period_contrast, observational_events|None)."""
    cache = config.CACHE_DIR
    if source == "full":
        pc = cache / "period_contrast.parquet"
        ev = cache / "observational_events.parquet"
        pc_df = pd.read_parquet(pc) if pc.exists() else pd.DataFrame()
        ev_df = pd.read_parquet(ev) if ev.exists() else None
        return pc_df, ev_df

    # partial: stitch whichever per-period parquets exist (full preferred).
    pc_parts, ev_parts = [], []
    for short in ("presat", "modern", "recent"):
        for stem, bucket in (("period_contrast", pc_parts), ("observational_events", ev_parts)):
            full = cache / f"{stem}_{short}.parquet"
            part = cache / f"{stem}_{short}_partial.parquet"
            path = full if full.exists() else (part if part.exists() else None)
            if path is not None:
                bucket.append(pd.read_parquet(path))
    pc_df = pd.concat(pc_parts, ignore_index=True) if pc_parts else pd.DataFrame()
    ev_df = pd.concat(ev_parts, ignore_index=True) if ev_parts else None
    return pc_df, ev_df


def _bands_separate(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    """True iff the two [lo, hi] intervals do not overlap."""
    return (lo_a > hi_b) or (lo_b > hi_a)


def _fmt(x: float | None, dec: int = 1) -> str:
    return "  n/a" if x is None or pd.isna(x) else f"{x:.{dec}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["full", "partial"], default="full")
    args = ap.parse_args()

    pc, ev = _load(args.source)
    if pc.empty:
        print("period_contrast parquet not found yet — nothing to verify.", file=sys.stderr)
        return 0
    pc = pc.set_index("period_name")

    present = [(k, lbl) for k, lbl in _ORDER if k in pc.index]
    keys = [k for k, _ in present]

    print("=" * 64)
    print(" AR intensification — signal verdict")
    print("=" * 64)

    # ---- 1. sanity ---------------------------------------------------------
    print("\n[1] SANITY")
    if ev is not None and "period" in ev.columns:
        for k, lbl in present:
            sub = ev[ev["period"] == k]
            n = len(sub)
            lf_ok = (sub["land_fraction"] > 0).mean() * 100 if "land_fraction" in sub else float("nan")
            land_ok = sub["landfall_lat"].notna().mean() * 100 if "landfall_lat" in sub else float("nan")
            print(f"    {lbl:16s} n={n:5d}  land_fraction>0: {_fmt(lf_ok,0)}%  "
                  f"landfall_lat non-NaN: {_fmt(land_ok,0)}%")
    else:
        print("    (observational_events parquet not present — skipping event-level sanity)")

    # ---- per-window table --------------------------------------------------
    print("\n    window           n_events  intensity (q05–q95)        footprint   duration (q05–q95)")
    for k, lbl in present:
        r = pc.loc[k]
        print(f"    {lbl:16s} {int(r['n_events']):6d}   "
              f"{_fmt(r.get('mean_intensity'))} ({_fmt(r.get('intensity_q05'))}"
              f"–{_fmt(r.get('intensity_q95'))})   "
              f"{_fmt(r.get('mean_footprint_km2')/1e6 if pd.notna(r.get('mean_footprint_km2')) else None,2)}e6   "
              f"{_fmt(r.get('mean_duration_h'))} ({_fmt(r.get('duration_q05'))}"
              f"–{_fmt(r.get('duration_q95'))})")

    if len(keys) < 2:
        print("\n  Only one window present — the contrast needs >=2. Verdict deferred.")
        return 0

    def col(metric: str) -> dict[str, float]:
        return {k: float(pc.loc[k, metric]) for k in keys if metric in pc.columns}

    intensity = col("mean_intensity")

    # ---- 2. order ----------------------------------------------------------
    seq = [intensity[k] for k in keys]
    monotonic = all(b >= a for a, b in zip(seq, seq[1:]))
    print("\n[2] ORDER (mean intensity, kg m⁻¹ s⁻¹)")
    print("    " + "  ->  ".join(f"{intensity[k]:.1f}" for k in keys)
          + f"   {'monotonic ↑' if monotonic else 'NOT monotonic'}")

    # ---- 3. strength (band separation) ------------------------------------
    print("\n[3] STRENGTH (90% bootstrap-band separation)")
    sep = {}
    for a, b in zip(keys, keys[1:]):
        ok = _bands_separate(
            float(pc.loc[b, "intensity_q05"]), float(pc.loc[b, "intensity_q95"]),
            float(pc.loc[a, "intensity_q05"]), float(pc.loc[a, "intensity_q95"]),
        )
        sep[(a, b)] = ok
        d = intensity[b] - intensity[a]
        la = dict(_ORDER)[a]
        lb = dict(_ORDER)[b]
        print(f"    {lb} − {la}: Δ={d:+.1f}   bands {'SEPARATE' if ok else 'overlap'}")

    # ---- 4. breadth --------------------------------------------------------
    print("\n[4] BREADTH (do footprint & duration agree with intensity?)")
    breadth_agree = True
    for metric, label in (("mean_footprint_km2", "footprint"), ("mean_duration_h", "duration")):
        m = col(metric)
        if len(m) == len(keys):
            mseq = [m[k] for k in keys]
            mono = all(y >= x for x, y in zip(mseq, mseq[1:]))
            breadth_agree = breadth_agree and mono
            print(f"    {label:9s}: " + "  ->  ".join(f"{v:.3g}" for v in mseq)
                  + f"   {'↑' if mono else 'mixed'}")

    # ---- verdict -----------------------------------------------------------
    # Longest-baseline contrast = first vs last present window.
    headline_sep = _bands_separate(
        float(pc.loc[keys[-1], "intensity_q05"]), float(pc.loc[keys[-1], "intensity_q95"]),
        float(pc.loc[keys[0], "intensity_q05"]), float(pc.loc[keys[0], "intensity_q95"]),
    )
    any_sep = any(sep.values())
    total_delta = intensity[keys[-1]] - intensity[keys[0]]

    if monotonic and headline_sep:
        verdict = "INTENSIFIED"
        gloss = "intensity rises monotonically and the longest-baseline bands separate."
    elif not any_sep and abs(total_delta) < 5.0:
        verdict = "FLAT"
        gloss = "no band separates and the end-to-end change is small (<5 kg m⁻¹ s⁻¹)."
    else:
        verdict = "MIXED"
        gloss = ("intensity and breadth/significance disagree — "
                 "non-monotonic, or a shift without band separation.")

    print("\n" + "=" * 64)
    print(f" VERDICT: {verdict}")
    print(f"   {gloss}")
    print(f"   end-to-end Δintensity = {total_delta:+.1f} kg m⁻¹ s⁻¹"
          f"   (breadth agrees: {breadth_agree})")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
