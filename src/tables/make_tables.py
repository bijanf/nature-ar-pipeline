r"""LaTeX table bodies for the manuscript from the result JSON files.

Writes into the manuscript tables folder:
  tab_terms.tex          headline decomposition per corridor (main text)
  tab_terms_original.tex the same for the 1940-1959 vs 2015-2024 pair (supplement)
  tab_trends.tex         Hamed-Rao trends of the yearly contributions (supplement)
  tab_sensitivity.tex    wind term for every period pair (supplement)
  tab_seasons.tex        seasonal terms (supplement)
  tab_cmip6_trends.tex   observed trend of the wind contribution against the CMIP6 members (supplement)
Captions are written by the authors; the bodies use the Copernicus rules
(\tophline, \middlehline, \bottomhline).

Usage: python -m src.tables.make_tables RESULTS_DIR OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ORDER = ["Amazon outflow", "SE South America", "Western Europe", "East Asia",
         "US West Coast", "SE US / Gulf", "South Africa", "SE Australia / NZ"]
NAME = {"SE South America": "SE South America", "SE US / Gulf": "SE United States and Gulf",
        "SE Australia / NZ": "SE Australia and New Zealand"}


def nm(n):
    return NAME.get(n, n)


def sig(p):
    return "$^{*}$" if p < 0.05 else ""


def fmt(v, d=1):
    return f"{v:+.{d}f}".replace("-", "$-$")


def terms_table(rows, out: Path):
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\tophline",
             r"Corridor & Base & Total & Moisture & \multicolumn{2}{c}{of which} & Wind & Covariance & Transient \\",
             r" & & & & fixed RH & RH change & & & \\", r"\middlehline"]
    for n in ORDER:
        r = next(x for x in rows if x["corridor"] == n)
        ci = r["wind_ci95"]
        lines.append(f"{nm(n)} & {r['base']:.0f} & {fmt(r['total'])} & {fmt(r['moist'])}{sig(r['moist_pfdr'])} & "
                     f"{fmt(r['moist_cc'])} & {fmt(r['moist_rh'])} & {fmt(r['wind'])}{sig(r['wind_pfdr'])} "
                     f"[{ci[0]:.1f}, {ci[1]:.1f}] & {fmt(r['cov'])} & {fmt(r['trans'])}{sig(r['trans_pfdr'])} \\\\")
    lines += [r"\bottomhline", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def trends_table(res, out: Path):
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\tophline",
             r"Corridor & \multicolumn{4}{c}{1940--2024} & \multicolumn{3}{c}{1979--2024} \\",
             r" & Total & Moisture & Wind & Transient & Moisture & Wind & Transient \\", r"\middlehline"]
    for n in ORDER:
        f, s = res["trends_full"][n], res["trends_sat"][n]
        g = lambda t, k: f"{fmt(t[k]['pct_per_decade'], 2)}{sig(t[k]['p'])}"  # noqa: E731
        lines.append(f"{nm(n)} & {g(f,'total')} & {g(f,'moist')} & {g(f,'wind')} & {g(f,'trans')} & "
                     f"{g(s,'moist')} & {g(s,'wind')} & {g(s,'trans')} \\\\")
    lines += [r"\bottomhline", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def sensitivity_table(rows, out: Path):
    Ls = sorted({r["L"] for r in rows})
    bases = sorted({r["base_start"] for r in rows})
    lines = [r"\begin{tabular}{ll" + "r" * len(Ls) + "}", r"\tophline",
             "Corridor & Base from & " + " & ".join(f"{L}\\,yr" for L in Ls) + r" \\", r"\middlehline"]
    for n in ORDER:
        for i, b in enumerate(bases):
            cells = []
            for L in Ls:
                r = [x for x in rows if x["corridor"] == n and x["L"] == L and x["base_start"] == b]
                cells.append(f"{fmt(r[0]['wind'])}{sig(r[0]['wind_pfdr'])}" if r else "--")
            lines.append(f"{nm(n) if i == 0 else ''} & {b} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomhline", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def seasons_table(res, out: Path):
    seasons = ["DJF", "MAM", "JJA", "SON"]
    lines = [r"\begin{tabular}{l" + "rrr" * 4 + "}", r"\tophline",
             "Corridor & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{s}}}" for s in seasons) + r" \\",
             " & " + " & ".join("Moisture & Wind & Transient" for _ in seasons) + r" \\", r"\middlehline"]
    for n in ORDER:
        cells = []
        for s in seasons:
            r = next(x for x in res[s] if x["corridor"] == n)
            cells += [f"{fmt(r['moist'])}{sig(r['moist_pfdr'])}", f"{fmt(r['wind'])}{sig(r['wind_pfdr'])}",
                      f"{fmt(r['trans'])}{sig(r['trans_pfdr'])}"]
        lines.append(f"{nm(n)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomhline", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def trend_test_table(res, out: Path):
    """Observed trend of the yearly wind contribution against the CMIP6 members (cmip6_trend_test.json)."""
    large = res["large"]
    head = " & ".join(large)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\tophline",
             f"Corridor & ERA5 & {head} & All members & Members \\\\", r"\middlehline"]
    for n in ORDER:
        r = res[n]["wind_slope_per_decade"]
        cells = [f"{fmt(r['obs'], 2)}{sig(r['obs_p'])}"]
        for m in large:
            e = r[m]
            cells.append(f"{fmt(e['mean'], 2)} $\\pm$ {e['sd']:.2f} ({100 * e['frac_as_extreme']:.0f}\\,\\%)")
        mm = r["multi_model"]
        cells.append(f"{fmt(mm['weighted_mean'], 2)} [{fmt(mm['p05'], 2)}, {fmt(mm['p95'], 2)}]")
        cells.append(str(mm["n_members_as_extreme"]))
        lines.append(f"{nm(n)} & " + " & ".join(cells) + " \\\\")
    lines += [r"\bottomhline", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def main():
    res_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    h = json.load(open(res_dir / "fullcolumn_headline.json"))
    terms_table(h["headline"], out_dir / "tab_terms.tex")
    terms_table(h["original"], out_dir / "tab_terms_original.tex")
    trends_table(h, out_dir / "tab_trends.tex")
    sensitivity_table(json.load(open(res_dir / "fullcolumn_sensitivity.json"))["matrix"], out_dir / "tab_sensitivity.tex")
    seasons_table(json.load(open(res_dir / "fullcolumn_seasons.json")), out_dir / "tab_seasons.tex")
    trend_test_table(json.load(open(res_dir / "cmip6_trend_test.json")), out_dir / "tab_cmip6_trends.tex")
    print("wrote tables to", out_dir)


if __name__ == "__main__":
    main()
