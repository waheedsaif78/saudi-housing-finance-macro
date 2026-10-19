"""01 - Acquisition: official files in data/raw -> one long Parquet (date, series, value, ...).

Sources (see README for URLs). To update, drop the new release into data/raw keeping the name pattern;
the LATEST file of each pattern is used automatically:
  * sama__Monthly_Bulletin*.xlsx         SAMA Monthly Statistical Bulletin: tables 12f, Finance-Co 2-7, 5-6
  * gastat__REPI_Q<n>_<yyyy>*.xlsx        GASTAT Real Estate Price Index (new methodology, 2023=100)
  * gastat__CPI_Tables_<Mon>_<yyyy>*.xlsx GASTAT CPI tables (2023=100), table 5.1
  * KAPSARC re-publication of the old RPI (2014=100) - used only for the long spliced history
All files are parsed with the Data Processing Unit (dpu); this script only maps the layouts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import re  # noqa: E402

import pandas as pd  # noqa: E402

from shfm import extract as ex  # noqa: E402
from shfm.common import PROCESSED, RAW, TABLES, load_grids, load_table  # noqa: E402

_MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def latest(pattern: str, key_re: str) -> Path:
    """Newest release among files matching `pattern`, ordered by the (year, period) parsed from the name."""
    def key(p: Path) -> tuple[int, int]:
        m = re.search(key_re, p.name)
        if not m:
            return (0, 0)
        period, year = m.group(1), int(m.group(2))
        return (year, int(period) if period.isdigit() else _MONTHS.get(period[:3].title(), 0))
    files = sorted(RAW.glob(pattern), key=key)
    if not files:
        raise FileNotFoundError(f"no file matching {pattern} in {RAW}")
    return files[-1]


def main() -> pd.DataFrame:
    f_sama = max(RAW.glob("sama__Monthly_Bulletin*.xlsx"), key=lambda p: p.stat().st_mtime)
    f_rpi = latest("gastat__REPI_Q*.xlsx", r"Q(\d)_(\d{4})")
    f_cpi = latest("gastat__CPI_Tables_*.xlsx", r"Tables_([A-Za-z]{3})[a-z]*_(\d{4})")
    print(f"inputs: {f_sama.name} | {f_rpi.name} | {f_cpi.name}")
    sama, rpi, cpi = load_grids(f_sama), load_grids(f_rpi), load_grids(f_cpi)

    parts = [
        ex.sama_mortgages(sama["12f"], "bank", "12f"),
        ex.sama_mortgages(sama["2-7"], "fin", "FinCo-7"),
        ex.sama_rates(sama["5-6"]),
        ex.gastat_cpi(cpi["5.1"], source=f"GASTAT CPI ({f_cpi.stem})"),
        ex.gastat_rpi(rpi["2"], ex.RPI_TYPES, "rpi", "RPI table 2", source=f"GASTAT RPI ({f_rpi.stem})"),
        ex.gastat_rpi(rpi["3"], ex.RPI_REGIONS, "rpi_region", "RPI table 3", source=f"GASTAT RPI ({f_rpi.stem})"),
        ex.kapsarc_old_rpi(load_table(RAW / "kapsarc__real-estate-indices.csv")),
    ]
    long = pd.concat(parts, ignore_index=True)
    long = long[long["series"] != "rpi_region_general"]  # duplicate of rpi_general
    long["date"] = pd.to_datetime(long["date"])
    long.to_parquet(PROCESSED / "long.parquet", index=False)
    ex.gastat_rpi_weights(rpi["1"]).to_csv(TABLES / "rpi_weights_2023.csv", index=False)

    # ---- validation 1: our monthly sums vs SAMA's published quarterly block
    q_pub = pd.concat([ex.sama_mortgages_quarterly(sama["12f"], "bank", "12f"),
                       ex.sama_mortgages_quarterly(sama["2-7"], "fin", "FinCo-7")])
    m = long[long.series.str.match(r"mortgage_(bank|fin)_(value_total|contracts)$")].copy()
    m["lender"] = m.series.str.split("_").str[1]
    m["what"] = m.series.str.split("_", n=2).str[2]
    m["q"] = m.date.dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()
    agg = m.groupby(["q", "lender", "what"]).agg(value=("value", "sum"), n=("value", "size")).reset_index()
    agg = agg[agg.n == 3].pivot_table(index=["q", "lender"], columns="what", values="value").reset_index()
    chk = agg.merge(q_pub, left_on=["q", "lender"], right_on=["date", "lender"])
    chk["value_gap_pct"] = 100 * (chk.value_total / chk.sama_q_value - 1)
    chk["contracts_gap_pct"] = 100 * (chk.contracts / chk.sama_q_contracts - 1)
    chk.to_csv(TABLES / "check_monthly_vs_sama_quarterly.csv", index=False)

    # ---- validation 2: KAPSARC copy of new RPI vs GASTAT workbook
    kap = load_table(RAW / "kapsarc__real-estate-price-index-by-sector-2023-100.csv")
    kmap = {"Index number": "rpi_general", "Residential: Villa": "rpi_villa", "Residential: Residential Plot": "rpi_res_plot"}
    kap = kap[(kap.measure == "Index") & (kap.periodicity == "Quarterly") & kap.sector.isin(list(kmap))]
    dates = [ex.quarter_end(int(y), int(str(q)[1])) for y, q in zip(kap.year, kap.quarter, strict=True)]
    kap = kap.assign(series=kap.sector.map(kmap), date=dates)
    k = kap.merge(long[["date", "series", "value"]], on=["date", "series"], suffixes=("_kapsarc", "_gastat"))
    k["abs_diff"] = (k.value_kapsarc - k.value_gastat).abs()
    k[["date", "series", "value_kapsarc", "value_gastat", "abs_diff"]].to_csv(TABLES / "check_kapsarc_vs_gastat_rpi.csv", index=False)

    print(f"long.parquet: {len(long):,} rows, {long.series.nunique()} series")
    print(long.groupby("series").date.agg(["min", "max", "count"]).to_string())
    print(f"\nSAMA monthly->quarterly max |gap|: value {chk.value_gap_pct.abs().max():.3f}%, "
          f"contracts {chk.contracts_gap_pct.abs().max():.3f}% over {len(chk)} lender-quarters")
    print(f"KAPSARC vs GASTAT RPI: {len(k)} matched points, max |diff| {k.abs_diff.max():.3f}")
    return long


if __name__ == "__main__":
    main()
