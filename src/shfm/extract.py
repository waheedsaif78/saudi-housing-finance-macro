"""Sheet-specific extractors: dpu grid -> long rows (date, freq, series, value, unit, source, source_table)."""

from __future__ import annotations

import re

import pandas as pd

from shfm.common import is_date, month_end, quarter_end, to_float

_Q_LABEL = re.compile(r"^Q([1-4])$")


def _long(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "freq", "series", "value", "unit", "source", "source_table"])


def _header_row(grid: pd.DataFrame, label: str = "Period") -> int:
    for i in range(len(grid)):
        if str(grid.iat[i, 0]).strip() == label:
            return i
    raise ValueError(f"header '{label}' not found")


def _last_quarter_row(grid: pd.DataFrame, start: int) -> int:
    last = -1
    for i in range(start, len(grid)):
        if _Q_LABEL.match(str(grid.iat[i, 0]).strip()):
            last = i
    if last < 0:
        raise ValueError("no quarterly block found")
    return last


# --------------------------------------------------------------------------- SAMA mortgages (12f / 2-7)
MORTGAGE_COLS = {
    "Total Number of Contracts": "contracts",
    "Houses": "value_villa",  # SAMA "Houses" == الفلل
    "Apartments": "value_apartment",
    "Land": "value_land",
    "Total": "value_total",
}


def sama_mortgages(grid: pd.DataFrame, lender: str, table: str) -> pd.DataFrame:
    """Monthly residential new-mortgage block (rows after the quarterly block)."""
    h = _header_row(grid)
    positions = {}
    for j in range(grid.shape[1]):
        key = str(grid.iat[h, j]).strip()
        if key in MORTGAGE_COLS:
            positions[MORTGAGE_COLS[key]] = j
    if len(positions) != len(MORTGAGE_COLS):
        raise ValueError(f"{table}: unexpected header {list(grid.iloc[h])}")
    rows: list[dict[str, object]] = []
    for i in range(_last_quarter_row(grid, h) + 1, len(grid)):
        cell = grid.iat[i, 0]
        if not is_date(cell):
            continue
        for name, j in positions.items():
            unit = "contracts" if name == "contracts" else "SAR mn"
            rows.append(dict(date=month_end(cell), freq="M", series=f"mortgage_{lender}_{name}",
                             value=to_float(grid.iat[i, j]), unit=unit, source="SAMA Monthly Bulletin", source_table=table))
    return _long(rows)


def sama_mortgages_quarterly(grid: pd.DataFrame, lender: str, table: str) -> pd.DataFrame:
    """SAMA's own quarterly block, kept only to validate our monthly->quarterly aggregation."""
    h = _header_row(grid)
    j_total = next(j for j in range(grid.shape[1]) if str(grid.iat[h, j]).strip() == "Total")
    j_cnt = next(j for j in range(grid.shape[1]) if str(grid.iat[h, j]).strip() == "Total Number of Contracts")
    rows = []
    for i in range(h + 1, len(grid)):
        m = _Q_LABEL.match(str(grid.iat[i, 0]).strip())
        if m:
            d = quarter_end(int(float(str(grid.iat[i, 1]))), int(m.group(1)))
            rows.append(dict(date=d, lender=lender, sama_q_value=to_float(grid.iat[i, j_total]),
                             sama_q_contracts=to_float(grid.iat[i, j_cnt])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- SAMA interest rates (5-6)
def sama_rates(grid: pd.DataFrame, table: str = "5-6") -> pd.DataFrame:
    h = _header_row(grid)
    cols = {"RR": "repo_rate", "RRR": "reverse_repo_rate"}
    positions = {cols[str(grid.iat[h, j]).strip()]: j for j in range(grid.shape[1]) if str(grid.iat[h, j]).strip() in cols}
    # 3M SAIBOR sits under the "Monthly Avg." header, labelled one row below.
    positions["saibor_3m"] = next(j for j in range(grid.shape[1]) if "SAIBOR" in str(grid.iat[h + 1, j]))
    rows: list[dict[str, object]] = []
    for i in range(_last_quarter_row(grid, h) + 1, len(grid)):
        cell = grid.iat[i, 0]
        if not is_date(cell):
            continue
        for name, j in positions.items():
            rows.append(dict(date=month_end(cell), freq="M", series=name, value=to_float(grid.iat[i, j]),
                             unit="% p.a.", source="SAMA Monthly Bulletin", source_table=table))
    return _long(rows)


# --------------------------------------------------------------------------- GASTAT CPI (5.1)
CPI_CODES = {"0": "cpi_general", "4": "cpi_housing_utilities", "4.1": "cpi_rent_actual"}


def gastat_cpi(grid: pd.DataFrame, table: str = "5.1", source: str = "GASTAT CPI") -> pd.DataFrame:
    code_row = next(i for i in range(10) if str(grid.iat[i, 0]).startswith("COICOP"))
    positions = {}
    for j in range(grid.shape[1]):
        code = str(grid.iat[code_row, j]).strip()
        code = code[:-2] if code.endswith(".0") else code
        if code in CPI_CODES and CPI_CODES[code] not in positions:
            positions[CPI_CODES[code]] = j
    rows: list[dict[str, object]] = []
    for i in range(code_row + 1, len(grid)):
        y, m = grid.iat[i, 0], grid.iat[i, 1]
        if not (isinstance(y, (int, float)) and isinstance(m, (int, float))):
            continue
        d = month_end(pd.Timestamp(int(y), int(m), 1))
        for name, j in positions.items():
            rows.append(dict(date=d, freq="M", series=name, value=to_float(grid.iat[i, j]), unit="index 2023=100",
                             source=source, source_table=table))
    return _long(rows)


# --------------------------------------------------------------------------- GASTAT RPI (new methodology, 2023=100)
RPI_TYPES = ["general", "residential", "res_plot", "villa", "apartment", "floor",
             "commercial", "com_plot", "com_building", "com_shop", "agricultural", "agri_land"]
RPI_REGIONS = ["general", "riyadh", "makkah", "madinah", "qassim", "eastern", "aseer", "tabuk",
               "hail", "northern_borders", "jazan", "najran", "baha", "jouf"]


def gastat_rpi(grid: pd.DataFrame, labels: list[str], prefix: str, table: str, source: str = "GASTAT RPI") -> pd.DataFrame:
    """Rows: year | quarter(ar) | Index,QoQ,YoY x N | 'Qn' | year. Annual-average rows are skipped."""
    hdr = next(i for i in range(len(grid)) if any(str(grid.iat[i, j]).strip() == "QoQ" for j in range(grid.shape[1])))
    index_cols = [j for j in range(grid.shape[1]) if str(grid.iat[hdr, j]).strip().startswith("Index")]
    if len(index_cols) != len(labels):
        raise ValueError(f"{table}: expected {len(labels)} index columns, found {len(index_cols)}")
    rows: list[dict[str, object]] = []
    for i in range(hdr + 1, len(grid)):
        year = grid.iat[i, 0]
        qlabels = [str(grid.iat[i, j]).strip() for j in range(grid.shape[1])]
        qmatch = next((_Q_LABEL.match(s) for s in qlabels if _Q_LABEL.match(s)), None)
        if not isinstance(year, (int, float)) or qmatch is None:
            continue
        d = quarter_end(int(year), int(qmatch.group(1)))
        for name, j in zip(labels, index_cols, strict=True):
            rows.append(dict(date=d, freq="Q", series=f"{prefix}_{name}", value=to_float(grid.iat[i, j]),
                             unit="index 2023=100", source=source, source_table=table))
    return _long(rows)


def gastat_rpi_weights(grid: pd.DataFrame) -> pd.DataFrame:
    names = {"General index": "general", "Residential": "residential", "Residential Plot": "res_plot", "Villa": "villa",
             "Apartment": "apartment", "Floor": "floor", "Commercial": "commercial", "Agricultural": "agricultural"}
    out = []
    for i in range(len(grid)):
        en = str(grid.iat[i, grid.shape[1] - 1]).strip()
        if en in names:
            out.append(dict(type=names[en], weight_pct=to_float(grid.iat[i, 1])))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- old RPI (2014=100, via KAPSARC)
OLD_RPI = {"General index": "general", "Residential": "residential", "Residential: Plot": "res_plot",
           "Residential: Villa": "villa", "Residential: Apartment": "apartment"}


def kapsarc_old_rpi(frame: pd.DataFrame) -> pd.DataFrame:
    q = frame[(frame["periodicity"] == "Quarterly") & frame["indicator"].isin(OLD_RPI)]
    rows = [dict(date=quarter_end(int(r.year), int(str(r.quarter)[1])), freq="Q", series=f"rpi_old_{OLD_RPI[r.indicator]}",
                 value=float(r.value), unit="index 2014=100", source="GASTAT via KAPSARC", source_table="real-estate-indices")
            for r in q.itertuples()]
    return _long(rows)
