"""Shared paths, dpu bridge and small helpers for the saudi-housing-finance-macro project."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
TABLES = PROJECT_ROOT / "outputs" / "tables"
for _p in (PROCESSED, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# The Data Processing Unit (dpu) is vendored in src/dpu. To use another copy (e.g. a newer dpu checkout),
# set DPU_PATH to the folder that CONTAINS the dpu package; it then takes precedence.
_SRC = PROJECT_ROOT / "src"
for _p in (str(_SRC), os.environ.get("DPU_PATH")):  # later insert wins
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from dpu import CleaningConfig, process_file  # noqa: E402

logging.getLogger("dpu").setLevel(logging.WARNING)

# Layout-preserving config: statistical bulletins are "grid" sheets (multi-row headers,
# repeated values), so we keep every column/duplicate and do typing ourselves.
GRID_CONFIG = CleaningConfig(
    drop_duplicates=False,
    drop_empty_columns=False,
    normalize_column_names=False,
    infer_types=False,
)

_CACHE: dict[Path, dict[str, pd.DataFrame]] = {}


def load_grids(path: Path, config: CleaningConfig = GRID_CONFIG) -> dict[str, pd.DataFrame]:
    """Parse a workbook/CSV with dpu and return {sheet_suffix: grid} where row 0 is the promoted header."""
    if path not in _CACHE:
        result = process_file(path, config)
        grids: dict[str, pd.DataFrame] = {}
        for table in result.tables:
            frame = table.frame
            header = pd.DataFrame([list(frame.columns)], columns=range(frame.shape[1]))
            body = pd.DataFrame(frame.values, columns=range(frame.shape[1]))
            suffix = table.name.split(".", 1)[1].strip() if "." in table.name else table.name
            grids[suffix] = pd.concat([header, body], ignore_index=True)
        _CACHE[path] = grids
    return _CACHE[path]


def load_table(path: Path) -> pd.DataFrame:
    """Parse a tidy file (CSV) with dpu's default cleaning (typed, de-duplicated)."""
    return process_file(path).tables[0].frame


def to_float(value: object) -> float:
    """Numeric cell -> float; placeholders such as '---', '-', None -> NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def is_date(value: object) -> bool:
    return isinstance(value, (datetime, pd.Timestamp))


def quarter_end(year: int, q: int) -> pd.Timestamp:
    return pd.Period(f"{year}Q{q}", freq="Q").to_timestamp(how="end").normalize()


def month_end(ts: object) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp(how="end").normalize()


def save_fig(fig, name: str) -> Path:  # type: ignore[no-untyped-def]
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
