"""Pure cleaning functions: ``DataFrame -> DataFrame``.

Every function returns a new DataFrame and never mutates its input.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Hashable
from typing import Any, Final

import pandas as pd

from dpu.models import CleaningConfig, CleaningReport
from dpu.parsers import make_unique

_ARABIC_DIGITS: Final[dict[int, str]] = {
    **{ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")},  # Arabic-Indic
    **{ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")},  # Extended (Persian)
    ord("٫"): ".",  # Arabic decimal separator
    ord("٬"): ",",  # Arabic thousands separator
    ord("−"): "-",  # Unicode minus
}
# Zero-width / bidi control characters common in copy-pasted Arabic data.
_INVISIBLE_CHARS: Final[re.Pattern[str]] = re.compile("[​-‏‪-‮⁦-⁩﻿]")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")
_NON_WORD: Final[re.Pattern[str]] = re.compile(r"[^\w]+")
_UNNAMED: Final[re.Pattern[str]] = re.compile(r"^unnamed:?\s*\d+(_level_\d+)?$", re.IGNORECASE)
_NUMBER: Final[re.Pattern[str]] = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_THOUSANDS: Final[re.Pattern[str]] = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_LEADING_ZERO: Final[re.Pattern[str]] = re.compile(r"^[+-]?0\d")
_DATE_LIKE: Final[re.Pattern[str]] = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{4}-\d{2}-\d{2}T")
_TRUE_TOKENS: Final[frozenset[str]] = frozenset({"true", "yes", "y", "t", "نعم"})
_FALSE_TOKENS: Final[frozenset[str]] = frozenset({"false", "no", "n", "f", "لا"})


# --------------------------------------------------------------------------- scalars
def normalize_digits(text: str) -> str:
    """Translate Arabic-Indic digits and separators to ASCII."""
    return text.translate(_ARABIC_DIGITS)


def clean_text(text: str) -> str:
    """NFKC-normalize, drop invisible bidi marks and collapse whitespace."""
    normalized = unicodedata.normalize("NFKC", text)
    return _WHITESPACE.sub(" ", _INVISIBLE_CHARS.sub("", normalized)).strip()


def normalize_column_name(name: Hashable, position: int) -> str:
    """Convert a header into ``snake_case`` (Unicode letters such as Arabic are kept)."""
    fallback = f"column_{position + 1}"
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return fallback
    text = clean_text(str(name)).lower()
    if _UNNAMED.match(text):
        return fallback
    snake = _NON_WORD.sub("_", text).strip("_")
    return re.sub(r"_+", "_", snake) or fallback


def parse_number(text: str) -> float | None:
    """Parse a human-formatted number, returning None if it is not one.

    Handles thousands separators, accounting negatives ``(1,234)``, percentages,
    and Arabic-Indic digits.
    """
    value = normalize_digits(text).strip().replace(" ", "")
    negative = value.startswith("(") and value.endswith(")")
    if negative:
        value = value[1:-1]
    percent = value.endswith("%")
    if percent:
        value = value[:-1]
    if _THOUSANDS.match(value):
        value = value.replace(",", "")
    if not _NUMBER.match(value):
        return None
    number = float(value)
    if percent:
        number /= 100
    return -number if negative else number


# --------------------------------------------------------------------------- frame steps
def _object_columns(frame: pd.DataFrame) -> list[int]:
    return [i for i, dtype in enumerate(frame.dtypes) if dtype == object or pd.api.types.is_string_dtype(dtype)]


def _map_columns(frame: pd.DataFrame, columns: list[int], func: Callable[[Any], object]) -> pd.DataFrame:
    result = frame.copy()
    for index in columns:
        result.isetitem(index, result.iloc[:, index].map(func, na_action="ignore").astype(object).array)
    return result


def normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a frame with snake_case unique column names, plus the rename map."""
    original = [str(c) for c in frame.columns]
    new_names = make_unique([normalize_column_name(c, i) for i, c in enumerate(frame.columns)])
    result = frame.copy()
    result.columns = pd.Index(new_names)
    renamed = {old: new for old, new in zip(original, new_names) if old != new}
    return result, renamed


def stringify_unhashable(frame: pd.DataFrame) -> pd.DataFrame:
    """Serialize nested lists/dicts (e.g. from JSON) so the frame becomes hashable."""

    def convert(value: object) -> object:
        if isinstance(value, (list, dict, set, tuple)):
            payload = sorted(value, key=str) if isinstance(value, set) else value
            return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return value

    return _map_columns(frame, _object_columns(frame), convert)


def strip_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize every string cell with :func:`clean_text`."""
    return _map_columns(frame, _object_columns(frame), lambda v: clean_text(v) if isinstance(v, str) else v)


def replace_null_tokens(frame: pd.DataFrame, tokens: frozenset[str]) -> pd.DataFrame:
    """Turn placeholder strings such as ``"N/A"`` or ``"-"`` into real missing values."""
    lowered = frozenset(token.lower() for token in tokens)

    def convert(value: object) -> object:
        return None if isinstance(value, str) and value.strip().lower() in lowered else value

    return _map_columns(frame, _object_columns(frame), convert)


def drop_empty_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.shape[1] == 0:
        return frame.copy(), 0
    result = frame.dropna(axis=0, how="all")
    return result, len(frame) - len(result)


def drop_empty_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    result = frame.dropna(axis=1, how="all") if len(frame) else frame.copy()
    return result, frame.shape[1] - result.shape[1]


def drop_duplicate_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.shape[1] == 0:
        return frame.copy(), 0
    result = frame.drop_duplicates()
    return result, len(frame) - len(result)


def try_convert_boolean(series: pd.Series) -> pd.Series | None:
    """Convert a column whose every value is a yes/no token to pandas ``boolean``."""
    values = series.dropna()
    if values.empty or not all(isinstance(v, str) for v in values):
        return None
    lowered = {str(v).strip().lower() for v in values}
    if not lowered <= (_TRUE_TOKENS | _FALSE_TOKENS):
        return None
    mapped = series.map(lambda v: str(v).strip().lower() in _TRUE_TOKENS, na_action="ignore")
    return mapped.astype("boolean")


def try_convert_numeric(series: pd.Series, threshold: float) -> pd.Series | None:
    """Convert to ``Int64``/``float64`` when at least ``threshold`` of values parse.

    Columns with leading-zero codes (``"00123"``) are left as text to avoid data loss.
    """
    values = series.dropna()
    if values.empty:
        return None
    texts = values.map(str)
    if texts.map(lambda t: bool(_LEADING_ZERO.match(normalize_digits(t).strip()))).any():
        return None
    parsed = series.map(lambda v: parse_number(str(v)), na_action="ignore")
    numeric = pd.to_numeric(parsed, errors="coerce")
    if numeric.notna().sum() / len(values) < threshold:
        return None
    non_null = numeric.dropna()
    if (non_null == non_null.round()).all() and (non_null.abs() < 2**53).all():
        return numeric.round().astype("Int64")
    return numeric.astype("float64")


def try_convert_datetime(series: pd.Series, threshold: float) -> pd.Series | None:
    """Convert date-looking text columns to ``datetime64``."""
    values = series.dropna()
    if values.empty:
        return None
    texts = values.map(lambda v: normalize_digits(str(v)))
    if texts.map(lambda t: bool(_DATE_LIKE.search(t))).mean() < threshold:
        return None
    converted = pd.to_datetime(series.map(lambda v: normalize_digits(str(v)), na_action="ignore"), errors="coerce", format="mixed")
    if converted.notna().sum() / len(values) < threshold:
        return None
    return converted


def infer_column_types(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict[str, str]]:
    """Try boolean, then numeric, then datetime conversion on every text column."""
    result = frame.copy()
    converted: dict[str, str] = {}
    for index in _object_columns(frame):
        series = frame.iloc[:, index]
        new = try_convert_boolean(series)
        if new is None:
            new = try_convert_numeric(series, threshold)
        if new is None:
            new = try_convert_datetime(series, threshold)
        if new is not None:
            result.isetitem(index, new.array)
            converted[str(frame.columns[index])] = str(new.dtype)
    return result, converted


# --------------------------------------------------------------------------- composition
def clean_frame(frame: pd.DataFrame, config: CleaningConfig | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline and describe what changed."""
    cfg = config or CleaningConfig()
    rows_before, columns_before = frame.shape
    result = stringify_unhashable(frame)
    renamed: dict[str, str] = {}
    if cfg.normalize_column_names:
        result, renamed = normalize_columns(result)
    result = replace_null_tokens(strip_strings(result), cfg.null_tokens)
    empty_rows = empty_columns = duplicates = 0
    if cfg.drop_empty_rows:
        result, empty_rows = drop_empty_rows(result)
    if cfg.drop_empty_columns:
        result, empty_columns = drop_empty_columns(result)
    if cfg.drop_duplicates:
        result, duplicates = drop_duplicate_rows(result)
    converted: dict[str, str] = {}
    if cfg.infer_types:
        result, converted = infer_column_types(result, cfg.type_inference_threshold)
    result = result.reset_index(drop=True)
    report = CleaningReport(
        rows_before=rows_before,
        rows_after=len(result),
        columns_before=columns_before,
        columns_after=result.shape[1],
        empty_rows_dropped=empty_rows,
        empty_columns_dropped=empty_columns,
        duplicates_dropped=duplicates,
        renamed_columns=renamed,
        converted_columns=converted,
    )
    return result, report
