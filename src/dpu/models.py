"""Dataclasses that describe every structure flowing through the unit."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd


class FileFormat(str, Enum):
    """Supported input formats (value == canonical file extension)."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    XLSX = "xlsx"
    XML = "xml"
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    PBIX = "pbix"


class DataProcessingError(Exception):
    """Base error for the data processing unit."""


class UnsupportedFormatError(DataProcessingError):
    """Raised when a file's format cannot be handled."""


class ParseError(DataProcessingError):
    """Raised when the content of a file is malformed."""


@dataclass(frozen=True, slots=True)
class RawDocument:
    """Bytes read from disk plus the metadata needed to parse them (I/O output)."""

    name: str
    file_format: FileFormat
    content: bytes
    source_path: Path | None = None

    @property
    def size_bytes(self) -> int:
        return len(self.content)


@dataclass(frozen=True, slots=True, eq=False)
class ParsedTable:
    """One table extracted from a document, before cleaning."""

    name: str
    frame: pd.DataFrame


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    """Tunable knobs for the cleaning pipeline."""

    null_tokens: frozenset[str] = frozenset(
        {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "#n/a"}
    )
    drop_empty_rows: bool = True
    drop_empty_columns: bool = True
    drop_duplicates: bool = True
    normalize_column_names: bool = True
    infer_types: bool = True
    # Minimum share of non-null values that must convert for a column to be re-typed.
    type_inference_threshold: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 < self.type_inference_threshold <= 1.0:
            raise ValueError("type_inference_threshold must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class CleaningReport:
    """What the cleaning step changed."""

    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    empty_rows_dropped: int = 0
    empty_columns_dropped: int = 0
    duplicates_dropped: int = 0
    renamed_columns: dict[str, str] = field(default_factory=dict)
    converted_columns: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, eq=False)
class CleanedTable:
    """A cleaned DataFrame plus the report describing how it was produced."""

    name: str
    frame: pd.DataFrame
    report: CleaningReport


@dataclass(frozen=True, slots=True, eq=False)
class ProcessingResult:
    """Final output for one input document."""

    source_name: str
    file_format: FileFormat
    tables: tuple[CleanedTable, ...]

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(table.name for table in self.tables)

    def get(self, name: str) -> pd.DataFrame:
        for table in self.tables:
            if table.name == name:
                return table.frame
        raise KeyError(name)
