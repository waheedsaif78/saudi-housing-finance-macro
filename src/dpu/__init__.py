"""Data Processing Unit: turn JSON/CSV/XLSX/XML/PDF/DOCX/PPTX/PBIX files into clean pandas tables."""

from __future__ import annotations

import logging

from dpu.cleaning import clean_frame
from dpu.logging_config import configure_logging
from dpu.models import (
    CleanedTable,
    CleaningConfig,
    CleaningReport,
    DataProcessingError,
    FileFormat,
    ParsedTable,
    ParseError,
    ProcessingResult,
    RawDocument,
    UnsupportedFormatError,
)
from dpu.pipeline import process_bytes, process_directory, process_document, process_file

logging.getLogger(__name__).setLevel(logging.INFO)

__all__ = [
    "CleanedTable",
    "CleaningConfig",
    "CleaningReport",
    "DataProcessingError",
    "FileFormat",
    "ParseError",
    "ParsedTable",
    "ProcessingResult",
    "RawDocument",
    "UnsupportedFormatError",
    "clean_frame",
    "configure_logging",
    "process_bytes",
    "process_directory",
    "process_document",
    "process_file",
]
