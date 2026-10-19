"""Orchestration: wires the I/O layer to the pure parsing and cleaning functions."""

from __future__ import annotations

import logging
from pathlib import Path

from dpu.cleaning import clean_frame
from dpu.io_ops import list_supported_files, read_document
from dpu.models import CleanedTable, CleaningConfig, DataProcessingError, FileFormat, ProcessingResult, RawDocument
from dpu.parsers import get_parser

logger = logging.getLogger(__name__)


def process_document(document: RawDocument, config: CleaningConfig | None = None) -> ProcessingResult:
    """Parse and clean an in-memory document (no file-system access)."""
    parser = get_parser(document.file_format)
    parsed = parser(document.content, document.name)
    if not parsed:
        logger.warning("No tabular content found in %s", document.name)
    tables: list[CleanedTable] = []
    for table in parsed:
        frame, report = clean_frame(table.frame, config)
        logger.info(
            "Cleaned %s: rows %d->%d, cols %d->%d, duplicates=%d, typed=%s",
            table.name,
            report.rows_before,
            report.rows_after,
            report.columns_before,
            report.columns_after,
            report.duplicates_dropped,
            report.converted_columns or "{}",
        )
        tables.append(CleanedTable(table.name, frame, report))
    return ProcessingResult(document.name, document.file_format, tuple(tables))


def process_bytes(
    content: bytes, name: str, file_format: FileFormat, config: CleaningConfig | None = None
) -> ProcessingResult:
    """Convenience wrapper for content that is already in memory (e.g. an upload)."""
    return process_document(RawDocument(name, file_format, content), config)


def process_file(
    path: str | Path, config: CleaningConfig | None = None, file_format: FileFormat | None = None
) -> ProcessingResult:
    """Read a file from disk, then parse and clean it."""
    return process_document(read_document(path, file_format), config)


def process_directory(
    directory: str | Path, config: CleaningConfig | None = None, recursive: bool = False
) -> tuple[list[ProcessingResult], dict[Path, str]]:
    """Process every supported file; failures are collected instead of aborting the batch."""
    results: list[ProcessingResult] = []
    failures: dict[Path, str] = {}
    for path in list_supported_files(directory, recursive):
        try:
            results.append(process_file(path, config))
        except (DataProcessingError, OSError) as exc:
            logger.error("Failed to process %s: %s", path.name, exc)
            failures[path] = str(exc)
    logger.info("Processed %d file(s), %d failure(s)", len(results), len(failures))
    return results, failures
