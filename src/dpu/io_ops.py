"""I/O boundary: the only module that touches the file system."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final

from dpu.detection import detect_format, format_from_name
from dpu.models import DataProcessingError, FileFormat, ProcessingResult, RawDocument

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIZE_BYTES: Final[int] = 200 * 1024 * 1024
_UNSAFE_FILENAME: Final[re.Pattern[str]] = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def read_document(
    path: str | Path,
    file_format: FileFormat | None = None,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
) -> RawDocument:
    """Read a file into a :class:`RawDocument`, detecting its format when not given."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise DataProcessingError(f"Not a regular file: {file_path}")
    size = file_path.stat().st_size
    if size > max_size_bytes:
        raise DataProcessingError(f"{file_path.name} is {size} bytes, above the {max_size_bytes}-byte limit")
    content = file_path.read_bytes()
    resolved_format = file_format or detect_format(file_path.name, content)
    logger.info("Read %s (%d bytes, format=%s)", file_path.name, size, resolved_format.value)
    return RawDocument(name=file_path.stem, file_format=resolved_format, content=content, source_path=file_path)


def list_supported_files(directory: str | Path, recursive: bool = False) -> list[Path]:
    """Return files in ``directory`` whose extension maps to a supported format."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    candidates = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in candidates if p.is_file() and format_from_name(p.name) is not None)


def safe_filename(name: str) -> str:
    """Replace characters that are illegal in Windows/Unix file names."""
    return _UNSAFE_FILENAME.sub("_", name).strip(" .") or "table"


def export_tables(result: ProcessingResult, output_dir: str | Path) -> list[Path]:
    """Write every cleaned table to ``output_dir`` as UTF-8 (BOM) CSV, Excel-friendly for Arabic."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for table in result.tables:
        target = out / f"{safe_filename(table.name)}.csv"
        table.frame.to_csv(target, index=False, encoding="utf-8-sig")
        written.append(target)
        logger.info("Wrote %s (%d rows)", target.name, len(table.frame))
    return written
