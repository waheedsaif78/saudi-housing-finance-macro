"""Pure format detection: file name extension first, content sniffing as fallback."""

from __future__ import annotations

import io
import zipfile
from pathlib import PurePath
from typing import Final

from dpu.models import FileFormat, UnsupportedFormatError

_EXTENSION_ALIASES: Final[dict[str, FileFormat]] = {
    "json": FileFormat.JSON,
    "jsonl": FileFormat.JSONL,
    "ndjson": FileFormat.JSONL,
    "csv": FileFormat.CSV,
    "txt": FileFormat.CSV,
    "tsv": FileFormat.TSV,
    "tab": FileFormat.TSV,
    "xlsx": FileFormat.XLSX,
    "xlsm": FileFormat.XLSX,
    "xml": FileFormat.XML,
    "pdf": FileFormat.PDF,
    "docx": FileFormat.DOCX,
    "pptx": FileFormat.PPTX,
    "pbix": FileFormat.PBIX,
    "pbit": FileFormat.PBIX,
}

# Zip members whose presence identifies an Office Open XML / Power BI container.
_ZIP_MARKERS: Final[tuple[tuple[str, FileFormat], ...]] = (
    ("xl/", FileFormat.XLSX),
    ("word/", FileFormat.DOCX),
    ("ppt/", FileFormat.PPTX),
    ("Report/", FileFormat.PBIX),
    ("DataModel", FileFormat.PBIX),
)


def format_from_name(name: str) -> FileFormat | None:
    """Map a file name's extension to a format, or None when unknown."""
    suffix = PurePath(name).suffix.lower().lstrip(".")
    return _EXTENSION_ALIASES.get(suffix)


def sniff_format(content: bytes) -> FileFormat | None:
    """Guess the format from the leading bytes / container layout."""
    head = content[:2048].lstrip(b"\xef\xbb\xbf \t\r\n")
    if head.startswith(b"%PDF"):
        return FileFormat.PDF
    if head.startswith(b"PK"):
        return _sniff_zip(content)
    if head.startswith((b"{", b"[")):
        return FileFormat.JSON
    if head.startswith(b"<"):
        return FileFormat.XML
    return None


def _sniff_zip(content: bytes) -> FileFormat | None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return None
    for marker, file_format in _ZIP_MARKERS:
        if any(member.startswith(marker) for member in names):
            return file_format
    return None


def detect_format(name: str, content: bytes) -> FileFormat:
    """Return the format of a document, raising if it cannot be determined."""
    detected = format_from_name(name) or sniff_format(content)
    if detected is None:
        raise UnsupportedFormatError(f"Cannot determine format of {name!r}")
    return detected
