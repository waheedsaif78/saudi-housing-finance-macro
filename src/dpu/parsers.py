"""Pure parsers: ``bytes -> tuple[ParsedTable, ...]``.

Every function here works on in-memory bytes only. No file system access and no
logging happen in this module, so each parser is deterministic and side-effect free.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Final, TypeAlias

import pandas as pd

from dpu.models import FileFormat, ParsedTable, ParseError, UnsupportedFormatError

Parser: TypeAlias = Callable[[bytes, str], tuple[ParsedTable, ...]]
Cell: TypeAlias = object

_TEXT_ENCODINGS: Final[tuple[str, ...]] = ("utf-8-sig", "cp1256", "latin-1")
_CSV_DELIMITERS: Final[str] = ",;\t|"


# --------------------------------------------------------------------------- helpers
def decode_text(content: bytes) -> str:
    """Decode bytes trying UTF-16 (BOM), UTF-8, Windows-Arabic, then Latin-1."""
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16")
    for encoding in _TEXT_ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ParseError("Unable to decode text content")  # pragma: no cover - latin-1 never fails


def make_unique(names: Sequence[str]) -> list[str]:
    """Suffix repeated names with ``_2``, ``_3``... keeping order."""
    seen: Counter[str] = Counter()
    taken = set(names)
    result: list[str] = []
    for name in names:
        seen[name] += 1
        if seen[name] == 1:
            result.append(name)
            continue
        candidate = f"{name}_{seen[name]}"
        while candidate in taken:
            seen[name] += 1
            candidate = f"{name}_{seen[name]}"
        taken.add(candidate)
        result.append(candidate)
    return result


def _is_blank(value: Cell) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def rows_to_frame(rows: Sequence[Sequence[Cell]]) -> pd.DataFrame:
    """Build a DataFrame from raw rows, promoting the first non-blank row to header.

    Ragged rows are padded with ``None`` (or truncated) to the header width.
    """
    non_blank = [list(row) for row in rows if not all(_is_blank(cell) for cell in row)]
    if not non_blank:
        return pd.DataFrame()
    width = max(len(row) for row in non_blank)
    header_row, *body = non_blank
    header = [
        "" if _is_blank(cell) else str(cell).strip()
        for cell in header_row + [None] * (width - len(header_row))
    ]
    header = make_unique([name or f"column_{index + 1}" for index, name in enumerate(header)])
    padded = [row + [None] * (width - len(row)) for row in body]
    return pd.DataFrame(padded, columns=header, dtype=object)


# --------------------------------------------------------------------------- JSON
def _frame_from_json_list(items: list[Any]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    if all(isinstance(item, dict) for item in items):
        return pd.json_normalize(items)
    if all(isinstance(item, list) for item in items):
        return rows_to_frame(items)
    return pd.DataFrame({"value": items})


def tables_from_json_value(data: Any, name: str) -> tuple[ParsedTable, ...]:
    """Convert an already-decoded JSON value into one or more tables."""
    if isinstance(data, list):
        return (ParsedTable(name, _frame_from_json_list(data)),)
    if isinstance(data, dict):
        record_lists = {
            key: value
            for key, value in data.items()
            if isinstance(value, list) and value and all(isinstance(v, dict) for v in value)
        }
        if record_lists:
            return tuple(
                ParsedTable(f"{name}.{key}", pd.json_normalize(value))
                for key, value in record_lists.items()
            )
        lengths = {len(v) for v in data.values() if isinstance(v, list)}
        if data and len(lengths) == 1 and all(isinstance(v, list) for v in data.values()):
            return (ParsedTable(name, pd.DataFrame(data)),)
        return (ParsedTable(name, pd.json_normalize([data])),)
    return (ParsedTable(name, pd.DataFrame({"value": [data]})),)


def parse_jsonl(content: bytes, name: str) -> tuple[ParsedTable, ...]:
    """Parse JSON Lines (one JSON value per non-empty line)."""
    records: list[Any] = []
    for line_number, line in enumerate(decode_text(content).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ParseError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
    if not records:
        raise ParseError("JSON Lines document is empty")
    return (ParsedTable(name, _frame_from_json_list(records)),)


def parse_json(content: bytes, name: str) -> tuple[ParsedTable, ...]:
    """Parse a JSON document; falls back to JSON Lines when it is not a single value."""
    text = decode_text(content)
    if not text.strip():
        raise ParseError("JSON document is empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        if "\n" in text.strip():
            try:
                return parse_jsonl(content, name)
            except ParseError:
                pass
        raise ParseError(f"Invalid JSON: {exc.msg} (line {exc.lineno})") from exc
    return tables_from_json_value(data, name)


# --------------------------------------------------------------------------- CSV
def sniff_delimiter(sample: str) -> str:
    """Detect a CSV delimiter, defaulting to a comma."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=_CSV_DELIMITERS).delimiter
    except csv.Error:
        return ","


def parse_delimited(content: bytes, name: str, delimiter: str | None = None) -> tuple[ParsedTable, ...]:
    """Parse CSV/TSV text. All values are read as strings; typing happens in cleaning."""
    text = decode_text(content)
    if not text.strip():
        raise ParseError("Delimited document is empty")
    sep = delimiter or sniff_delimiter(text[:65536])
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=sep,
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=True,
            engine="python",
        )
    except (pd.errors.ParserError, ValueError) as exc:
        raise ParseError(f"Malformed delimited data: {exc}") from exc
    return (ParsedTable(name, frame),)


def parse_csv(content: bytes, name: str) -> tuple[ParsedTable, ...]:
    return parse_delimited(content, name)


def parse_tsv(content: bytes, name: str) -> tuple[ParsedTable, ...]:
    return parse_delimited(content, name, delimiter="\t")


# --------------------------------------------------------------------------- XLSX
def parse_xlsx(content: bytes, name: str) -> tuple[ParsedTable, ...]:
    """One table per non-empty worksheet; leading blank rows are skipped."""
    try:
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None, dtype=object)
    except Exception as exc:  # openpyxl/zipfile raise heterogeneous errors
        raise ParseError(f"Invalid XLSX workbook: {exc}") from exc
    tables: list[ParsedTable] = []
    for sheet_name, sheet in sheets.items():
        rows: list[list[Cell]] = sheet.astype(object).where(sheet.notna(), None).values.tolist()
        frame = rows_to_frame(rows)
        if not frame.empty or len(frame.columns):
            tables.append(ParsedTable(f"{name}.{sheet_name}", frame))
    return tuple(tables)


# --------------------------------------------------------------------------- XML
def _local_name(tag: object) -> str:
    text = str(tag)
    return text.rsplit("}", 1)[-1] if "}" in text else text


def _element_children(element: Any) -> list[Any]:
    return [child for child in element if isinstance(child.tag, str)]


def flatten_xml_element(element: Any, prefix: str = "") -> dict[str, str | None]:
    """Flatten attributes and nested children into dotted keys."""
    record: dict[str, str | None] = {}
    for attr, value in element.attrib.items():
        record[f"{prefix}@{_local_name(attr)}"] = value
    children = _element_children(element)
    if not children:
        text = (element.text or "").strip()
        if text or not record:
            record[prefix.rstrip(".") or _local_name(element.tag)] = text or None
        return record
    tag_counts = Counter(_local_name(child.tag) for child in children)
    occurrences: Counter[str] = Counter()
    for child in children:
        tag = _local_name(child.tag)
        occurrences[tag] += 1
        key = tag if tag_counts[tag] == 1 else f"{tag}_{occurrences[tag]}"
        record.update(flatten_xml_element(child, f"{prefix}{key}."))
    return record


def _find_record_container(root: Any) -> tuple[Any, str] | None:
    """Locate the element whose children repeat the same tag most often."""
    best: tuple[Any, str] | None = None
    best_count = 1
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        counts = Counter(_local_name(child.tag) for child in _element_children(element))
        if not counts:
            continue
        tag, count = counts.most_common(1)[0]
        if count > best_count:
            best, best_count = (element, tag), count
    return best


def parse_xml(content: bytes, name: str) -> tuple[ParsedTable, ...]:
    """Parse XML safely (no DTD loading, no entity expansion, no network)."""
    from lxml import etree

    if not content.strip():
        raise ParseError("XML document is empty")
    parser = etree.XMLParser(
        resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False, remove_comments=True
    )
    try:
        root = etree.fromstring(content, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ParseError(f"Malformed XML: {exc}") from exc
    container = _find_record_container(root)
    if container is None:
        return (ParsedTable(name, pd.DataFrame([flatten_xml_element(root)])),)
    element, tag = container
    records = [
        flatten_xml_element(child)
        for child in _element_children(element)
        if _local_name(child.tag) == tag
    ]
    return (ParsedTable(f"{name}.{tag}", pd.DataFrame(records)),)


# --------------------------------------------------------------------------- PDF
def parse_pdf(content: bytes, name: str) -> tuple[ParsedTable, ...]:
    """Extract ruled tables per page; fall back to a page/line text table."""
    import pdfplumber

    tables: list[ParsedTable] = []
    text_rows: list[dict[str, object]] = []
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                for table_index, raw in enumerate(page.extract_tables(), start=1):
                    frame = rows_to_frame(raw)
                    if not frame.empty:
                        tables.append(ParsedTable(f"{name}.page{page_number}.table{table_index}", frame))
                for line_number, line in enumerate((page.extract_text() or "").splitlines(), start=1):
                    if line.strip():
                        text_rows.append({"page": page_number, "line": line_number, "text": line})
    except Exception as exc:  # pdfminer raises many unrelated exception types
        raise ParseError(f"Invalid PDF: {exc}") from exc
    if tables:
        return tuple(tables)
    if text_rows:
        return (ParsedTable(f"{name}.text", pd.DataFrame(text_rows)),)
    return ()


# --------------------------------------------------------------------------- DOCX
def parse_docx(content: bytes, name: str) -> tuple[ParsedTable, ...]:
    """Extract Word tables; fall back to a paragraph table when there are none."""
    import docx

    try:
        document = docx.Document(io.BytesIO(content))
    except Exception as exc:  # python-docx raises zipfile/KeyError/ValueError variants
        raise ParseError(f"Invalid DOCX document: {exc}") from exc
    tables = [
        ParsedTable(f"{name}.table{index}", rows_to_frame([[cell.text for cell in row.cells] for row in table.rows]))
        for index, table in enumerate(document.tables, start=1)
    ]
    tables = [table for table in tables if not table.frame.empty]
    if tables:
        return tuple(tables)
    paragraphs = [
        {"paragraph": index, "style": paragraph.style.name if paragraph.style else None, "text": paragraph.text}
        for index, paragraph in enumerate(document.paragraphs, start=1)
        if paragraph.text.strip()
    ]
    return (ParsedTable(f"{name}.paragraphs", pd.DataFrame(paragraphs)),) if paragraphs else ()


# --------------------------------------------------------------------------- PPTX
def _iter_shapes(shapes: Any) -> Iterator[Any]:
    for shape in shapes:
        if hasattr(shape, "shapes"):  # group shape
            yield from _iter_shapes(shape.shapes)
        else:
            yield shape


def _chart_frame(chart: Any) -> pd.DataFrame:
    plot = chart.plots[0]
    data: dict[str, list[object]] = {"category": [str(c) for c in plot.categories]}
    for series in plot.series:
        data[str(series.name)] = list(series.values)
    return pd.DataFrame(data)


def parse_pptx(content: bytes, name: str) -> tuple[ParsedTable, ...]:
    """Extract tables and chart data per slide; fall back to slide text."""
    from pptx import Presentation

    try:
        presentation = Presentation(io.BytesIO(content))
    except Exception as exc:  # python-pptx raises zipfile/KeyError/ValueError variants
        raise ParseError(f"Invalid PPTX presentation: {exc}") from exc
    tables: list[ParsedTable] = []
    text_rows: list[dict[str, object]] = []
    for slide_number, slide in enumerate(presentation.slides, start=1):
        for shape_index, shape in enumerate(_iter_shapes(slide.shapes), start=1):
            if getattr(shape, "has_table", False):
                rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
                tables.append(ParsedTable(f"{name}.slide{slide_number}.table{shape_index}", rows_to_frame(rows)))
            elif getattr(shape, "has_chart", False):
                tables.append(ParsedTable(f"{name}.slide{slide_number}.chart{shape_index}", _chart_frame(shape.chart)))
            elif getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip():
                text_rows.append({"slide": slide_number, "shape": shape.name, "text": shape.text_frame.text})
    if tables:
        return tuple(tables)
    return (ParsedTable(f"{name}.text", pd.DataFrame(text_rows)),) if text_rows else ()


# --------------------------------------------------------------------------- PBIX
def _visual_rows(layout: Mapping[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for section in layout.get("sections", []):
        for visual in section.get("visualContainers", []):
            try:
                config = json.loads(visual.get("config", "{}"))
            except json.JSONDecodeError:
                config = {}
            single = config.get("singleVisual", {})
            fields = [
                projection.get("queryRef", "")
                for role in single.get("projections", {}).values()
                for projection in role
            ]
            rows.append(
                {
                    "page": section.get("displayName"),
                    "visual_type": single.get("visualType"),
                    "x": visual.get("x"),
                    "y": visual.get("y"),
                    "width": visual.get("width"),
                    "height": visual.get("height"),
                    "fields": ", ".join(field for field in fields if field) or None,
                }
            )
    return rows


def _pbixray_tables(content: bytes, name: str) -> list[ParsedTable]:
    """Read the VertiPaq data model through the optional ``pbixray`` package."""
    try:
        from pbixray import PBIXRay
    except ImportError:
        return []
    try:
        model = PBIXRay(io.BytesIO(content))
        return [ParsedTable(f"{name}.model.{table}", model.get_table(table)) for table in model.tables]
    except Exception:  # pbixray is best-effort: fall back to the report layout
        return []


def parse_pbix(content: bytes, name: str) -> tuple[ParsedTable, ...]:
    """Power BI: data-model tables (if ``pbixray`` is installed) + report visuals inventory.

    The PBIX data model is XPress9-compressed VertiPaq; without ``pbixray`` only the
    report layout (pages, visuals, bound fields) can be extracted.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ParseError(f"Invalid PBIX container: {exc}") from exc
    with archive:
        layout_bytes = archive.read("Report/Layout") if "Report/Layout" in archive.namelist() else None
    tables = _pbixray_tables(content, name)
    if layout_bytes is not None:
        try:
            layout = json.loads(layout_bytes.decode("utf-16-le").lstrip("﻿"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ParseError(f"Invalid PBIX report layout: {exc}") from exc
        tables.append(ParsedTable(f"{name}.visuals", pd.DataFrame(_visual_rows(layout))))
    if not tables:
        raise ParseError("PBIX contains no readable report layout or data model")
    return tuple(tables)


# --------------------------------------------------------------------------- registry
PARSERS: Final[Mapping[FileFormat, Parser]] = {
    FileFormat.JSON: parse_json,
    FileFormat.JSONL: parse_jsonl,
    FileFormat.CSV: parse_csv,
    FileFormat.TSV: parse_tsv,
    FileFormat.XLSX: parse_xlsx,
    FileFormat.XML: parse_xml,
    FileFormat.PDF: parse_pdf,
    FileFormat.DOCX: parse_docx,
    FileFormat.PPTX: parse_pptx,
    FileFormat.PBIX: parse_pbix,
}


def get_parser(file_format: FileFormat) -> Parser:
    try:
        return PARSERS[file_format]
    except KeyError as exc:  # pragma: no cover - every enum member is registered
        raise UnsupportedFormatError(f"No parser registered for {file_format.value}") from exc
