"""CLI: ``python -m dpu FILE [FILE ...] --out DIR``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from dpu.io_ops import export_tables
from dpu.logging_config import configure_logging
from dpu.models import DataProcessingError
from dpu.pipeline import process_file

logger = logging.getLogger("dpu.cli")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dpu", description="Convert files to cleaned CSV tables.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output"))
    args = parser.parse_args(argv)
    configure_logging(logging.INFO)
    exit_code = 0
    for path in args.files:
        try:
            export_tables(process_file(path), args.out)
        except (DataProcessingError, OSError) as exc:
            logger.error("%s: %s", path, exc)
            exit_code = 1
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
