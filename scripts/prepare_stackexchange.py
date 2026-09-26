"""Extract `data/stackexchange/clean_stackexchange.csv` from its byte-split 7z archive.

The clean StackExchange CSV (205,943 rows; columns Text, CreationDate,
Year, Label) is checked in as a byte-split 7z archive
(`clean_stackexchange.7z.001/.002/.003`) to stay under typical
file-size limits. This script concatenates the volumes into a single
`.7z` file in a temporary directory and extracts it with `py7zr`.
Idempotent: if the CSV is already present with the expected row count,
it does nothing.

Usage:

    python scripts/prepare_stackexchange.py [--force]
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import py7zr

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "stackexchange"
EXPECTED_ROWS = 205_943


def _volumes() -> list[Path]:
    volumes = sorted(DATA_DIR.glob("clean_stackexchange.7z.*"))
    if not volumes:
        raise FileNotFoundError(f"No 'clean_stackexchange.7z.NNN' volumes found under {DATA_DIR}")
    return volumes


def _row_count(csv_path: Path) -> int:
    # `Text` values contain embedded newlines, so counting raw lines
    # overcounts; `csv.reader` respects quoting and counts records correctly.
    with open(csv_path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus the header row


def prepare(force: bool = False) -> Path:
    """Extract the CSV if it's missing or the wrong size (or `force`); return its path."""
    csv_path = DATA_DIR / "clean_stackexchange.csv"

    if csv_path.exists() and not force:
        rows = _row_count(csv_path)
        if rows == EXPECTED_ROWS:
            return csv_path
        print(f"{csv_path} has {rows} rows, expected {EXPECTED_ROWS}; re-extracting.")

    volumes = _volumes()
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "clean_stackexchange.7z"
        with open(archive_path, "wb") as out:
            for volume in volumes:
                out.write(volume.read_bytes())

        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path=DATA_DIR)

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-extract even if the CSV already looks right"
    )
    args = parser.parse_args()

    csv_path = prepare(force=args.force)
    print(f"{csv_path}: {_row_count(csv_path)} rows")


if __name__ == "__main__":
    main()
