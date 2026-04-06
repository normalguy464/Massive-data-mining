from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("data/raw/reddit/pushshift_1/0sanitymemes_submissions_cleaned.parquet")
DEFAULT_LIMIT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read and preview a single raw Reddit parquet file."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to a single raw parquet file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of rows to preview.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")

    input_path = resolve_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"--input must point to a single parquet file: {input_path}")
    if input_path.suffix.lower() != ".parquet":
        raise ValueError(f"--input must be a parquet file: {input_path}")

    # Read just one raw parquet file and preview a small slice for inspection.
    parquet_file = pq.ParquetFile(input_path)
    schema = parquet_file.schema_arrow
    table = parquet_file.read().slice(0, args.limit)
    rows = table.to_pylist()

    print(f"File: {input_path}")
    print(f"Row count: {parquet_file.metadata.num_rows}")
    print(f"Column count: {len(schema.names)}")
    print("Schema:")
    for field in schema:
        print(f"  - {field.name}: {field.type}")

    print(f"\nPreviewing first {len(rows)} row(s):")
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}] {json.dumps(row, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
