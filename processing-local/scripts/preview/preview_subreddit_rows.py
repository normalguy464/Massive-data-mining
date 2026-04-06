from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("data/processed/selected_submissions.parquet")
DEFAULT_LIMIT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the first N rows for a subreddit from a parquet dataset."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input parquet file or directory.",
    )
    parser.add_argument(
        "--subreddit",
        required=True,
        help="Subreddit name to preview.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of rows to print.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def dataset_from_input(path: Path) -> ds.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        return ds.dataset(path, format="parquet")
    parquet_files = sorted(path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {path}")
    return ds.dataset(path, format="parquet")


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")

    input_path = resolve_path(args.input)
    dataset = dataset_from_input(input_path)
    if "subreddit" not in set(dataset.schema.names):
        raise ValueError(f"Input dataset does not contain a 'subreddit' column: {input_path}")

    filter_expr = ds.field("subreddit") == args.subreddit
    table = dataset.to_table(filter=filter_expr).slice(0, args.limit)
    rows = table.to_pylist()
    if not rows:
        raise RuntimeError(f"No rows found for subreddit={args.subreddit!r} in {input_path}")

    print(
        f"Showing {len(rows)} row(s) for subreddit={args.subreddit!r} "
        f"from {input_path}:"
    )
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}] {json.dumps(row, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
