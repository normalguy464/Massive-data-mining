from __future__ import annotations

import argparse
import heapq
import re
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/raw/reddit/pushshift_1")
DEFAULT_OUTPUT = Path("data/processed/selected_submissions.parquet")
DEFAULT_TOP_N = 1000
DEFAULT_BATCH_SIZE = 50_000
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BOT_AUTHOR_PATTERN = re.compile(r"bot", re.IGNORECASE)
KNOWN_BOT_AUTHORS = {
    "automoderator",
    "remindmebot",
    "qualityvote",
    "bot_metric",
    "haikusbot",
    "wikitextbot",
    "vredditdownloader",
    "image_linker_bot",
    "trollabot",
    "bot",
}
REQUIRED_INPUT_COLUMNS = {
    "name",
    "subreddit",
    "author",
    "created_utc",
    "title",
    "selftext",
    "score",
    "upvote_ratio",
    "num_comments",
    "subreddit_subscribers",
    "domain",
}
OUTPUT_COLUMNS = [
    "name",
    "subreddit",
    "author",
    "created_utc",
    "title",
    "selftext",
    "score",
    "upvote_ratio",
    "num_comments",
    "subreddit_subscribers",
    "domain",
    "rank_in_subreddit",
]


def parse_args() -> argparse.Namespace:
    # Expose the main selection and filtering knobs via CLI so the script can
    # be reused with different raw inputs and thresholds.
    parser = argparse.ArgumentParser(
        description=(
            "Build selected_submissions directly from already-filtered raw submissions "
            "by keeping the top-N representative posts per subreddit."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input parquet file or directory containing raw submissions.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output parquet path for selected_submissions.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Maximum number of submissions to keep per subreddit.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Scanner batch size for parquet ingestion.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Minimum post score required before a post can compete for top-N.",
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=0,
        help="Minimum number of comments required before a post can compete for top-N.",
    )
    parser.add_argument(
        "--min-upvote-ratio",
        type=float,
        default=0.0,
        help="Minimum upvote_ratio required before a post can compete for top-N.",
    )
    parser.add_argument(
        "--min-title-chars",
        type=int,
        default=15,
        help="Minimum title length required before a post can compete for top-N.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    # Treat relative paths as project-root relative for consistent CLI usage.
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def dataset_from_input(path: Path) -> ds.Dataset:
    # Accept either a single parquet file or a directory tree of parquet files.
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        return ds.dataset(path, format="parquet")
    parquet_files = sorted(path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {path}")
    return ds.dataset(path, format="parquet")


def validate_required_columns(actual_columns: set[str], required_columns: set[str], label: str) -> None:
    # Fail early if the input schema does not match what downstream code expects.
    missing = sorted(required_columns - actual_columns)
    if missing:
        raise ValueError(f"Missing required columns in {label}: {', '.join(missing)}")


def normalize_text(value: Any, allow_empty: bool = False) -> str | None:
    # Standardize text fields by trimming whitespace and optionally allowing
    # empty strings for non-critical fields like selftext/domain.
    if value is None:
        return "" if allow_empty else None
    text = str(value).strip()
    if not text and not allow_empty:
        return None
    return text


def normalize_date(value: Any) -> str | None:
    # The script expects dates to already be materialized as YYYY-MM-DD strings.
    text = normalize_text(value)
    if text is None or not DATE_PATTERN.fullmatch(text):
        return None
    return text


def normalize_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def normalize_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def is_bot_author(author: str | None) -> bool:
    if author is None:
        return False
    normalized_author = author.strip().casefold()
    if not normalized_author:
        return False
    if normalized_author in KNOWN_BOT_AUTHORS:
        return True
    return bool(BOT_AUTHOR_PATTERN.search(normalized_author))


def build_record(payload: dict[str, list[Any]], idx: int) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    # Build one clean candidate record from the current batch row. If any
    # required field is missing/invalid, skip the row entirely.
    subreddit_raw = payload["subreddit"][idx]
    subreddit_name = normalize_text(subreddit_raw)
    if subreddit_name is None:
        return None

    name = normalize_text(payload["name"][idx])
    author = normalize_text(payload["author"][idx])
    created_utc = normalize_date(payload["created_utc"][idx])
    title = normalize_text(payload["title"][idx])
    score = normalize_int(payload["score"][idx])
    upvote_ratio = normalize_float(payload["upvote_ratio"][idx])
    num_comments = normalize_int(payload["num_comments"][idx])
    subreddit_subscribers = normalize_int(payload["subreddit_subscribers"][idx])

    if None in (
        name,
        author,
        created_utc,
        title,
        score,
        upvote_ratio,
        num_comments,
        subreddit_subscribers,
    ):
        return None
    if is_bot_author(author):
        return None

    selftext = normalize_text(payload["selftext"][idx], allow_empty=True)
    domain = normalize_text(payload["domain"][idx], allow_empty=True)
    record = {
        "name": name,
        "subreddit": subreddit_name,
        "author": author,
        "created_utc": created_utc,
        "title": title,
        "selftext": selftext if selftext is not None else "",
        "score": score,
        "upvote_ratio": upvote_ratio,
        "num_comments": num_comments,
        "subreddit_subscribers": subreddit_subscribers,
        "domain": domain if domain is not None else "",
    }
    # Rank posts primarily by score, then comments, then approval ratio, then
    # recency-ish/date string, with name as a deterministic tiebreaker.
    sort_key = (score, num_comments, upvote_ratio, created_utc, name)
    return sort_key, record


def collect_top_posts(
    dataset: ds.Dataset,
    top_n: int,
    batch_size: int,
    min_score: int,
    min_comments: int,
    min_upvote_ratio: float,
    min_title_chars: int,
) -> dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]]:
    validate_required_columns(set(dataset.schema.names), REQUIRED_INPUT_COLUMNS, "input dataset")
    heaps: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {}

    # Stream batches from parquet instead of loading the full dataset into memory.
    scanner = dataset.scanner(columns=OUTPUT_COLUMNS[:-1], batch_size=batch_size)
    for batch in scanner.to_batches():
        payload = batch.to_pydict()
        size = len(next(iter(payload.values()), []))
        for idx in range(size):
            built = build_record(payload, idx)
            if built is None:
                continue
            sort_key, record = built
            # Apply quality gates before a post is allowed to compete for the
            # per-subreddit top-N slots.
            if int(record["score"]) < min_score:
                continue
            if int(record["num_comments"]) < min_comments:
                continue
            if float(record["upvote_ratio"]) < min_upvote_ratio:
                continue
            if len(str(record["title"]).strip()) < min_title_chars:
                continue

            subreddit = str(record["subreddit"])
            heap = heaps.setdefault(subreddit, [])
            # Keep only the strongest N posts per subreddit by maintaining
            # a small min-heap of the current winners.
            if len(heap) < top_n:
                heapq.heappush(heap, (sort_key, record))
            elif sort_key > heap[0][0]:
                heapq.heapreplace(heap, (sort_key, record))

    return heaps


def finalize_rows(heaps: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subreddit in sorted(heaps):
        # Convert each subreddit heap into a fully ranked descending list.
        ranked = sorted(heaps[subreddit], key=lambda item: item[0], reverse=True)
        for rank, (_, record) in enumerate(ranked, start=1):
            row = dict(record)
            row["rank_in_subreddit"] = rank
            rows.append(row)
    return rows


def write_output(rows: list[dict[str, Any]], output_path: Path) -> None:
    # Persist the curated rows with a stable output column order.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    table = table.select(OUTPUT_COLUMNS)
    pq.write_table(table, output_path)


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    # Validate CLI parameters up front to avoid a long scan ending in a simple
    # configuration error.
    if args.top_n <= 0:
        raise ValueError("--top-n must be greater than 0.")
    if args.min_score < 0:
        raise ValueError("--min-score must be greater than or equal to 0.")
    if args.min_comments < 0:
        raise ValueError("--min-comments must be greater than or equal to 0.")
    if args.min_upvote_ratio < 0.0 or args.min_upvote_ratio > 1.0:
        raise ValueError("--min-upvote-ratio must be between 0 and 1.")
    if args.min_title_chars < 0:
        raise ValueError("--min-title-chars must be greater than or equal to 0.")

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    # Read, filter, rank, and write the final selected submissions dataset.
    dataset = dataset_from_input(input_path)
    heaps = collect_top_posts(
        dataset=dataset,
        top_n=args.top_n,
        batch_size=args.batch_size,
        min_score=args.min_score,
        min_comments=args.min_comments,
        min_upvote_ratio=args.min_upvote_ratio,
        min_title_chars=args.min_title_chars,
    )
    rows = finalize_rows(heaps)
    if not rows:
        raise RuntimeError("No selected submissions were produced from the provided inputs.")

    write_output(rows, output_path)
    non_empty_subreddits = sum(1 for heap in heaps.values() if heap)
    elapsed_seconds = time.perf_counter() - start_time
    print(
        f"Wrote {output_path} with {len(rows)} submissions across "
        f"{non_empty_subreddits} subreddits (top_n={args.top_n}, "
        f"min_score={args.min_score}, min_comments={args.min_comments}, "
        f"min_upvote_ratio={args.min_upvote_ratio}, min_title_chars={args.min_title_chars}). "
        f"Elapsed: {elapsed_seconds:.2f}s ({elapsed_seconds / 60.0:.2f}m)."
    )


if __name__ == "__main__":
    main()
