from __future__ import annotations

import argparse
import time

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from embedding_utils import (
    DEFAULT_INPUT,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MIN_CHAR_LEN,
    DEFAULT_MIN_WORD_LEN,
    DEFAULT_PREPARED_OUTPUT,
    DEFAULT_SCAN_BATCH_SIZE,
    DEFAULT_TOKENIZER,
    PREPARED_OUTPUT_COLUMNS,
    build_prepared_output_schema,
    coerce_float,
    coerce_int,
    coerce_string,
    dataset_from_input,
    is_deleted_marker,
    normalize_text,
    resolve_path,
    setup_logger,
    soft_truncate,
    validate_required_columns,
    validate_record,
)


REQUIRED_INPUT_COLUMNS = {
    "name",
    "subreddit",
    "title",
    "score",
    "num_comments",
    "rank_in_subreddit",
}
INPUT_COLUMNS = [
    "name",
    "subreddit",
    "title",
    "score",
    "num_comments",
    "rank_in_subreddit",
]
EMBEDDING_TEXT_SOURCE = "title_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build post_embedding_input directly from selected submissions by preparing "
            "title-only text for embedding."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input parquet file or directory containing selected submissions.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_PREPARED_OUTPUT),
        help="Output parquet path for prepared post_embedding_input.",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="Hugging Face tokenizer checkpoint used to measure token lengths.",
    )
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        default=DEFAULT_SCAN_BATCH_SIZE,
        help="Scanner batch size used for parquet ingestion.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Soft character cap applied before tokenization.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Tokenizer max_length used for truncation metadata.",
    )
    parser.add_argument(
        "--min-char-len",
        type=int,
        default=DEFAULT_MIN_CHAR_LEN,
        help="Minimum character length required after validation.",
    )
    parser.add_argument(
        "--min-word-len",
        type=int,
        default=DEFAULT_MIN_WORD_LEN,
        help="Minimum word count required after validation.",
    )
    return parser.parse_args()


def prepare_batch_rows_from_selected_submissions(
    *,
    payload: dict[str, list[object]],
    tokenizer: AutoTokenizer,
    max_chars: int,
    max_length: int,
    min_char_len: int,
    min_word_len: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows_without_tokens: list[dict[str, object]] = []
    batch_stats = {
        "seen_rows": 0,
        "kept_rows": 0,
        "dropped_invalid": 0,
        "char_truncated_rows": 0,
        "token_truncated_rows": 0,
    }

    size = len(next(iter(payload.values()), []))
    for idx in range(size):
        batch_stats["seen_rows"] += 1

        raw_name = coerce_string(payload["name"][idx])
        raw_subreddit = coerce_string(payload["subreddit"][idx])
        raw_title = coerce_string(payload["title"][idx])
        rank_in_subreddit = coerce_int(payload["rank_in_subreddit"][idx])
        score = coerce_float(payload["score"][idx])
        num_comments = coerce_int(payload["num_comments"][idx])

        if not validate_record(
            name=raw_name,
            subreddit=raw_subreddit,
            title_text=raw_title,
            rank_in_subreddit=rank_in_subreddit,
            score=score,
            num_comments=num_comments,
            char_len=len(raw_title.strip()) if raw_title is not None else None,
            word_len=len(raw_title.split()) if raw_title is not None else None,
            min_char_len=min_char_len,
            min_word_len=min_word_len,
        ):
            batch_stats["dropped_invalid"] += 1
            continue

        name = normalize_text(raw_name)
        subreddit = normalize_text(raw_subreddit)
        embedding_text = normalize_text(raw_title)
        if not name or not subreddit or not embedding_text or is_deleted_marker(embedding_text):
            batch_stats["dropped_invalid"] += 1
            continue

        word_len = len(embedding_text.split())
        char_len = len(embedding_text)
        if char_len < min_char_len or word_len < min_word_len:
            batch_stats["dropped_invalid"] += 1
            continue

        embedding_text, truncated_by_chars = soft_truncate(embedding_text, max_chars=max_chars)
        clean_char_len = len(embedding_text)
        word_len = len(embedding_text.split())
        if truncated_by_chars:
            batch_stats["char_truncated_rows"] += 1

        rows_without_tokens.append(
            {
                "name": name,
                "subreddit": subreddit,
                "embedding_text": embedding_text,
                "embedding_text_source": EMBEDDING_TEXT_SOURCE,
                "rank_in_subreddit": rank_in_subreddit,
                "score": score,
                "num_comments": num_comments,
                "char_len": clean_char_len,
                "word_len": word_len,
                "clean_char_len": clean_char_len,
                "truncated_by_chars": truncated_by_chars,
            }
        )

    if not rows_without_tokens:
        return [], batch_stats

    tokenized = tokenizer(
        [str(row["embedding_text"]) for row in rows_without_tokens],
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )

    prepared_rows: list[dict[str, object]] = []
    for row, input_ids in zip(rows_without_tokens, tokenized["input_ids"], strict=True):
        token_len_raw = len(input_ids)
        truncated_by_tokens = token_len_raw > max_length
        if truncated_by_tokens:
            batch_stats["token_truncated_rows"] += 1

        prepared_rows.append(
            {
                **row,
                "token_len": token_len_raw,
                "truncated_by_tokens": truncated_by_tokens,
                "is_valid_for_embedding": True,
            }
        )
        batch_stats["kept_rows"] += 1

    return prepared_rows, batch_stats


def write_prepared_output(
    *,
    input_path: str,
    output_path: str,
    tokenizer_name: str,
    scan_batch_size: int,
    max_chars: int,
    max_length: int,
    min_char_len: int,
    min_word_len: int,
) -> dict[str, int]:
    start_time = time.perf_counter()
    logger, log_path = setup_logger("step5_prepare_embeding_input")
    dataset = dataset_from_input(resolve_path(input_path))
    output_file = resolve_path(output_path)
    validate_required_columns(dataset.schema.names, REQUIRED_INPUT_COLUMNS, "input dataset")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.model_max_length = max(tokenizer.model_max_length, 10_000_000)
    scanner = dataset.scanner(columns=INPUT_COLUMNS, batch_size=scan_batch_size)
    output_schema = build_prepared_output_schema()
    stats = {
        "seen_rows": 0,
        "kept_rows": 0,
        "dropped_invalid": 0,
        "char_truncated_rows": 0,
        "token_truncated_rows": 0,
    }

    logger.info("Starting merged prepare step from %s", resolve_path(input_path))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for batch in scanner.to_batches():
            payload = batch.to_pydict()
            prepared_rows, batch_stats = prepare_batch_rows_from_selected_submissions(
                payload=payload,
                tokenizer=tokenizer,
                max_chars=max_chars,
                max_length=max_length,
                min_char_len=min_char_len,
                min_word_len=min_word_len,
            )
            for key, value in batch_stats.items():
                stats[key] += value
            if not prepared_rows:
                continue

            table = pa.Table.from_pylist(prepared_rows, schema=output_schema).select(PREPARED_OUTPUT_COLUMNS)
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if stats["kept_rows"] == 0:
        raise RuntimeError("No valid rows were produced for embedding input.")
    elapsed_seconds = time.perf_counter() - start_time
    stats["duration_seconds"] = elapsed_seconds
    logger.info(
        "Wrote prepared embedding input to %s with %s rows from %s scanned rows",
        output_file,
        stats["kept_rows"],
        stats["seen_rows"],
    )
    logger.info(
        "Prepare step finished in %.2f seconds (%.2f minutes)",
        elapsed_seconds,
        elapsed_seconds / 60.0,
    )
    logger.info("Step completed. Log file: %s", log_path)
    return stats


def main() -> None:
    args = parse_args()
    if args.scan_batch_size <= 0:
        raise ValueError("--scan-batch-size must be greater than 0.")
    if args.max_chars <= 0:
        raise ValueError("--max-chars must be greater than 0.")
    if args.max_length <= 0:
        raise ValueError("--max-length must be greater than 0.")
    if args.min_char_len <= 0:
        raise ValueError("--min-char-len must be greater than 0.")
    if args.min_word_len <= 0:
        raise ValueError("--min-word-len must be greater than 0.")

    stats = write_prepared_output(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        scan_batch_size=args.scan_batch_size,
        max_chars=args.max_chars,
        max_length=args.max_length,
        min_char_len=args.min_char_len,
        min_word_len=args.min_word_len,
    )

    print(
        "Prepared embedding input "
        f"at {resolve_path(args.output)} with {stats['kept_rows']} rows "
        f"from {stats['seen_rows']} scanned rows; "
        f"dropped_invalid={stats['dropped_invalid']}, "
        f"truncated_by_chars={stats['char_truncated_rows']}, "
        f"truncated_by_tokens={stats['token_truncated_rows']}, "
        f"elapsed={stats['duration_seconds']:.2f}s."
    )


if __name__ == "__main__":
    main()
