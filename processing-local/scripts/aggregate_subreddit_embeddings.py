from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid

from embedding_utils import (
    DEFAULT_EMBEDDING_OUTPUT,
    DEFAULT_SUBREDDIT_EMBEDDINGS_OUTPUT,
    EMBEDDING_OUTPUT_COLUMNS,
    SUBREDDIT_EMBEDDING_COLUMNS,
    build_subreddit_embedding_schema,
    dataset_from_input,
    resolve_path,
    setup_logger,
    validate_required_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate post embeddings into one normalized vector per subreddit."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_EMBEDDING_OUTPUT),
        help="Input parquet file or directory containing post embeddings.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SUBREDDIT_EMBEDDINGS_OUTPUT),
        help="Output parquet path for subreddit embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Scanner batch size for streaming post embeddings.",
    )
    return parser.parse_args()


def l2_normalize(vector: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return np.zeros_like(vector, dtype=np.float32), 0.0
    return (vector / norm).astype(np.float32), norm


def resolve_existing_input_path(input_path: str) -> Path:
    candidate = resolve_path(input_path)
    if candidate.exists():
        return candidate

    candidate_name = candidate.name
    matches: list[Path] = []
    for root in (PROJECT_ROOT / "data", PROJECT_ROOT):
        if not root.exists():
            continue
        for match in root.rglob(candidate_name):
            if match.is_file():
                matches.append(match)

    unique_matches = sorted({match.resolve() for match in matches})
    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        formatted_matches = ", ".join(str(match) for match in unique_matches[:5])
        raise FileNotFoundError(
            f"Input path does not exist: {candidate}. Found multiple files named "
            f"{candidate_name}: {formatted_matches}. Please pass --input explicitly."
        )
    raise FileNotFoundError(
        f"Input path does not exist: {candidate}. No file named {candidate_name} "
        f"was found under {PROJECT_ROOT / 'data'}."
    )


def aggregate_subreddit_embeddings(
    *,
    input_path: str,
    output_path: str,
    batch_size: int,
) -> dict[str, int | float]:
    start_time = time.perf_counter()
    logger, log_path = setup_logger("step7_aggregate_subreddit_embeddings")
    input_file = resolve_existing_input_path(input_path)
    try:
        dataset = dataset_from_input(input_file)
    except ArrowInvalid as exc:
        raise RuntimeError(
            f"Input parquet is not readable: {input_file}. "
            "The file exists, but its parquet footer is missing or corrupted. "
            "Please point --input to a valid post_embeddings parquet file."
        ) from exc
    output_file = resolve_path(output_path)
    validate_required_columns(dataset.schema.names, set(EMBEDDING_OUTPUT_COLUMNS), "post embeddings dataset")

    scanner = dataset.scanner(columns=EMBEDDING_OUTPUT_COLUMNS, batch_size=batch_size)
    aggregates: dict[str, dict[str, object]] = {}
    embedding_dim: int | None = None
    seen_rows = 0

    logger.info("Starting subreddit aggregation from %s", input_file)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            seen_rows += 1
            subreddit = str(row["subreddit"])
            vector = np.asarray(row["embedding"], dtype=np.float32)
            row_embedding_dim = int(row["embedding_dim"])
            if embedding_dim is None:
                embedding_dim = row_embedding_dim
                logger.info("Detected embedding_dim=%s", embedding_dim)
            elif row_embedding_dim != embedding_dim:
                raise RuntimeError(
                    f"Inconsistent embedding_dim at row {seen_rows}: "
                    f"expected {embedding_dim}, found {row_embedding_dim}."
                )
            if vector.ndim != 1 or vector.shape[0] != embedding_dim:
                raise RuntimeError(
                    f"Invalid embedding length at row {seen_rows}: "
                    f"expected {embedding_dim}, found shape {tuple(vector.shape)}."
                )
            normalized_vector, _ = l2_normalize(vector)

            bucket = aggregates.setdefault(
                subreddit,
                {
                    "sum_vector": np.zeros(embedding_dim, dtype=np.float32),
                    "n_posts": 0,
                    "sum_char_len": 0.0,
                    "sum_score": 0.0,
                },
            )
            bucket["sum_vector"] = bucket["sum_vector"] + normalized_vector
            bucket["n_posts"] = int(bucket["n_posts"]) + 1
            bucket["sum_char_len"] = float(bucket["sum_char_len"]) + float(row["char_len"])
            bucket["sum_score"] = float(bucket["sum_score"]) + float(row["score"])

    if not aggregates or embedding_dim is None:
        raise RuntimeError("No subreddit embeddings could be aggregated from the input dataset.")

    rows: list[dict[str, object]] = []
    for subreddit, bucket in sorted(aggregates.items()):
        n_posts = int(bucket["n_posts"])
        mean_vector = np.asarray(bucket["sum_vector"], dtype=np.float32) / max(n_posts, 1)
        final_vector, mean_vector_norm = l2_normalize(mean_vector)
        rows.append(
            {
                "subreddit": subreddit,
                "n_posts": n_posts,
                "embedding_dim": embedding_dim,
                "subreddit_embedding": final_vector.tolist(),
                "vector_norm": mean_vector_norm,
                "avg_post_char_len": float(bucket["sum_char_len"]) / max(n_posts, 1),
                "avg_post_score": float(bucket["sum_score"]) / max(n_posts, 1),
            }
        )

    table = pa.Table.from_pylist(rows, schema=build_subreddit_embedding_schema(embedding_dim)).select(
        SUBREDDIT_EMBEDDING_COLUMNS
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_output_file = output_file.with_suffix(output_file.suffix + ".tmp")
    if temp_output_file.exists():
        temp_output_file.unlink()
    try:
        pq.write_table(table, temp_output_file)
        temp_output_file.replace(output_file)
    finally:
        if temp_output_file.exists():
            temp_output_file.unlink()
    elapsed_seconds = time.perf_counter() - start_time
    logger.info(
        "Wrote subreddit embeddings to %s with %s rows from %s post embeddings",
        output_file,
        table.num_rows,
        seen_rows,
    )
    logger.info(
        "Aggregation step finished in %.2f seconds (%.2f minutes)",
        elapsed_seconds,
        elapsed_seconds / 60.0,
    )
    logger.info("Step completed. Log file: %s", log_path)

    return {
        "seen_rows": seen_rows,
        "subreddit_rows": table.num_rows,
        "embedding_dim": embedding_dim,
        "duration_seconds": elapsed_seconds,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    stats = aggregate_subreddit_embeddings(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
    )
    print(
        "Wrote subreddit embeddings "
        f"with {stats['subreddit_rows']} rows from {stats['seen_rows']} post embeddings; "
        f"embedding_dim={stats['embedding_dim']}, "
        f"elapsed={stats['duration_seconds']:.2f}s."
    )


if __name__ == "__main__":
    main()
