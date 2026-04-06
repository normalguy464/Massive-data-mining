from __future__ import annotations

import argparse
import math
import time
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from embedding_utils import (
    DEFAULT_PRUNED_SIMILARITY_EDGES_OUTPUT,
    DEFAULT_SIMILARITY_BLOCK_SIZE,
    DEFAULT_SIMILARITY_EDGES_OUTPUT,
    DEFAULT_SIMILARITY_PERCENTILE,
    DEFAULT_SUBREDDIT_EMBEDDINGS_OUTPUT,
    SIMILARITY_EDGE_COLUMNS,
    SUBREDDIT_EMBEDDING_COLUMNS,
    build_similarity_edge_schema,
    dataset_from_input,
    resolve_path,
    setup_logger,
    validate_required_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cosine-similarity edges between subreddit embeddings using "
            "a blockwise streaming pass, then prune by percentile."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_SUBREDDIT_EMBEDDINGS_OUTPUT),
        help="Input parquet path for subreddit embeddings.",
    )
    parser.add_argument(
        "--output-all",
        default=str(DEFAULT_SIMILARITY_EDGES_OUTPUT),
        help="Output parquet path for all subreddit similarity edges.",
    )
    parser.add_argument(
        "--output-pruned",
        default=str(DEFAULT_PRUNED_SIMILARITY_EDGES_OUTPUT),
        help="Output parquet path for pruned subreddit similarity edges.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_SIMILARITY_PERCENTILE,
        help="Percentile threshold used to prune edges by cosine similarity.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_SIMILARITY_BLOCK_SIZE,
        help="Number of subreddit vectors processed per block during pairwise similarity scanning.",
    )
    parser.add_argument(
        "--skip-output-all",
        action="store_true",
        help="Skip writing the full all-pairs edge list and only persist the pruned graph.",
    )
    return parser.parse_args()


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


def iter_score_blocks(embeddings: np.ndarray, block_size: int) -> Iterator[np.ndarray]:
    n_rows = embeddings.shape[0]
    for left_start in range(0, n_rows, block_size):
        left_end = min(left_start + block_size, n_rows)
        left_block = embeddings[left_start:left_end]

        same_block_scores = left_block @ left_block.T
        tri_i, tri_j = np.triu_indices(left_end - left_start, k=1)
        if tri_i.size:
            yield same_block_scores[tri_i, tri_j].astype(np.float32, copy=False)

        for right_start in range(left_end, n_rows, block_size):
            right_end = min(right_start + block_size, n_rows)
            right_block = embeddings[right_start:right_end]
            cross_scores = left_block @ right_block.T
            yield cross_scores.reshape(-1).astype(np.float32, copy=False)


def iter_edge_blocks(
    embeddings: np.ndarray,
    block_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n_rows = embeddings.shape[0]
    for left_start in range(0, n_rows, block_size):
        left_end = min(left_start + block_size, n_rows)
        left_block = embeddings[left_start:left_end]

        same_block_scores = left_block @ left_block.T
        tri_i, tri_j = np.triu_indices(left_end - left_start, k=1)
        if tri_i.size:
            yield (
                (tri_i + left_start).astype(np.int32, copy=False),
                (tri_j + left_start).astype(np.int32, copy=False),
                same_block_scores[tri_i, tri_j].astype(np.float32, copy=False),
            )

        for right_start in range(left_end, n_rows, block_size):
            right_end = min(right_start + block_size, n_rows)
            right_block = embeddings[right_start:right_end]
            cross_scores = left_block @ right_block.T
            left_indices = np.repeat(np.arange(left_start, left_end, dtype=np.int32), right_end - right_start)
            right_indices = np.tile(np.arange(right_start, right_end, dtype=np.int32), left_end - left_start)
            yield (
                left_indices,
                right_indices,
                cross_scores.reshape(-1).astype(np.float32, copy=False),
            )


def count_total_pairs(n_rows: int) -> int:
    return n_rows * (n_rows - 1) // 2


def compute_keep_count(total_pairs: int, percentile: float) -> int:
    if total_pairs <= 0:
        return 0
    tail_fraction = max(0.0, min(1.0, (100.0 - percentile) / 100.0))
    if tail_fraction <= 0.0:
        return 1
    keep_count = int(math.ceil(total_pairs * tail_fraction))
    return min(total_pairs, max(1, keep_count))


def compute_threshold_streaming(
    embeddings: np.ndarray,
    percentile: float,
    block_size: int,
    logger: object,
) -> tuple[float, int, int]:
    total_pairs = count_total_pairs(int(embeddings.shape[0]))
    if total_pairs <= 0:
        raise RuntimeError("At least two subreddit embeddings are required to build a similarity graph.")

    keep_count = compute_keep_count(total_pairs, percentile)
    retained_top_scores = np.empty(0, dtype=np.float32)
    min_seen_score: float | None = None
    processed_pairs = 0

    logger.info(
        "Computing streaming similarity threshold for %s total pairs with percentile=%.2f and keep_count=%s",
        total_pairs,
        percentile,
        keep_count,
    )

    for score_block in iter_score_blocks(embeddings, block_size):
        if score_block.size == 0:
            continue
        processed_pairs += int(score_block.size)
        block_min = float(np.min(score_block))
        min_seen_score = block_min if min_seen_score is None else min(min_seen_score, block_min)

        combined = score_block if retained_top_scores.size == 0 else np.concatenate((retained_top_scores, score_block))
        if combined.size > keep_count:
            split_index = combined.size - keep_count
            retained_top_scores = np.partition(combined, split_index)[split_index:]
        else:
            retained_top_scores = combined

    if processed_pairs != total_pairs:
        raise RuntimeError(
            f"Streaming threshold scan mismatch: expected {total_pairs} pairs but processed {processed_pairs}."
        )
    if retained_top_scores.size == 0 or min_seen_score is None:
        raise RuntimeError("No similarity scores were produced during threshold computation.")

    threshold = float(np.min(retained_top_scores))
    if keep_count >= total_pairs:
        threshold = float(min_seen_score)

    logger.info(
        "Finished threshold scan with processed_pairs=%s, threshold=%.6f",
        processed_pairs,
        threshold,
    )
    return threshold, keep_count, total_pairs


def build_edge_table(
    subreddit_names: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    scores: np.ndarray,
    schema: pa.Schema,
) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "source_subreddit": subreddit_names[source_indices].tolist(),
            "target_subreddit": subreddit_names[target_indices].tolist(),
            "cosine_similarity": scores.astype(np.float64, copy=False),
        },
        schema=schema,
    ).select(SIMILARITY_EDGE_COLUMNS)


def write_similarity_edges(
    *,
    subreddit_names: np.ndarray,
    embeddings: np.ndarray,
    output_all_path: str,
    output_pruned_path: str,
    threshold: float,
    block_size: int,
    skip_output_all: bool,
    logger: object,
) -> tuple[int, int]:
    schema = build_similarity_edge_schema()
    output_all_file = resolve_path(output_all_path)
    output_pruned_file = resolve_path(output_pruned_path)
    output_all_file.parent.mkdir(parents=True, exist_ok=True)
    output_pruned_file.parent.mkdir(parents=True, exist_ok=True)

    all_writer: pq.ParquetWriter | None = None
    pruned_writer: pq.ParquetWriter | None = None
    all_edge_count = 0
    pruned_edge_count = 0

    try:
        for source_indices, target_indices, scores in iter_edge_blocks(embeddings, block_size):
            if scores.size == 0:
                continue

            if not skip_output_all:
                all_table = build_edge_table(subreddit_names, source_indices, target_indices, scores, schema)
                if all_writer is None:
                    all_writer = pq.ParquetWriter(output_all_file, all_table.schema)
                all_writer.write_table(all_table)
                all_edge_count += int(all_table.num_rows)

            keep_mask = scores >= threshold
            if np.any(keep_mask):
                pruned_table = build_edge_table(
                    subreddit_names,
                    source_indices[keep_mask],
                    target_indices[keep_mask],
                    scores[keep_mask],
                    schema,
                )
                if pruned_writer is None:
                    pruned_writer = pq.ParquetWriter(output_pruned_file, pruned_table.schema)
                pruned_writer.write_table(pruned_table)
                pruned_edge_count += int(pruned_table.num_rows)
    finally:
        if all_writer is not None:
            all_writer.close()
        if pruned_writer is not None:
            pruned_writer.close()

    if pruned_edge_count == 0:
        raise RuntimeError("No similarity edges met the pruning threshold.")

    if skip_output_all:
        logger.info("Skipped writing the full all-pairs edge list.")
    else:
        logger.info("Wrote all similarity edges to %s with %s rows", output_all_file, all_edge_count)
    logger.info(
        "Wrote pruned similarity edges to %s with %s rows at threshold=%.6f",
        output_pruned_file,
        pruned_edge_count,
        threshold,
    )
    return all_edge_count, pruned_edge_count


def build_similarity_graph(
    *,
    input_path: str,
    output_all_path: str,
    output_pruned_path: str,
    percentile: float,
    block_size: int,
    skip_output_all: bool,
) -> dict[str, float | int]:
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("--percentile must be between 0 and 100.")
    if block_size <= 0:
        raise ValueError("--block-size must be greater than 0.")

    start_time = time.perf_counter()
    logger, log_path = setup_logger("step8_build_similarity_graph")
    dataset = dataset_from_input(resolve_path(input_path))
    validate_required_columns(dataset.schema.names, set(SUBREDDIT_EMBEDDING_COLUMNS), "subreddit embeddings dataset")

    table = dataset.to_table(columns=SUBREDDIT_EMBEDDING_COLUMNS)
    rows = table.to_pylist()
    if len(rows) < 2:
        raise RuntimeError("At least two subreddit embeddings are required to build a similarity graph.")

    subreddit_names = np.asarray([str(row["subreddit"]) for row in rows], dtype=object)
    embeddings = np.asarray([row["subreddit_embedding"] for row in rows], dtype=np.float32)
    embeddings = l2_normalize_rows(embeddings)

    logger.info(
        "Loaded %s subreddit embeddings from %s using block_size=%s",
        len(rows),
        resolve_path(input_path),
        block_size,
    )
    threshold, keep_count, total_pairs = compute_threshold_streaming(
        embeddings=embeddings,
        percentile=percentile,
        block_size=block_size,
        logger=logger,
    )
    all_edge_count, pruned_edge_count = write_similarity_edges(
        subreddit_names=subreddit_names,
        embeddings=embeddings,
        output_all_path=output_all_path,
        output_pruned_path=output_pruned_path,
        threshold=threshold,
        block_size=block_size,
        skip_output_all=skip_output_all,
        logger=logger,
    )

    elapsed_seconds = time.perf_counter() - start_time
    logger.info(
        "Similarity step finished in %.2f seconds (%.2f minutes)",
        elapsed_seconds,
        elapsed_seconds / 60.0,
    )
    logger.info("Step completed. Log file: %s", log_path)

    return {
        "subreddit_count": len(rows),
        "all_edge_count": all_edge_count,
        "pruned_edge_count": pruned_edge_count,
        "percentile": percentile,
        "threshold": threshold,
        "keep_count": keep_count,
        "total_pairs": total_pairs,
        "duration_seconds": elapsed_seconds,
    }


def main() -> None:
    args = parse_args()
    stats = build_similarity_graph(
        input_path=args.input,
        output_all_path=args.output_all,
        output_pruned_path=args.output_pruned,
        percentile=args.percentile,
        block_size=args.block_size,
        skip_output_all=args.skip_output_all,
    )
    print(
        "Built similarity graph "
        f"for {stats['subreddit_count']} subreddits with {stats['pruned_edge_count']} pruned edges "
        f"from {stats['total_pairs']} total pairs at percentile={stats['percentile']}; "
        f"threshold={stats['threshold']:.6f}, "
        f"elapsed={stats['duration_seconds']:.2f}s."
    )


if __name__ == "__main__":
    main()
