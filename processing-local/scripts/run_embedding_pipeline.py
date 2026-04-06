from __future__ import annotations

import argparse
import time

from embedding_utils import (
    DEFAULT_EMBEDDING_OUTPUT,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_PREPARED_OUTPUT,
    DEFAULT_TOKENIZER,
    resolve_path,
    setup_logger,
)
from embed_posts import write_post_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the embedding step only from an already prepared embedding-input parquet."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_PREPARED_OUTPUT),
        help="Input parquet file or directory containing prepared embedding input.",
    )
    parser.add_argument(
        "--embedding-output",
        default=str(DEFAULT_EMBEDDING_OUTPUT),
        help="Output parquet path for post-level embeddings.",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--inference-batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    total_start_time = time.perf_counter()
    args = parse_args()
    logger, log_path = setup_logger("run_embedding_pipeline")

    embedding_start = time.perf_counter()
    embedding_stats = write_post_embeddings(
        input_path=args.input,
        output_path=args.embedding_output,
        tokenizer_name=args.tokenizer,
        model_name=args.model_name,
        inference_batch_size=args.inference_batch_size,
        max_length=args.max_length,
        device_name=args.device,
    )
    embedding_elapsed = time.perf_counter() - embedding_start
    logger.info(
        "Embedding phase finished in %.2f seconds (%.2f minutes) and wrote %s rows to %s",
        embedding_elapsed,
        embedding_elapsed / 60.0,
        embedding_stats["kept_rows"],
        resolve_path(args.embedding_output),
    )

    total_elapsed = time.perf_counter() - total_start_time
    logger.info(
        "Embedding pipeline finished in %.2f seconds (%.2f minutes). Log file: %s",
        total_elapsed,
        total_elapsed / 60.0,
        log_path,
    )

    print(
        "Wrote post embeddings "
        f"with {embedding_stats['kept_rows']} embedded rows "
        f"from prepared input {resolve_path(args.input)}; "
        f"embed_elapsed={embedding_elapsed:.2f}s, "
        f"total_elapsed={total_elapsed:.2f}s."
    )


if __name__ == "__main__":
    main()
