from __future__ import annotations

import argparse
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

from embedding_utils import (
    DEFAULT_EMBEDDING_OUTPUT,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_PREPARED_OUTPUT,
    DEFAULT_TOKENIZER,
    EMBEDDING_INPUT_COLUMNS,
    EMBEDDING_OUTPUT_COLUMNS,
    build_embedding_output_schema,
    dataset_from_input,
    resolve_path,
    setup_logger,
    validate_required_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DistilBERT embeddings for each prepared Reddit post."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_PREPARED_OUTPUT),
        help="Input parquet file or directory containing prepared embedding input.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_EMBEDDING_OUTPUT),
        help="Output parquet path for post-level embeddings.",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="Hugging Face tokenizer checkpoint used for model input.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face encoder checkpoint used to produce post embeddings.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
        help="Mini-batch size used for DistilBERT inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Tokenizer max_length used during embedding inference.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device for embedding inference.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is available.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked_hidden = last_hidden_state * mask
    pooled = masked_hidden.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return pooled / counts


def embed_rows(
    rows: list[dict[str, object]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    model_name: str,
    inference_batch_size: int,
    max_length: int,
    device: torch.device,
    embedding_dim: int,
) -> list[dict[str, object]]:
    embedded_rows: list[dict[str, object]] = []
    for start in range(0, len(rows), inference_batch_size):
        chunk = rows[start : start + inference_batch_size]
        encoded = tokenizer(
            [str(row["embedding_text"]) for row in chunk],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

        pooled_cpu = pooled.to(dtype=torch.float32).cpu()
        embedding_norms = torch.linalg.vector_norm(pooled_cpu, dim=1).tolist()
        pooled_list = pooled_cpu.tolist()
        token_counts = encoded["attention_mask"].sum(dim=1).to("cpu").tolist()
        for row, embedding, token_count, embedding_norm in zip(
            chunk,
            pooled_list,
            token_counts,
            embedding_norms,
            strict=True,
        ):
            embedded_rows.append(
                {
                    "name": row["name"],
                    "subreddit": row["subreddit"],
                    "embedding_text": row["embedding_text"],
                    "embedding_text_source": row["embedding_text_source"],
                    "rank_in_subreddit": row["rank_in_subreddit"],
                    "score": row["score"],
                    "num_comments": row["num_comments"],
                    "char_len": row["char_len"],
                    "word_len": row["word_len"],
                    "clean_char_len": row["clean_char_len"],
                    "token_len": row["token_len"],
                    "token_count": int(token_count),
                    "truncated_by_chars": row["truncated_by_chars"],
                    "truncated_by_tokens": row["truncated_by_tokens"],
                    "model_name": model_name,
                    "pooling_method": "mean_pooling",
                    "embedding_dim": embedding_dim,
                    "embedding_norm": embedding_norm,
                    "embedding": embedding,
                }
            )
    return embedded_rows


def write_post_embeddings(
    *,
    input_path: str,
    output_path: str,
    tokenizer_name: str,
    model_name: str,
    inference_batch_size: int,
    max_length: int,
    device_name: str,
) -> dict[str, object]:
    start_time = time.perf_counter()
    logger, log_path = setup_logger("step6_embed_posts")
    dataset = dataset_from_input(resolve_path(input_path))
    output_file = resolve_path(output_path)
    validate_required_columns(dataset.schema.names, set(EMBEDDING_INPUT_COLUMNS), "prepared dataset")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    device = resolve_device(device_name)
    model.to(device)
    model.eval()
    embedding_dim = int(getattr(model.config, "hidden_size"))
    output_schema = build_embedding_output_schema(embedding_dim)
    scanner = dataset.scanner(columns=EMBEDDING_INPUT_COLUMNS, batch_size=max(inference_batch_size * 16, 128))
    total_rows = dataset.count_rows()
    progress_log_every_rows = max(inference_batch_size * 1_000, 100_000)

    stats: dict[str, object] = {
        "seen_rows": 0,
        "kept_rows": 0,
        "embedding_dim": embedding_dim,
        "device": str(device),
        "total_rows": total_rows,
    }

    logger.info(
        "Starting post embedding inference from %s using model=%s tokenizer=%s",
        resolve_path(input_path),
        model_name,
        tokenizer_name,
    )
    logger.info(
        "Prepared dataset has %s rows; inference_batch_size=%s, scanner_batch_size=%s, device=%s",
        total_rows,
        inference_batch_size,
        max(inference_batch_size * 16, 128),
        device,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    next_progress_log = progress_log_every_rows
    try:
        for batch in scanner.to_batches():
            rows = batch.to_pylist()
            if not rows:
                continue
            stats["seen_rows"] += len(rows)

            embedded_rows = embed_rows(
                rows=rows,
                tokenizer=tokenizer,
                model=model,
                model_name=model_name,
                inference_batch_size=inference_batch_size,
                max_length=max_length,
                device=device,
                embedding_dim=embedding_dim,
            )
            table = pa.Table.from_pylist(embedded_rows, schema=output_schema).select(EMBEDDING_OUTPUT_COLUMNS)
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema)
            writer.write_table(table)
            stats["kept_rows"] += len(embedded_rows)

            kept_rows = int(stats["kept_rows"])
            if kept_rows >= next_progress_log or kept_rows == total_rows:
                elapsed_seconds = time.perf_counter() - start_time
                rows_per_second = kept_rows / elapsed_seconds if elapsed_seconds > 0 else 0.0
                remaining_rows = max(total_rows - kept_rows, 0)
                eta_seconds = remaining_rows / rows_per_second if rows_per_second > 0 else 0.0
                logger.info(
                    "Progress: %s/%s rows (%.2f%%) at %.2f rows/s; elapsed=%.2f min; eta=%.2f min",
                    kept_rows,
                    total_rows,
                    (kept_rows / total_rows * 100.0) if total_rows else 100.0,
                    rows_per_second,
                    elapsed_seconds / 60.0,
                    eta_seconds / 60.0,
                )
                next_progress_log += progress_log_every_rows
    finally:
        if writer is not None:
            writer.close()

    if int(stats["kept_rows"]) == 0:
        raise RuntimeError("No embeddings were produced from the prepared dataset.")
    elapsed_seconds = time.perf_counter() - start_time
    stats["duration_seconds"] = elapsed_seconds
    logger.info(
        "Wrote post embeddings to %s with %s rows from %s prepared rows",
        output_file,
        stats["kept_rows"],
        stats["seen_rows"],
    )
    logger.info(
        "Embedding step finished in %.2f seconds (%.2f minutes)",
        elapsed_seconds,
        elapsed_seconds / 60.0,
    )
    logger.info("Step completed. Log file: %s", log_path)
    return stats


def main() -> None:
    args = parse_args()
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be greater than 0.")
    if args.max_length <= 0:
        raise ValueError("--max-length must be greater than 0.")

    stats = write_post_embeddings(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        model_name=args.model_name,
        inference_batch_size=args.inference_batch_size,
        max_length=args.max_length,
        device_name=args.device,
    )

    print(
        "Wrote post embeddings "
        f"to {resolve_path(args.output)} with {stats['kept_rows']} rows "
        f"from {stats['seen_rows']} prepared rows; "
        f"embedding_dim={stats['embedding_dim']}, "
        f"device={stats['device']}, "
        f"elapsed={stats['duration_seconds']:.2f}s."
    )


if __name__ == "__main__":
    main()
