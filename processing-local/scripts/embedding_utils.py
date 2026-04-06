from __future__ import annotations

import logging
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/processed/selected_submissions.parquet")
DEFAULT_PREPARED_OUTPUT = Path("data/processed/post_embedding_input.parquet")
DEFAULT_EMBEDDING_OUTPUT = Path("data/embeddings/post_embeddings.parquet")
DEFAULT_SUBREDDIT_EMBEDDINGS_OUTPUT = Path("data/embeddings/subreddit_embeddings.parquet")
DEFAULT_SIMILARITY_EDGES_OUTPUT = Path("data/graph/subreddit_similarity_edges.parquet")
DEFAULT_PRUNED_SIMILARITY_EDGES_OUTPUT = Path("data/graph/subreddit_similarity_edges_pruned.parquet")
DEFAULT_SCAN_BATCH_SIZE = 5_000
DEFAULT_INFERENCE_BATCH_SIZE = 32
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_TOKENIZER = DEFAULT_MODEL_NAME
DEFAULT_MAX_CHARS = 2_000
DEFAULT_MAX_LENGTH = 256
DEFAULT_MIN_CHAR_LEN = 15
DEFAULT_MIN_WORD_LEN = 3
DEFAULT_SIMILARITY_PERCENTILE = 97.0
DEFAULT_SIMILARITY_BLOCK_SIZE = 512
LOGS_DIR = PROJECT_ROOT / "logs"
DELETED_MARKERS = {"[deleted]", "[removed]", "deleted", "removed"}
REQUIRED_INPUT_COLUMNS = {
    "name",
    "subreddit",
    "embedding_text",
    "embedding_text_source",
    "rank_in_subreddit",
    "score",
    "num_comments",
}
PREPARE_INPUT_COLUMNS = [
    "name",
    "subreddit",
    "embedding_text",
    "embedding_text_source",
    "rank_in_subreddit",
    "score",
    "num_comments",
]
PREPARED_OUTPUT_COLUMNS = [
    "name",
    "subreddit",
    "embedding_text",
    "embedding_text_source",
    "rank_in_subreddit",
    "score",
    "num_comments",
    "char_len",
    "word_len",
    "clean_char_len",
    "token_len",
    "truncated_by_chars",
    "truncated_by_tokens",
    "is_valid_for_embedding",
]
EMBEDDING_INPUT_COLUMNS = [
    "name",
    "subreddit",
    "embedding_text",
    "embedding_text_source",
    "rank_in_subreddit",
    "score",
    "num_comments",
    "char_len",
    "word_len",
    "clean_char_len",
    "token_len",
    "truncated_by_chars",
    "truncated_by_tokens",
]
EMBEDDING_OUTPUT_COLUMNS = [
    "name",
    "subreddit",
    "embedding_text",
    "embedding_text_source",
    "rank_in_subreddit",
    "score",
    "num_comments",
    "char_len",
    "word_len",
    "clean_char_len",
    "token_len",
    "token_count",
    "truncated_by_chars",
    "truncated_by_tokens",
    "model_name",
    "pooling_method",
    "embedding_dim",
    "embedding_norm",
    "embedding",
]
SUBREDDIT_EMBEDDING_COLUMNS = [
    "subreddit",
    "n_posts",
    "embedding_dim",
    "subreddit_embedding",
    "vector_norm",
    "avg_post_char_len",
    "avg_post_score",
]
SIMILARITY_EDGE_COLUMNS = [
    "source_subreddit",
    "target_subreddit",
    "cosine_similarity",
]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def setup_logger(script_name: str) -> tuple[logging.Logger, Path]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{script_name}_{timestamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logging initialized at %s", log_path)
    return logger, log_path


def dataset_from_input(path: Path) -> ds.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        return ds.dataset(path, format="parquet")
    parquet_files = sorted(path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {path}")
    return ds.dataset(path, format="parquet")


def validate_required_columns(actual_columns: Iterable[str], required_columns: set[str], label: str) -> None:
    missing = sorted(required_columns - set(actual_columns))
    if missing:
        raise ValueError(f"Missing required columns in {label}: {', '.join(missing)}")


def coerce_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    text = str(value).strip()
    if not text:
        return None
    parsed = float(text)
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    filtered_chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith("C") and char not in {"\n", "\t", " "}:
            continue
        filtered_chars.append(char)
    normalized = "".join(filtered_chars).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def is_deleted_marker(text: str) -> bool:
    return text.casefold() in DELETED_MARKERS


def soft_truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False

    candidate = text[: max_chars + 1]
    boundary = candidate.rfind(" ")
    if boundary >= max(1, int(max_chars * 0.8)):
        truncated = candidate[:boundary].rstrip()
    else:
        truncated = text[:max_chars].rstrip()
    return truncated, True


def validate_record(
    name: str | None,
    subreddit: str | None,
    title_text: str | None,
    rank_in_subreddit: int | None,
    score: float | None,
    num_comments: int | None,
    char_len: int | None,
    word_len: int | None,
    min_char_len: int,
    min_word_len: int,
) -> bool:
    if name is None or subreddit is None or title_text is None:
        return False
    if not name.strip() or not subreddit.strip():
        return False
    if not title_text.strip():
        return False
    lowered = title_text.strip().casefold()
    if lowered in DELETED_MARKERS:
        return False
    if rank_in_subreddit is None or num_comments is None or score is None:
        return False
    if char_len is None or word_len is None:
        return False
    if char_len <= 0 or word_len <= 0:
        return False
    if char_len < min_char_len or word_len < min_word_len:
        return False
    return True


def prepare_batch_rows(
    payload: dict[str, list[Any]],
    tokenizer: Any,
    max_chars: int,
    max_length: int,
    min_char_len: int,
    min_word_len: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows_without_tokens: list[dict[str, Any]] = []
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
        raw_embedding_text = coerce_string(payload["embedding_text"][idx])
        raw_embedding_text_source = coerce_string(payload["embedding_text_source"][idx])
        rank_in_subreddit = coerce_int(payload["rank_in_subreddit"][idx])
        score = coerce_float(payload["score"][idx])
        num_comments = coerce_int(payload["num_comments"][idx])

        if not validate_record(
            name=raw_name,
            subreddit=raw_subreddit,
            title_text=raw_embedding_text,
            rank_in_subreddit=rank_in_subreddit,
            score=score,
            num_comments=num_comments,
            char_len=len(raw_embedding_text.strip()) if raw_embedding_text is not None else None,
            word_len=len(raw_embedding_text.split()) if raw_embedding_text is not None else None,
            min_char_len=min_char_len,
            min_word_len=min_word_len,
        ):
            batch_stats["dropped_invalid"] += 1
            continue

        name = normalize_text(raw_name)
        subreddit = normalize_text(raw_subreddit)
        embedding_text = normalize_text(raw_embedding_text)
        embedding_text_source = normalize_text(raw_embedding_text_source or "title_only")

        if not name or not subreddit or not embedding_text or is_deleted_marker(embedding_text):
            batch_stats["dropped_invalid"] += 1
            continue

        clean_word_len = len(embedding_text.split())
        clean_char_len = len(embedding_text)
        if clean_char_len < min_char_len or clean_word_len < min_word_len:
            batch_stats["dropped_invalid"] += 1
            continue

        embedding_text, truncated_by_chars = soft_truncate(embedding_text, max_chars=max_chars)
        clean_char_len = len(embedding_text)
        clean_word_len = len(embedding_text.split())
        if truncated_by_chars:
            batch_stats["char_truncated_rows"] += 1

        rows_without_tokens.append(
            {
                "name": name,
                "subreddit": subreddit,
                "embedding_text": embedding_text,
                "embedding_text_source": embedding_text_source,
                "rank_in_subreddit": rank_in_subreddit,
                "score": score,
                "num_comments": num_comments,
                "char_len": clean_char_len,
                "word_len": clean_word_len,
                "clean_char_len": clean_char_len,
                "truncated_by_chars": truncated_by_chars,
            }
        )

    if not rows_without_tokens:
        return [], batch_stats

    tokenized = tokenizer(
        [row["embedding_text"] for row in rows_without_tokens],
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )

    prepared_rows: list[dict[str, Any]] = []
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


def build_prepared_output_schema() -> pa.Schema:
    return pa.schema(
        [
            ("name", pa.string()),
            ("subreddit", pa.string()),
            ("embedding_text", pa.string()),
            ("embedding_text_source", pa.string()),
            ("rank_in_subreddit", pa.int64()),
            ("score", pa.float64()),
            ("num_comments", pa.int64()),
            ("char_len", pa.int64()),
            ("word_len", pa.int64()),
            ("clean_char_len", pa.int64()),
            ("token_len", pa.int64()),
            ("truncated_by_chars", pa.bool_()),
            ("truncated_by_tokens", pa.bool_()),
            ("is_valid_for_embedding", pa.bool_()),
        ]
    )


def build_embedding_output_schema(embedding_dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("name", pa.string()),
            ("subreddit", pa.string()),
            ("embedding_text", pa.string()),
            ("embedding_text_source", pa.string()),
            ("rank_in_subreddit", pa.int64()),
            ("score", pa.float64()),
            ("num_comments", pa.int64()),
            ("char_len", pa.int64()),
            ("word_len", pa.int64()),
            ("clean_char_len", pa.int64()),
            ("token_len", pa.int64()),
            ("token_count", pa.int64()),
            ("truncated_by_chars", pa.bool_()),
            ("truncated_by_tokens", pa.bool_()),
            ("model_name", pa.string()),
            ("pooling_method", pa.string()),
            ("embedding_dim", pa.int64()),
            ("embedding_norm", pa.float64()),
            ("embedding", pa.list_(pa.float32())),
        ]
    )


def build_subreddit_embedding_schema(embedding_dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("subreddit", pa.string()),
            ("n_posts", pa.int64()),
            ("embedding_dim", pa.int64()),
            ("subreddit_embedding", pa.list_(pa.float32())),
            ("vector_norm", pa.float64()),
            ("avg_post_char_len", pa.float64()),
            ("avg_post_score", pa.float64()),
        ]
    )


def build_similarity_edge_schema() -> pa.Schema:
    return pa.schema(
        [
            ("source_subreddit", pa.string()),
            ("target_subreddit", pa.string()),
            ("cosine_similarity", pa.float64()),
        ]
    )
