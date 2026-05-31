from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from typing import Dict, List, Any

import requests

from chat_with_llm_api import (
    post_chat,
    extract_assistant_text,
    DEFAULT_BASE_URL,
    DEFAULT_API_KEY,
    DEFAULT_API_KEY_HEADER,
)

REQUEST_TIMEOUT = 300
MAX_RETRIES = 6


def read_clusters(csv_path: str, top_n: int) -> Dict[str, List[str]]:
    clusters: Dict[str, List[str]] = {}

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            subreddit = row.get("subreddit") or row.get("Subreddit")
            community = (
                row.get("community_id")
                or row.get("community")
                or row.get("cluster")
            )

            if not subreddit or not community:
                continue

            clusters.setdefault(community, []).append(subreddit)

    for k, v in list(clusters.items()):
        clusters[k] = v[:top_n]

    return clusters


def build_prompt_for_cluster(cluster_id: str, subreddits: List[str]) -> str:
    lines = [
        "Dưới đây là danh sách subreddit thuộc cùng một cluster.",
        "",
        f"Cluster ID: {cluster_id}",
        "",
        "Subreddits:",
    ]

    lines.extend([f"- {s}" for s in subreddits])

    lines.extend(
        [
            "",
            "Hãy:",
            "1. Đặt tên ngắn gọn cho cluster (3-5 từ)",
            "2. Giải thích ngắn gọn",
            "",
            "CHỈ trả về JSON hợp lệ.",
            "",
            'Ví dụ:',
            '{"name":"Game PC","reason":"Các subreddit liên quan gaming PC"}',
        ]
    )

    return "\n".join(lines)


def clean_json_response(text: str) -> str:
    """
    Remove markdown code fences and surrounding garbage.
    """

    text = text.strip()

    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

    text = text.strip()

    # Try extracting first JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        return match.group(0)

    return text


def call_llm_with_retry(
    base_url: str,
    api_key: str,
    api_key_header: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:

    last_exception = None

    for attempt in range(MAX_RETRIES):

        try:
            resp = post_chat(
                base_url,
                api_key,
                api_key_header,
                payload,
                timeout=REQUEST_TIMEOUT,
            )

            if not isinstance(resp, dict):
                raise ValueError("LLM response is not JSON dict")

            return resp

        except requests.exceptions.Timeout as exc:
            last_exception = exc

            sleep_time = min(60, 2 ** attempt)

            print(
                f"[Retry {attempt+1}/{MAX_RETRIES}] Timeout. Sleep {sleep_time}s...",
                file=sys.stderr,
            )

            time.sleep(sleep_time)

        except requests.exceptions.ConnectionError as exc:
            last_exception = exc

            sleep_time = min(60, 2 ** attempt)

            print(
                f"[Retry {attempt+1}/{MAX_RETRIES}] Connection error. Sleep {sleep_time}s...",
                file=sys.stderr,
            )

            time.sleep(sleep_time)

        except requests.exceptions.HTTPError as exc:
            last_exception = exc

            status = None

            try:
                status = exc.response.status_code
            except Exception:
                pass

            if status == 404:
                raise RuntimeError(
                    f"404 endpoint not found: {base_url}"
                ) from exc

            if status == 401:
                raise RuntimeError(
                    "401 unauthorized: invalid API key"
                ) from exc

            if status == 429:
                retry_after = 5

                try:
                    retry_after = int(
                        exc.response.headers.get("Retry-After", "5")
                    )
                except Exception:
                    pass

                print(
                    f"[Retry {attempt+1}/{MAX_RETRIES}] Rate limited. Sleep {retry_after}s...",
                    file=sys.stderr,
                )

                time.sleep(retry_after)
                continue

            if status in {500, 502, 503, 504}:
                sleep_time = min(60, 2 ** attempt)

                print(
                    f"[Retry {attempt+1}/{MAX_RETRIES}] Server error {status}. Sleep {sleep_time}s...",
                    file=sys.stderr,
                )

                time.sleep(sleep_time)
                continue

            raise

        except Exception as exc:
            last_exception = exc

            sleep_time = min(60, 2 ** attempt)

            print(
                f"[Retry {attempt+1}/{MAX_RETRIES}] Unknown error: {exc}",
                file=sys.stderr,
            )

            time.sleep(sleep_time)

    raise RuntimeError(f"LLM failed after retries: {last_exception}")


def parse_llm_json(text: str) -> tuple[str, str]:
    text = clean_json_response(text)

    try:
        parsed = json.loads(text)

        name = str(parsed.get("name", "")).strip()
        reason = str(parsed.get("reason", "")).strip()

        return name, reason

    except Exception:
        return "", text.strip()


def name_clusters(
    csv_path: str,
    out_path: str,
    base_url: str,
    api_key: str,
    api_key_header: str,
    model: str | None,
    limit: int,
    top_n: int,
) -> None:

    clusters = read_clusters(csv_path, top_n)

    cluster_items = list(clusters.items())

    if limit and limit > 0:
        cluster_items = cluster_items[:limit]

    results = []

    for idx, (community_id, subreddits) in enumerate(cluster_items):

        print(
            f"[{idx+1}/{len(cluster_items)}] Processing cluster {community_id}...",
            flush=True,
        )

        prompt = build_prompt_for_cluster(
            community_id,
            subreddits,
        )

        payload = {
            "model": model or "",
            "prompt": prompt,
            "system": (
                "Bạn là trợ lý AI chuyên đặt tên cluster subreddit."
                " Luôn trả JSON hợp lệ."
            ),
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 128,
            },
        }

        try:
            resp = call_llm_with_retry(
                base_url,
                api_key,
                api_key_header,
                payload,
            )

            assistant_text = extract_assistant_text(resp)

            if not assistant_text.strip():
                raise ValueError("Empty assistant response")

            name, reason = parse_llm_json(assistant_text)

        except Exception as exc:

            print(
                f"[ERROR] Cluster {community_id}: {exc}",
                file=sys.stderr,
            )

            name = ""
            reason = f"ERROR: {exc}"

        results.append(
            (
                community_id,
                name,
                reason,
                ", ".join(subreddits),
            )
        )

        # avoid rate burst
        time.sleep(1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as outfh:

        writer = csv.writer(outfh)

        writer.writerow(
            [
                "community_id",
                "name",
                "reason",
                "example_subreddits",
            ]
        )

        writer.writerows(results)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Name subreddit clusters via LLM"
    )

    parser.add_argument(
        "--csv",
        default="result/community_result.csv",
    )

    parser.add_argument(
        "--out",
        default="result/cluster_names.csv",
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "LLM_API_BASE_URL",
            DEFAULT_BASE_URL,
        ),
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv(
            "SHIELD_API_KEY",
            DEFAULT_API_KEY,
        ),
    )

    parser.add_argument(
        "--api-key-header",
        default=os.getenv(
            "SHIELD_API_KEY_HEADER",
            DEFAULT_API_KEY_HEADER,
        ),
    )

    parser.add_argument(
        "--model",
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if not args.base_url:
        raise RuntimeError("Missing base_url")

    if not args.api_key:
        raise RuntimeError("Missing api_key")

    name_clusters(
        csv_path=args.csv,
        out_path=args.out,
        base_url=args.base_url,
        api_key=args.api_key,
        api_key_header=args.api_key_header,
        model=args.model,
        limit=args.limit,
        top_n=args.top_n,
    )

    print(f"\nSaved results to: {args.out}")


if __name__ == "__main__":
    main()
