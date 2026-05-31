"""
Build a subreddit -> content mapping for the demo UI.

The script uses Reddit's public JSON endpoints with a custom User-Agent and
writes a CSV that can be consumed by web-demo/scripts/build-demo-data.mjs.

Examples:
  python processing-local/scripts/crawl_subreddit_content.py --top-similarity 16
  python processing-local/scripts/crawl_subreddit_content.py --names 196 19684
  python processing-local/scripts/crawl_subreddit_content.py --limit 200 --append
"""

from __future__ import annotations

import argparse
import base64
import csv
import heapq
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMMUNITY_CSV = REPO_ROOT / "result" / "community_result.csv"
DEFAULT_CLUSTER_NAMES_CSV = REPO_ROOT / "result" / "cluster_names.csv"
DEFAULT_SIMILARITY_CSV = REPO_ROOT / "subreddit_similarity_results.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "result" / "subreddit_content_map.csv"
DEFAULT_USER_AGENT = "MassiveDataMiningSubredditContentCrawler/1.0"

CSV_FIELDS = [
    "subreddit",
    "display_name",
    "title",
    "public_description",
    "description",
    "content",
    "subscribers",
    "over18",
    "url",
    "source",
    "status",
    "fetched_at",
    "error",
]


def normalize_name(value: str | None) -> str:
    return str(value or "").strip().strip("/").removeprefix("r/")


def normalize_key(value: str | None) -> str:
    return normalize_name(value).casefold()


def collapse_text(value: str | None) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`>#|~]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_content(title: str, public_description: str, description: str) -> str:
    parts = [collapse_text(title), collapse_text(public_description), collapse_text(description)]
    seen: set[str] = set()
    unique_parts = []
    for part in parts:
        key = part.casefold()
        if part and key not in seen:
            unique_parts.append(part)
            seen.add(key)
    return " | ".join(unique_parts)


def read_existing_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        return {
            normalize_key(row.get("subreddit") or row.get("display_name")): row
            for row in rows
            if normalize_key(row.get("subreddit") or row.get("display_name"))
        }


def read_subreddits_from_community_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    names: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            name = normalize_name(row.get("subreddit"))
            if name:
                names.append(name)
    return names


def read_community_fallbacks(community_csv: Path, cluster_names_csv: Path) -> dict[str, dict[str, str]]:
    if not community_csv.exists() or not cluster_names_csv.exists():
        return {}

    community_names: dict[str, dict[str, str]] = {}
    with cluster_names_csv.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            community_id = str(row.get("community_id") or "").strip()
            if not community_id:
                continue
            community_names[community_id] = {
                "community_id": community_id,
                "community_name": row.get("name") or f"Community {community_id}",
                "reason": row.get("reason") or "",
            }

    fallbacks: dict[str, dict[str, str]] = {}
    with community_csv.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            subreddit = normalize_name(row.get("subreddit"))
            community_id = str(row.get("community_id") or "").strip()
            if not subreddit or not community_id:
                continue
            info = community_names.get(
                community_id,
                {
                    "community_id": community_id,
                    "community_name": f"Community {community_id}",
                    "reason": "",
                },
            )
            fallbacks[normalize_key(subreddit)] = info
    return fallbacks


def read_subreddits_from_input_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if "," in first_line:
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = csv.DictReader(file)
            fields = rows.fieldnames or []
            name_field = next(
                (
                    field
                    for field in fields
                    if field.lower() in {"subreddit", "display_name", "name", "source", "target"}
                ),
                fields[0] if fields else "",
            )
            return [normalize_name(row.get(name_field)) for row in rows if normalize_name(row.get(name_field))]

    names = re.split(r"[\s,]+", text)
    return [normalize_name(name) for name in names if normalize_name(name)]


def read_top_similarity_names(path: Path, limit: int) -> list[str]:
    if limit <= 0 or not path.exists():
        return []

    heap: list[tuple[float, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            source = normalize_name(row.get("Subreddit_A") or row.get("source"))
            target = normalize_name(row.get("Subreddit_B") or row.get("target"))
            try:
                score = float(row.get("Similarity_Score") or row.get("score") or 0)
            except ValueError:
                continue
            if not source or not target:
                continue
            item = (score, source, target)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    names: list[str] = []
    for _, source, target in sorted(heap, reverse=True):
        names.extend([source, target])
    return names


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        name = normalize_name(value)
        key = normalize_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(name)
    return result


def fetch_json(
    url: str,
    user_agent: str,
    timeout: float,
    retries: int = 2,
    bearer_token: str | None = None,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        headers = {"User-Agent": user_agent, "Accept": "application/json"}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(min(8, 2 ** attempt))
        except (URLError, TimeoutError) as error:
            last_error = error
            time.sleep(min(8, 2 ** attempt))
    if last_error:
        raise last_error
    raise RuntimeError("Unknown fetch error")


def fetch_reddit_oauth_token(
    client_id: str,
    client_secret: str,
    user_agent: str,
    timeout: float,
) -> str:
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    request = Request(
        "https://www.reddit.com/api/v1/access_token",
        data=urlencode({"grant_type": "client_credentials"}).encode("utf-8"),
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": user_agent,
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"Reddit OAuth did not return access_token: {payload}")
    return token


def search_subreddit(
    name: str,
    user_agent: str,
    timeout: float,
    bearer_token: str | None = None,
) -> str | None:
    query = urlencode({"q": name, "limit": 10, "include_over_18": "on"})
    if bearer_token:
        url = f"https://oauth.reddit.com/subreddits/search?{query}"
    else:
        url = f"https://www.reddit.com/subreddits/search.json?{query}"
    payload = fetch_json(url, user_agent, timeout, bearer_token=bearer_token)
    children = payload.get("data", {}).get("children", [])
    for child in children:
        display_name = child.get("data", {}).get("display_name") or ""
        if display_name.casefold() == name.casefold():
            return display_name
    first = children[0].get("data", {}).get("display_name") if children else None
    return first or None


def fetch_subreddit_about(
    name: str,
    user_agent: str,
    timeout: float,
    use_search: bool,
    bearer_token: str | None = None,
) -> tuple[dict | None, str]:
    if bearer_token:
        exact_url = f"https://oauth.reddit.com/r/{quote(name, safe='')}/about"
        direct_source = "reddit_oauth"
        search_source = "reddit_oauth_search"
    else:
        exact_url = f"https://www.reddit.com/r/{quote(name, safe='')}/about.json"
        direct_source = "reddit_about"
        search_source = "reddit_search"

    try:
        return fetch_json(exact_url, user_agent, timeout, bearer_token=bearer_token), direct_source
    except HTTPError as error:
        if error.code not in {403, 404} or not use_search:
            raise

    match = search_subreddit(name, user_agent, timeout, bearer_token=bearer_token)
    if not match:
        return None, search_source
    if bearer_token:
        search_url = f"https://oauth.reddit.com/r/{quote(match, safe='')}/about"
    else:
        search_url = f"https://www.reddit.com/r/{quote(match, safe='')}/about.json"
    return fetch_json(search_url, user_agent, timeout, bearer_token=bearer_token), search_source


def build_row(name: str, payload: dict | None, source: str, status: str, error: str = "") -> dict[str, str]:
    data = payload.get("data", {}) if payload else {}
    display_name = data.get("display_name") or name
    title = collapse_text(data.get("title"))
    public_description = collapse_text(data.get("public_description"))
    description = collapse_text(data.get("description"))
    content = build_content(title, public_description, description)
    url = data.get("url") or f"/r/{display_name}/"

    return {
        "subreddit": name,
        "display_name": display_name,
        "title": title,
        "public_description": public_description,
        "description": description,
        "content": content,
        "subscribers": str(data.get("subscribers") or ""),
        "over18": str(bool(data.get("over18"))) if data else "",
        "url": url,
        "source": source,
        "status": status if content or error else "empty",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }


def build_fallback_row(name: str, fallback: dict[str, str] | None, error: str = "") -> dict[str, str]:
    community_name = fallback.get("community_name") if fallback else ""
    reason = fallback.get("reason") if fallback else ""
    content = build_content(
        name,
        "",
        f'Thuộc community "{community_name}". {reason}' if community_name or reason else "",
    )
    return {
        "subreddit": name,
        "display_name": name,
        "title": name,
        "public_description": "",
        "description": "",
        "content": content,
        "subscribers": "",
        "over18": "",
        "url": f"/r/{name}/",
        "source": "community_inference",
        "status": "fallback_error" if error else "inferred",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }


def write_rows(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl subreddit descriptions into a CSV mapping.")
    parser.add_argument("--community-csv", type=Path, default=DEFAULT_COMMUNITY_CSV)
    parser.add_argument("--cluster-names-csv", type=Path, default=DEFAULT_CLUSTER_NAMES_CSV)
    parser.add_argument("--similarity-csv", type=Path, default=DEFAULT_SIMILARITY_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--input", action="append", type=Path, default=[], help="TXT/CSV file containing subreddit names.")
    parser.add_argument("--names", nargs="*", default=[], help="Subreddit names to fetch first.")
    parser.add_argument("--top-similarity", type=int, default=0, help="Fetch unique names from top N similarity pairs.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of subreddits crawled after de-duplication.")
    parser.add_argument("--sleep", type=float, default=0.7, help="Seconds to wait between Reddit requests.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--append", action="store_true", help="Keep existing successful rows and fetch only missing names.")
    parser.add_argument("--no-search", action="store_true", help="Do not fallback to Reddit subreddit search on 403/404.")
    parser.add_argument("--no-community-fallback", action="store_true", help="Do not fill content from cluster_names.csv when Reddit blocks a request.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected names without fetching.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--client-id", default=os.getenv("REDDIT_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.getenv("REDDIT_CLIENT_SECRET"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names: list[str] = []
    names.extend(args.names)
    names.extend(read_top_similarity_names(args.similarity_csv, args.top_similarity))
    for input_path in args.input:
        names.extend(read_subreddits_from_input_file(input_path))
    if not names:
        names.extend(read_subreddits_from_community_csv(args.community_csv))

    names = unique_in_order(names)
    if args.limit > 0:
        names = names[: args.limit]

    existing = read_existing_rows(args.output) if args.append else {}
    fetch_names = [name for name in names if normalize_key(name) not in existing]

    print(f"Selected {len(names)} subreddit(s).")
    if args.append:
        print(f"Existing rows kept: {len(existing)}. Missing rows to fetch: {len(fetch_names)}.")

    if args.dry_run:
        for name in fetch_names:
            print(name)
        return 0

    bearer_token = None
    if args.client_id and args.client_secret:
        print("Using Reddit OAuth client credentials.")
        bearer_token = fetch_reddit_oauth_token(
            args.client_id,
            args.client_secret,
            args.user_agent,
            args.timeout,
        )
    else:
        print(
            "No REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET found; using public Reddit endpoints. "
            "If Reddit returns 403, set OAuth credentials and run again with --append."
        )

    community_fallbacks = (
        {}
        if args.no_community_fallback
        else read_community_fallbacks(args.community_csv, args.cluster_names_csv)
    )
    rows_by_key = dict(existing)
    for index, name in enumerate(fetch_names, start=1):
        print(f"[{index}/{len(fetch_names)}] Fetching r/{name}", flush=True)
        fallback = community_fallbacks.get(normalize_key(name))
        try:
            payload, source = fetch_subreddit_about(
                name,
                user_agent=args.user_agent,
                timeout=args.timeout,
                use_search=not args.no_search,
                bearer_token=bearer_token,
            )
            row = build_row(name, payload, source, "ok")
            if not row.get("content") and fallback:
                row = build_fallback_row(name, fallback, "Reddit response did not include content")
        except Exception as error:  # noqa: BLE001 - keep crawl moving and store the error.
            row = (
                build_fallback_row(name, fallback, str(error))
                if fallback
                else build_row(name, None, "reddit_about", "error", str(error))
            )
            print(f"  error: {error}", file=sys.stderr, flush=True)
        rows_by_key[normalize_key(name)] = row
        if index < len(fetch_names) and args.sleep > 0:
            time.sleep(args.sleep)

    ordered_rows = []
    for name in names:
        row = rows_by_key.get(normalize_key(name))
        if row:
            ordered_rows.append(row)
    for key, row in rows_by_key.items():
        if key not in {normalize_key(name) for name in names}:
            ordered_rows.append(row)

    write_rows(args.output, ordered_rows)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
