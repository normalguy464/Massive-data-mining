import os
import json
from typing import Any, Dict

import requests


def load_dotenv(path: str | None = None) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ if not already set."""
    if path is None:
        path = os.path.join(os.getcwd(), ".env")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
    except FileNotFoundError:
        return


# Load .env at import time so other modules can use env vars.
load_dotenv()


DEFAULT_BASE_URL = os.getenv("LLM_API_URL", os.getenv("LLM_API_BASE_URL"))
DEFAULT_API_KEY = os.getenv("SHIELD_API_KEY", "")
DEFAULT_API_KEY_HEADER = os.getenv("SHIELD_API_KEY_HEADER", "x-api-key")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "")


def _normalize_url(url: str) -> str:
    # Validate type and value
    if not url or not isinstance(url, str):
        raise ValueError(
            "LLM base URL is not set. Set LLM_API_URL or LLM_API_BASE_URL in your .env or environment."
        )

    # If user provides a base URL without path, assume /v1/chat
    if url.endswith("/v1/chat") or url.endswith("/v1/generate") or "/v1/" in url:
        return url
    return url.rstrip("/") + "/v1/chat"


def post_chat(base_url: str, api_key: str, api_key_header: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    url = _normalize_url(base_url)
    headers = {
        "Content-Type": "application/json",
        api_key_header: api_key,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


def extract_assistant_text(response: Dict[str, Any]) -> str:
    # Try common shapes
    if not isinstance(response, dict):
        return json.dumps(response, ensure_ascii=False)

    if "response" in response and isinstance(response["response"], str):
        return response["response"]

    if "message" in response and isinstance(response["message"], dict):
        c = response["message"].get("content")
        if isinstance(c, str):
            return c

    if "content" in response and isinstance(response["content"], str):
        return response["content"]

    # fallback: pretty JSON
    return json.dumps(response, ensure_ascii=False, indent=2)


def main_example() -> None:
    base = DEFAULT_BASE_URL
    api_key = DEFAULT_API_KEY
    header = DEFAULT_API_KEY_HEADER
    model = DEFAULT_MODEL

    if not api_key:
        print("SHIELD_API_KEY not set in environment or .env. Set it before running.")
        return

    payload = {
        "model": model or "",
        "prompt": "Giải thích attention trong Transformer",
        "system": "Bạn là trợ lý AI chuyên về machine learning.",
        "options": {"temperature": 0.7, "num_predict": 256},
    }

    resp = post_chat(base, api_key, header, payload, timeout=300)
    print("RESPONSE:\n", extract_assistant_text(resp))


if __name__ == "__main__":
    main_example()