from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from pymongo import MongoClient

from src.storage import get_mongo_settings
from src.utils import load_config, setup_env

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "electronics-price-pressure-research/1.0 "
    "(academic research link text extraction; contact project owner)"
)

SKIP_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bz2",
    ".dmg",
    ".doc",
    ".docx",
    ".exe",
    ".gz",
    ".iso",
    ".jpg",
    ".jpeg",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".tar",
    ".tgz",
    ".webm",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}

SKIP_DOMAINS = {
    "twitter.com",
    "x.com",
    "youtube.com",
    "youtu.be",
    "reddit.com",
}


class _ArticleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._block_tag: str | None = None
        self._block_parts: list[str] = []
        self.blocks: list[str] = []
        self.visible_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript", "svg", "form", "nav", "footer"}:
            self._skip_depth += 1
            return

        if self._skip_depth > 0:
            return

        if normalized in {"article", "p", "li", "h1", "h2", "h3"} and self._block_tag is None:
            self._block_tag = normalized
            self._block_parts = []

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript", "svg", "form", "nav", "footer"}:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return

        if self._skip_depth > 0:
            return

        if self._block_tag == normalized:
            text = _normalize_whitespace(" ".join(self._block_parts))
            if len(text) >= 25:
                self.blocks.append(text)
            self._block_tag = None
            self._block_parts = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return

        text = data.strip()
        if not text:
            return

        self.visible_parts.append(text)
        if self._block_tag is not None:
            self._block_parts.append(text)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value or "")).strip()


def _domain(url: str) -> str:
    parsed = urlparse(str(url or ""))
    return parsed.netloc.lower().removeprefix("www.")


def _should_skip_url(url: str) -> str | None:
    parsed = urlparse(str(url or ""))
    if parsed.scheme not in {"http", "https"}:
        return "unsupported_scheme"

    domain = parsed.netloc.lower().removeprefix("www.")
    if domain in SKIP_DOMAINS:
        return "skipped_domain"

    suffix = Path(parsed.path.lower()).suffix
    if suffix in SKIP_EXTENSIONS:
        return "skipped_file_type"

    return None


def _is_text_response(content_type: str) -> bool:
    normalized = content_type.lower()
    return (
        not normalized
        or "text/html" in normalized
        or "application/xhtml" in normalized
        or "text/plain" in normalized
    )


def _extract_text_from_response(body: bytes, content_type: str, max_chars: int) -> str:
    encoding = "utf-8"
    match = re.search(r"charset=([^;\s]+)", content_type, flags=re.IGNORECASE)
    if match:
        encoding = match.group(1)

    html = body.decode(encoding, errors="replace")
    parser = _ArticleTextParser()
    parser.feed(html)
    parser.close()

    blocks = parser.blocks
    if blocks:
        text = "\n\n".join(blocks)
    else:
        text = " ".join(parser.visible_parts)

    return _normalize_whitespace(text)[:max_chars]


def fetch_linked_content(
    url: str,
    timeout_seconds: float,
    max_response_bytes: int,
    max_chars: int,
    min_text_chars: int,
    user_agent: str,
) -> dict[str, Any]:
    skipped_reason = _should_skip_url(url)
    if skipped_reason:
        return {
            "linked_fetch_status": skipped_reason,
            "linked_fetch_error": skipped_reason,
            "linked_content": "",
            "linked_content_char_count": 0,
        }

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.2",
    }

    try:
        with requests.get(
            url,
            headers=headers,
            timeout=timeout_seconds,
            stream=True,
            allow_redirects=True,
        ) as response:
            status_code = int(response.status_code)
            content_type = response.headers.get("Content-Type", "")

            payload = {
                "linked_fetch_http_status": status_code,
                "linked_fetch_content_type": content_type[:200],
                "linked_fetch_final_url": response.url,
            }

            if status_code >= 400:
                payload.update(
                    {
                        "linked_fetch_status": "http_error",
                        "linked_fetch_error": f"http_{status_code}",
                        "linked_content": "",
                        "linked_content_char_count": 0,
                    }
                )
                return payload

            if not _is_text_response(content_type):
                payload.update(
                    {
                        "linked_fetch_status": "non_text_content",
                        "linked_fetch_error": content_type[:200],
                        "linked_content": "",
                        "linked_content_char_count": 0,
                    }
                )
                return payload

            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                chunks.append(chunk)
                total += len(chunk)
                if total > max_response_bytes:
                    break

            text = _extract_text_from_response(b"".join(chunks), content_type, max_chars=max_chars)
            status = "success" if len(text) >= min_text_chars else "no_text"
            payload.update(
                {
                    "linked_fetch_status": status,
                    "linked_fetch_error": "" if status == "success" else "text_too_short",
                    "linked_content": text,
                    "linked_content_char_count": len(text),
                }
            )
            return payload

    except requests.RequestException as exc:
        return {
            "linked_fetch_status": "request_error",
            "linked_fetch_error": str(exc)[:500],
            "linked_content": "",
            "linked_content_char_count": 0,
        }


def _candidate_filter(retry_failed: bool) -> dict[str, Any]:
    base_filter: dict[str, Any] = {
        "provider": "hackernews",
        "linked_url": {"$nin": [None, ""]},
    }
    if retry_failed:
        return base_filter

    base_filter["$or"] = [
        {"linked_fetch_status": {"$exists": False}},
        {"linked_fetch_status": None},
        {"linked_fetch_status": ""},
    ]
    return base_filter


def enrich_hackernews_links(
    config_path: str = "configs/config.yaml",
    limit: int = 25,
    retry_failed: bool = False,
    timeout_seconds_override: float | None = None,
    min_interval_seconds_override: float | None = None,
    progress_every: int = 25,
) -> dict[str, int]:
    setup_env()
    config = load_config(config_path)
    mongo_cfg = config["storage"]["mongo"]
    fetch_cfg = config.get("hackernews_link_fetch") or config.get("news", {}).get("hackernews_link_fetch", {})

    timeout_seconds = float(
        timeout_seconds_override
        if timeout_seconds_override is not None
        else fetch_cfg.get("timeout_seconds", 20)
    )
    min_interval_seconds = float(
        min_interval_seconds_override
        if min_interval_seconds_override is not None
        else fetch_cfg.get("min_request_interval_seconds", 1.0)
    )
    max_response_bytes = int(fetch_cfg.get("max_response_bytes", 2_000_000))
    max_chars = int(fetch_cfg.get("max_chars", 15_000))
    min_text_chars = int(fetch_cfg.get("min_text_chars", 300))
    user_agent = str(fetch_cfg.get("user_agent") or DEFAULT_USER_AGENT)

    uri, db_name = get_mongo_settings(config)
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    stats = {"checked": 0, "success": 0, "no_text": 0, "failed_or_skipped": 0}

    try:
        client.admin.command("ping")
        db = client[db_name]
        raw_collection = db[mongo_cfg["test_news_collection"]]
        clean_collection = db[mongo_cfg["test_clean_news_collection"]]

        cursor = (
            raw_collection.find(
                _candidate_filter(retry_failed),
                {
                    "_id": 0,
                    "url": 1,
                    "title": 1,
                    "linked_url": 1,
                    "linked_source_domain": 1,
                },
            )
            .sort("seen_date", -1)
            .limit(limit)
        )

        for record in cursor:
            stats["checked"] += 1
            linked_url = str(record.get("linked_url") or "")
            title = str(record.get("title") or "")[:100]
            logger.info("Fetching linked content %s/%s: %s", stats["checked"], limit, title)

            payload = fetch_linked_content(
                linked_url,
                timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes,
                max_chars=max_chars,
                min_text_chars=min_text_chars,
                user_agent=user_agent,
            )
            payload["linked_fetch_at"] = datetime.now(UTC).isoformat()
            payload["linked_source_domain"] = record.get("linked_source_domain") or _domain(linked_url)

            if payload["linked_fetch_status"] == "success":
                stats["success"] += 1
            elif payload["linked_fetch_status"] == "no_text":
                stats["no_text"] += 1
            else:
                stats["failed_or_skipped"] += 1

            filter_query = {"url": record.get("url")}
            raw_collection.update_one(filter_query, {"$set": payload})
            clean_collection.update_one(filter_query, {"$set": payload})

            if min_interval_seconds > 0:
                time.sleep(min_interval_seconds)

            if progress_every > 0 and stats["checked"] % progress_every == 0:
                print(
                    "HN linked-content enrichment progress: "
                    f"checked={stats['checked']}/{limit}, "
                    f"success={stats['success']}, no_text={stats['no_text']}, "
                    f"failed_or_skipped={stats['failed_or_skipped']}",
                    flush=True,
                )

    finally:
        client.close()

    print(
        "HN linked-content enrichment: "
        f"checked={stats['checked']}, success={stats['success']}, "
        f"no_text={stats['no_text']}, failed_or_skipped={stats['failed_or_skipped']}"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch linked article text for Hacker News rows.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--min-interval-seconds", type=float)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    enrich_hackernews_links(
        config_path=args.config,
        limit=args.limit,
        retry_failed=args.retry_failed,
        timeout_seconds_override=args.timeout_seconds,
        min_interval_seconds_override=args.min_interval_seconds,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
