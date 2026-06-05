from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.news_providers.common import RequestThrottler, fetch_json_with_retry
from src.news_providers.nyt import EXPECTED_NEWS_COLUMNS, validate_common_news_row
from src.utils import load_config, setup_env

logger = logging.getLogger(__name__)

HN_SEARCH_BY_DATE_URL = "https://hn.algolia.com/api/v1/search_by_date"


def _domain(url: str) -> str:
    parsed = urlparse(str(url or ""))
    return parsed.netloc.lower().removeprefix("www.")


def _hn_discussion_url(object_id: str | int | None) -> str:
    if not object_id:
        return ""
    return f"https://news.ycombinator.com/item?id={object_id}"


def _story_content(hit: dict) -> str:
    parts = [
        hit.get("title"),
        hit.get("story_text"),
    ]
    seen = set()
    clean_parts = []
    for part in parts:
        text = str(part or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        clean_parts.append(text)
    return "\n\n".join(clean_parts)


def _unix_seconds(dt: pd.Timestamp, end_of_day: bool = False) -> int:
    if end_of_day:
        dt = dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return int(pd.Timestamp(dt, tz="UTC").timestamp())


def _extract_hits(payload: dict) -> list[dict]:
    hits = payload.get("hits", [])
    if not isinstance(hits, list):
        return []
    return [hit for hit in hits if isinstance(hit, dict)]


def normalize_hackernews_hit(
    hit: dict,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> dict:
    linked_url = str(hit.get("url") or "").strip()
    object_id = hit.get("objectID")
    hn_url = _hn_discussion_url(object_id)
    source_domain = _domain(linked_url)

    return {
        "provider": "hackernews",
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "title": hit.get("title") or "",
        "url": hn_url,
        "source": source_domain or "Hacker News",
        "language": "en",
        "seen_date": hit.get("created_at"),
        "content": _story_content(hit),
        "social_image": "",
        "source_country": "",
        "source_type": "tech_attention_link",
        "linked_url": linked_url,
        "linked_source_domain": source_domain,
        "hn_object_id": str(object_id or ""),
        "hn_discussion_url": hn_url,
        "hn_author": hit.get("author") or "",
        "hn_points": int(hit.get("points") or 0),
        "hn_num_comments": int(hit.get("num_comments") or 0),
        "hn_created_at_i": int(hit.get("created_at_i") or 0),
    }


def _passes_hackernews_filters(row: dict, hackernews_cfg: dict) -> bool:
    if bool(hackernews_cfg.get("require_url", True)) and not row.get("linked_url"):
        return False
    if bool(hackernews_cfg.get("require_title", True)) and not str(row.get("title") or "").strip():
        return False

    min_points = int(hackernews_cfg.get("min_points", 0))
    if int(row.get("hn_points") or 0) < min_points:
        return False

    return True


def fetch_hackernews_rows(
    news_cfg: dict,
    query: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    max_records: int,
    max_pages: int,
    max_retries: int,
    retry_base_sleep: float,
    throttler: RequestThrottler | None = None,
) -> list[dict]:
    hackernews_cfg = news_cfg.get("hackernews", {})
    min_interval = float(hackernews_cfg.get("min_request_interval_seconds", 0.5))
    hits_per_page = min(100, int(hackernews_cfg.get("hits_per_page", 100)))
    tags = hackernews_cfg.get("tags", "story")
    throttler = throttler or RequestThrottler(min_interval)

    rows = []
    page = 0
    while page < max_pages and len(rows) < max_records:
        params = {
            "query": query,
            "tags": tags,
            "numericFilters": (
                f"created_at_i>={_unix_seconds(window_start)},"
                f"created_at_i<={_unix_seconds(window_end, end_of_day=True)}"
            ),
            "page": page,
            "hitsPerPage": hits_per_page,
        }
        payload = fetch_json_with_retry(
            url=HN_SEARCH_BY_DATE_URL,
            params=params,
            provider_name="Hacker News",
            max_retries=max_retries,
            base_sleep=retry_base_sleep,
            before_request=throttler.wait,
        )
        hits = _extract_hits(payload)
        logger.info("Fetched %s Hacker News hits from page %s for %s", len(hits), page, query)
        if not hits:
            break

        for hit in hits:
            row = normalize_hackernews_hit(hit, window_start, window_end)
            if not _passes_hackernews_filters(row, hackernews_cfg):
                continue
            rows.append(row)
            if len(rows) >= max_records:
                break

        page += 1

    return rows


def _configured_queries(news_cfg: dict, query: str) -> list[str]:
    hackernews_cfg = news_cfg.get("hackernews", {})
    queries = hackernews_cfg.get("queries")
    if isinstance(queries, list) and queries:
        return [str(value).strip() for value in queries if str(value).strip()]
    return [str(hackernews_cfg.get("query") or query).strip()]


def _group_windows(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    months_per_request: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if months_per_request <= 1:
        return windows

    grouped = []
    for start_idx in range(0, len(windows), months_per_request):
        chunk = windows[start_idx : start_idx + months_per_request]
        grouped.append((chunk[0][0], chunk[-1][1]))
    return grouped


def _max_requests_reached(request_count: int, hackernews_cfg: dict) -> bool:
    max_requests = hackernews_cfg.get("max_requests_per_run")
    if max_requests in {None, ""}:
        return False
    return request_count >= int(max_requests)


def append_hackernews_rows(
    rows: list[dict],
    news_cfg: dict,
    query: str,
    max_records: int,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_retries: int,
    retry_base_sleep: float,
) -> None:
    hackernews_cfg = news_cfg.get("hackernews", {})
    max_pages = max(1, int(hackernews_cfg.get("max_pages_per_window", 1)))
    months_per_request = max(1, int(hackernews_cfg.get("months_per_request", 3)))
    queries = _configured_queries(news_cfg, query)
    ordered_windows = _group_windows(list(windows), months_per_request)
    if bool(hackernews_cfg.get("newest_first", True)):
        ordered_windows = list(reversed(ordered_windows))

    min_interval = float(hackernews_cfg.get("min_request_interval_seconds", 0.5))
    throttler = RequestThrottler(min_interval)
    seen_urls = {str(row.get("url")) for row in rows if row.get("url")}
    request_count = 0

    logger.info("Starting Hacker News ingestion across %s windows", len(ordered_windows))
    for idx, (window_start, window_end) in enumerate(ordered_windows, start=1):
        if _max_requests_reached(request_count, hackernews_cfg):
            logger.info("Stopping Hacker News ingestion after %s configured requests.", request_count)
            break

        logger.info(
            "Fetching Hacker News window %s/%s: %s to %s",
            idx,
            len(ordered_windows),
            window_start.date(),
            window_end.date(),
        )
        for hn_query in queries:
            if _max_requests_reached(request_count, hackernews_cfg):
                break

            fetched_rows = fetch_hackernews_rows(
                news_cfg=news_cfg,
                query=hn_query,
                window_start=window_start,
                window_end=window_end,
                max_records=max_records,
                max_pages=max_pages,
                max_retries=max_retries,
                retry_base_sleep=retry_base_sleep,
                throttler=throttler,
            )
            request_count += max_pages
            for row in fetched_rows:
                url = str(row.get("url") or "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                rows.append(row)


def test_hackernews_format(
    config_path: str,
    start_date: str,
    end_date: str,
    query: str,
    pages: int,
    max_records: int,
    output_path: str,
) -> list[dict]:
    setup_env()
    config = load_config(config_path)
    news_cfg = config["news"]
    rows = fetch_hackernews_rows(
        news_cfg=news_cfg,
        query=query,
        window_start=pd.to_datetime(start_date),
        window_end=pd.to_datetime(end_date),
        max_records=max_records,
        max_pages=pages,
        max_retries=int(news_cfg.get("max_retries", 3)),
        retry_base_sleep=float(news_cfg.get("retry_base_sleep", 3)),
    )
    row_issues = [(idx, validate_common_news_row(row)) for idx, row in enumerate(rows, start=1)]
    row_issues = [(idx, issues) for idx, issues in row_issues if issues]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Fetched rows: {len(rows)}")
    print(f"Rows with format issues: {len(row_issues)}")
    print(f"Sample output path: {output}")
    if rows:
        print("Columns:", ", ".join(EXPECTED_NEWS_COLUMNS))
        print("First title:", rows[0].get("title"))
        print("First linked source:", rows[0].get("linked_source_domain"))
        print("First HN points/comments:", rows[0].get("hn_points"), rows[0].get("hn_num_comments"))
    if row_issues:
        print("Issue samples:")
        for idx, issues in row_issues[:5]:
            print(f"  row {idx}: {'; '.join(issues)}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Hacker News Algolia output against the project news schema."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--query", default="semiconductor")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--output", default="artifacts/hackernews/hackernews_format_sample.jsonl")
    args = parser.parse_args()

    test_hackernews_format(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        query=args.query,
        pages=args.pages,
        max_records=args.max_records,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
