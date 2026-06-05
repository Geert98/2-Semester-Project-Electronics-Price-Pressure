from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.news_providers.common import RequestThrottler, api_key_from_env, fetch_json_with_retry
from src.utils import load_config, setup_env

logger = logging.getLogger(__name__)

NYT_ARTICLE_SEARCH_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

EXPECTED_NEWS_COLUMNS = [
    "provider",
    "window_start",
    "window_end",
    "title",
    "url",
    "source",
    "language",
    "seen_date",
    "content",
    "social_image",
    "source_country",
]


def _short_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=True)[:800]


def _to_nyt_date(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y%m%d")


def _headline_text(doc: dict) -> str:
    headline = doc.get("headline") or {}
    return headline.get("main") or headline.get("print_headline") or ""


def _keyword_text(doc: dict) -> str:
    keywords = doc.get("keywords") or []
    values = []
    for keyword in keywords:
        value = keyword.get("value") if isinstance(keyword, dict) else None
        if value:
            values.append(str(value))
    return ", ".join(values)


def _first_multimedia_url(doc: dict) -> str:
    multimedia = doc.get("multimedia") or []
    for item in multimedia:
        url = item.get("url") if isinstance(item, dict) else None
        if not url:
            continue
        if url.startswith("http"):
            return url
        return f"https://www.nytimes.com/{url.lstrip('/')}"
    return ""


def _combined_content(doc: dict) -> str:
    parts = [
        doc.get("abstract"),
        doc.get("lead_paragraph"),
        doc.get("snippet"),
        _keyword_text(doc),
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


def normalize_nyt_doc(
    doc: dict,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> dict:
    return {
        "provider": "nyt",
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "title": _headline_text(doc),
        "url": doc.get("web_url"),
        "source": doc.get("source") or "The New York Times",
        "language": "en",
        "seen_date": doc.get("pub_date"),
        "content": _combined_content(doc),
        "social_image": _first_multimedia_url(doc),
        "source_country": "US",
    }


def validate_common_news_row(row: dict) -> list[str]:
    issues = []
    for column in EXPECTED_NEWS_COLUMNS:
        if column not in row:
            issues.append(f"missing column: {column}")

    if not str(row.get("url") or "").startswith("http"):
        issues.append("url is missing or invalid")
    if not str(row.get("title") or "").strip():
        issues.append("title is empty")
    if pd.isna(pd.to_datetime(row.get("seen_date"), errors="coerce")):
        issues.append("seen_date cannot be parsed")

    content = str(row.get("content") or "").strip()
    if not content:
        issues.append("content is empty")

    return issues


def fetch_nyt_rows(
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
    nyt_cfg = news_cfg.get("nyt", {})
    api_key = api_key_from_env(news_cfg, "nyt", "NYT_API_KEY")
    min_interval = float(nyt_cfg.get("min_request_interval_seconds", 12))
    sort = nyt_cfg.get("sort", "newest")
    throttler = throttler or RequestThrottler(min_interval)

    page_size = 10
    page = 0
    total_fetched = 0
    rows: list[dict] = []

    while page < max_pages and total_fetched < max_records:
        params = {
            "api-key": api_key,
            "q": query,
            "begin_date": _to_nyt_date(window_start),
            "end_date": _to_nyt_date(window_end),
            "sort": sort,
            "page": page,
            "fl": "web_url,snippet,lead_paragraph,abstract,source,multimedia,headline,keywords,pub_date,document_type,news_desk,section_name,type_of_material,word_count",
        }

        payload = fetch_json_with_retry(
            url=NYT_ARTICLE_SEARCH_URL,
            params=params,
            provider_name="NYT",
            max_retries=max_retries,
            base_sleep=retry_base_sleep,
            before_request=throttler.wait,
        )

        if "fault" in payload:
            raise RuntimeError(f"NYT API returned an error: {_short_payload(payload)}")
        if payload.get("status") == "ERROR":
            raise RuntimeError(f"NYT API returned an error: {_short_payload(payload)}")

        response = payload.get("response")
        if not isinstance(response, dict):
            raise RuntimeError(f"NYT API response is missing the response object: {_short_payload(payload)}")

        docs = response.get("docs") or []
        logger.info("Fetched %s NYT articles from page %s", len(docs), page)
        if not docs:
            break

        for doc in docs:
            rows.append(normalize_nyt_doc(doc, window_start, window_end))
            total_fetched += 1
            if total_fetched >= max_records:
                break

        page += 1

    return rows


def append_nyt_rows(
    rows: list[dict],
    news_cfg: dict,
    query: str,
    max_records: int,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_retries: int,
    retry_base_sleep: float,
) -> None:
    nyt_cfg = news_cfg.get("nyt", {})
    max_pages = max(1, int(nyt_cfg.get("max_pages_per_window", 1)))
    min_interval = float(nyt_cfg.get("min_request_interval_seconds", 12))
    max_requests = nyt_cfg.get("max_requests_per_run")
    request_count = 0
    throttler = RequestThrottler(min_interval)
    ordered_windows = list(windows)
    if bool(nyt_cfg.get("newest_first", True)):
        ordered_windows = list(reversed(ordered_windows))

    logger.info("Starting NYT ingestion across %s monthly windows", len(ordered_windows))
    for idx, (window_start, window_end) in enumerate(ordered_windows, start=1):
        if max_requests not in {None, ""} and request_count >= int(max_requests):
            logger.info("Stopping NYT ingestion after %s configured requests.", request_count)
            break

        logger.info(
            "Fetching NYT window %s/%s: %s to %s",
            idx,
            len(ordered_windows),
            window_start.date(),
            window_end.date(),
        )
        rows.extend(
            fetch_nyt_rows(
                news_cfg=news_cfg,
                query=query,
                window_start=window_start,
                window_end=window_end,
                max_records=max_records,
                max_pages=max_pages,
                max_retries=max_retries,
                retry_base_sleep=retry_base_sleep,
                throttler=throttler,
            )
        )
        request_count += max_pages


def _select_query(news_cfg: dict, query_mode: str | None, query: str | None) -> str:
    if query:
        return query
    mode = query_mode or news_cfg.get("query_mode")
    if mode and news_cfg.get(f"query_{mode}"):
        return str(news_cfg[f"query_{mode}"])
    return str(news_cfg["query"])


def test_nyt_format(
    config_path: str,
    start_date: str,
    end_date: str,
    query_mode: str | None,
    query: str | None,
    pages: int,
    max_records: int,
    output_path: str,
) -> list[dict]:
    setup_env()
    config = load_config(config_path)
    news_cfg = config["news"]
    resolved_query = _select_query(news_cfg, query_mode=query_mode, query=query)
    rows = fetch_nyt_rows(
        news_cfg=news_cfg,
        query=resolved_query,
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
        print("Columns:", ", ".join(rows[0].keys()))
        print("First title:", rows[0].get("title"))
        print("First content chars:", len(str(rows[0].get("content") or "")))
    if row_issues:
        print("Issue samples:")
        for idx, issues in row_issues[:5]:
            print(f"  row {idx}: {'; '.join(issues)}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Test NYT Article Search output against the project news schema.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-01-31")
    parser.add_argument("--query-mode", choices=["strict", "broad"], default="strict")
    parser.add_argument("--query")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=10)
    parser.add_argument("--output", default="artifacts/nyt/nyt_format_sample.jsonl")
    args = parser.parse_args()

    test_nyt_format(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        query_mode=args.query_mode,
        query=args.query,
        pages=args.pages,
        max_records=args.max_records,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
