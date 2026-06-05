from __future__ import annotations

import argparse
from html import unescape
import json
import logging
import os
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from src.news_providers.common import RequestThrottler, api_key_from_env, fetch_json_with_retry
from src.news_providers.nyt import EXPECTED_NEWS_COLUMNS, validate_common_news_row
from src.utils import load_config, setup_env

logger = logging.getLogger(__name__)

DEFAULT_REUTERS_HOST = "reuters-business-and-financial-news.p.rapidapi.com"
DEFAULT_REQUIRED_TERMS = [
    "semiconductor",
    "semiconductors",
    "microchip",
    "microchips",
    "computer chip",
    "computer chips",
    "chipmaker",
    "chipmakers",
    "chip industry",
    "ai chip",
    "ai chips",
    "memory chip",
    "memory chips",
    "dram",
    "nand",
    "high bandwidth memory",
    "hbm",
    "integrated circuit",
    "integrated circuits",
]
DEFAULT_PRESSURE_TERMS = [
    "price",
    "prices",
    "pricing",
    "shortage",
    "shortages",
    "supply",
    "supplies",
    "demand",
    "tariff",
    "tariffs",
    "export control",
    "export controls",
    "restriction",
    "restrictions",
    "sanction",
    "sanctions",
    "inventory",
    "inventories",
    "oversupply",
    "production",
    "capacity",
    "manufacturing",
    "regulation",
    "regulations",
]
DEFAULT_EXCLUDE_TERMS = [
    "potato chip",
    "potato chips",
    "poker chip",
    "poker chips",
    "casino chip",
    "casino chips",
    "fish and chips",
    "chocolate chip",
    "blue chip",
    "blue-chip",
]


def _select_query(news_cfg: dict, query_mode: str | None, query: str | None) -> str:
    if query:
        return query
    mode = query_mode or news_cfg.get("query_mode")
    if mode and news_cfg.get(f"query_{mode}"):
        return str(news_cfg[f"query_{mode}"])
    return str(news_cfg["query"])


def _config_list(config: dict, key: str, default: list[str]) -> list[str]:
    values = config.get(key, default)
    if not isinstance(values, list):
        return default
    return [str(value).strip().lower() for value in values if str(value).strip()]


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _rapidapi_host(news_cfg: dict) -> str:
    reuters_cfg = news_cfg.get("reuters", {})
    host_env = reuters_cfg.get("host_env", "RAPIDAPI_REUTERS_HOST")
    return os.getenv(host_env) or reuters_cfg.get("host") or DEFAULT_REUTERS_HOST


def _rapidapi_headers(news_cfg: dict) -> dict:
    api_key = api_key_from_env(news_cfg, "reuters", "RAPIDAPI_REUTERS_KEY")
    host = _rapidapi_host(news_cfg)
    return {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
    }


def _base_url(news_cfg: dict) -> str:
    reuters_cfg = news_cfg.get("reuters", {})
    host = _rapidapi_host(news_cfg)
    return str(reuters_cfg.get("base_url") or f"https://{host}").rstrip("/")


def _first_value(record: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _clean_text(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", unescape(without_tags)).strip()


def _stringify_date(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("date") or "").strip()
    return str(value or "").strip()


def _reuters_url(value: str) -> str:
    url = value.strip()
    if not url:
        return ""
    if url.startswith("http"):
        return url
    return f"https://www.reuters.com/{url.lstrip('/')}"


def _first_image_url(article: dict) -> str:
    image = _first_value(article, ("image", "image_url", "thumbnail", "thumbnail_url", "urlToImage"))
    if image:
        return image

    files = article.get("files") or []
    if isinstance(files, list):
        for item in files:
            if isinstance(item, dict) and item.get("urlCdn"):
                return str(item["urlCdn"]).strip()

    return ""


def _nested_value(record: dict, paths: tuple[tuple[str, ...], ...]) -> str:
    for path in paths:
        value = record
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _articles_description_text(value: object) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return _clean_text(text)

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content")
            else:
                content = item
            text = _clean_text(str(content or ""))
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    return ""


def _combined_content(record: dict) -> str:
    candidates = [
        _articles_description_text(record.get("articlesDescription")),
        _first_value(
            record,
            (
                "content",
                "body",
                "text",
                "articleBody",
                "article_body",
                "articleText",
                "article_text",
                "story",
                "description",
                "articlesShortDescription",
                "summary",
                "snippet",
                "lead",
                "lead_paragraph",
            ),
        ),
        _nested_value(record, (("article", "body"), ("article", "text"), ("article", "content"))),
    ]
    parts = []
    for candidate in candidates:
        text = _clean_text(str(candidate or ""))
        if not text:
            continue
        if any(text in existing for existing in parts):
            continue
        parts = [existing for existing in parts if existing not in text]
        parts.append(text)
    return "\n\n".join(parts)


def _passes_reuters_filters(row: dict, reuters_cfg: dict) -> bool:
    if not bool(reuters_cfg.get("filter_relevance", True)):
        return True

    title = str(row.get("title") or "")
    content = str(row.get("content") or "")
    combined_text = f"{title}\n{content}".lower()
    min_content_chars = int(reuters_cfg.get("min_content_chars", 300))

    if len(content.strip()) < min_content_chars:
        return False

    exclude_terms = _config_list(reuters_cfg, "exclude_terms", DEFAULT_EXCLUDE_TERMS)
    if _contains_any(combined_text, exclude_terms):
        return False

    required_terms = _config_list(reuters_cfg, "required_terms", DEFAULT_REQUIRED_TERMS)
    if required_terms and not _contains_any(combined_text, required_terms):
        return False

    pressure_terms = _config_list(reuters_cfg, "pressure_terms", DEFAULT_PRESSURE_TERMS)
    if bool(reuters_cfg.get("require_pressure_term", True)) and not _contains_any(
        combined_text,
        pressure_terms,
    ):
        return False

    return True


def _extract_articles(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    for key in ("data", "articles", "results", "items", "news", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _extract_articles(value)
            if nested:
                return nested

    return []


def normalize_reuters_article(
    article: dict,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> dict:
    title = _first_value(
        article,
        (
            "title",
            "headline",
            "headlineText",
            "articleTitle",
            "article_title",
            "articlesName",
            "name",
        ),
    )
    url = _reuters_url(
        _first_value(
            article,
            (
                "url",
                "link",
                "webUrl",
                "web_url",
                "articleUrl",
                "article_url",
                "urlSupplier",
                "canonicalSupplier",
            ),
        )
    )
    seen_date = _stringify_date(
        next(
            (
                article[key]
                for key in (
                    "published_at",
                    "publishedAt",
                    "published",
                    "publishDate",
                    "publish_date",
                    "publishedDate",
                    "date",
                    "article_date",
                    "created_at",
                )
                if key in article and article[key] is not None
            ),
            "",
        )
    )
    source = _first_value(article, ("source", "source_name", "publisher", "provider", "agency")) or "Reuters"

    return {
        "provider": "reuters_rapidapi",
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "title": title,
        "url": url,
        "source": source,
        "language": "en",
        "seen_date": seen_date,
        "content": _combined_content(article),
        "social_image": _first_image_url(article),
        "source_country": "",
    }


def _endpoint_path(
    endpoint_mode: str,
    from_date: str,
    to_date: str,
    keyword: str,
    page: int,
    limit: int,
) -> str:
    if endpoint_mode == "date_range":
        return f"/get-articles-between-dates/{from_date}/{to_date}/{page}/{limit}"
    if endpoint_mode == "keyword":
        return f"/get-articles-by-keyword-name/{quote(keyword)}/{page}/{limit}"
    return (
        "/get-articles-by-keyword-name-date-range/"
        f"{from_date}/{to_date}/{quote(keyword)}/{page}/{limit}"
    )


def fetch_reuters_rows(
    news_cfg: dict,
    keyword: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    max_records: int,
    max_pages: int,
    max_retries: int,
    retry_base_sleep: float,
) -> tuple[list[dict], list[dict]]:
    reuters_cfg = news_cfg.get("reuters", {})
    min_interval = float(reuters_cfg.get("min_request_interval_seconds", 1.0))
    endpoint_mode = str(reuters_cfg.get("endpoint_mode", "keyword_date_range"))
    page_size = min(int(reuters_cfg.get("page_size", 20)), 20)
    throttler = RequestThrottler(min_interval)
    headers = _rapidapi_headers(news_cfg)
    base_url = _base_url(news_cfg)

    rows: list[dict] = []
    raw_articles: list[dict] = []
    page = 0
    from_date = window_start.strftime("%Y-%m-%d")
    to_date = window_end.strftime("%Y-%m-%d")

    while page < max_pages and len(rows) < max_records:
        path = _endpoint_path(
            endpoint_mode=endpoint_mode,
            from_date=from_date,
            to_date=to_date,
            keyword=keyword,
            page=page,
            limit=page_size,
        )
        payload = fetch_json_with_retry(
            url=f"{base_url}{path}",
            params={},
            headers=headers,
            provider_name="Reuters RapidAPI",
            max_retries=max_retries,
            base_sleep=retry_base_sleep,
            before_request=throttler.wait,
        )
        articles = _extract_articles(payload)
        logger.info("Fetched %s Reuters articles from page %s for %s", len(articles), page, keyword)
        if not articles:
            break

        for article in articles:
            raw_articles.append(article)
            row = normalize_reuters_article(article, window_start, window_end)
            if not _passes_reuters_filters(row, reuters_cfg):
                continue

            rows.append(row)
            if len(rows) >= max_records:
                break

        page += 1

    return rows, raw_articles


def _configured_keywords(news_cfg: dict, query: str) -> list[str]:
    reuters_cfg = news_cfg.get("reuters", {})
    keywords = reuters_cfg.get("keywords")
    if isinstance(keywords, list) and keywords:
        return [str(keyword).strip() for keyword in keywords if str(keyword).strip()]

    keyword = str(reuters_cfg.get("keyword") or "").strip()
    if keyword:
        return [keyword]

    return [query]


def _max_requests_reached(request_count: int, reuters_cfg: dict) -> bool:
    max_requests = reuters_cfg.get("max_requests_per_run")
    if max_requests in {None, ""}:
        return False
    return request_count >= int(max_requests)


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


def append_reuters_rows(
    rows: list[dict],
    news_cfg: dict,
    query: str,
    max_records: int,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_retries: int,
    retry_base_sleep: float,
) -> None:
    reuters_cfg = news_cfg.get("reuters", {})
    max_pages = max(1, int(reuters_cfg.get("max_pages_per_window", 1)))
    keywords = _configured_keywords(news_cfg, query)
    months_per_request = max(1, int(reuters_cfg.get("months_per_request", 1)))
    ordered_windows = _group_windows(list(windows), months_per_request)
    seen_urls = {str(row.get("url")) for row in rows if row.get("url")}
    request_count = 0
    if bool(reuters_cfg.get("newest_first", True)):
        ordered_windows = list(reversed(ordered_windows))

    logger.info("Starting Reuters RapidAPI ingestion across %s monthly windows", len(ordered_windows))
    for idx, (window_start, window_end) in enumerate(ordered_windows, start=1):
        if _max_requests_reached(request_count, reuters_cfg):
            logger.info("Stopping Reuters ingestion after %s configured requests.", request_count)
            break

        logger.info(
            "Fetching Reuters window %s/%s: %s to %s",
            idx,
            len(ordered_windows),
            window_start.date(),
            window_end.date(),
        )
        for keyword in keywords:
            if _max_requests_reached(request_count, reuters_cfg):
                logger.info("Stopping Reuters ingestion after %s configured requests.", request_count)
                break

            fetched_rows, _ = fetch_reuters_rows(
                news_cfg=news_cfg,
                keyword=keyword,
                window_start=window_start,
                window_end=window_end,
                max_records=max_records,
                max_pages=max_pages,
                max_retries=max_retries,
                retry_base_sleep=retry_base_sleep,
            )
            request_count += max_pages
            for row in fetched_rows:
                url = str(row.get("url") or "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                rows.append(row)


def test_reuters_format(
    config_path: str,
    start_date: str,
    end_date: str,
    query_mode: str | None,
    query: str | None,
    keyword: str,
    pages: int,
    max_records: int,
    filter_relevance: bool,
    output_path: str,
    raw_output_path: str,
) -> list[dict]:
    setup_env()
    config = load_config(config_path)
    news_cfg = config["news"]
    news_cfg = dict(news_cfg)
    news_cfg["reuters"] = dict(news_cfg.get("reuters", {}))
    news_cfg["reuters"]["filter_relevance"] = filter_relevance

    resolved_query = _select_query(news_cfg, query_mode=query_mode, query=query)
    resolved_keyword = keyword or str(news_cfg.get("reuters", {}).get("keyword") or resolved_query)
    rows, raw_articles = fetch_reuters_rows(
        news_cfg=news_cfg,
        keyword=resolved_keyword,
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

    raw_output = Path(raw_output_path)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with raw_output.open("w", encoding="utf-8") as f:
        for article in raw_articles:
            f.write(json.dumps(article, ensure_ascii=True) + "\n")

    print(f"Raw articles fetched: {len(raw_articles)}")
    print(f"Rows kept after relevance filter: {len(rows)}")
    print(f"Rows with format issues: {len(row_issues)}")
    print(f"Sample output path: {output}")
    print(f"Raw sample output path: {raw_output}")
    if rows:
        print("Columns:", ", ".join(EXPECTED_NEWS_COLUMNS))
        print("First title:", rows[0].get("title"))
        print("First content chars:", len(str(rows[0].get("content") or "")))
    if raw_articles:
        print("First raw keys:", ", ".join(sorted(raw_articles[0].keys())))
    if row_issues:
        print("Issue samples:")
        for idx, issues in row_issues[:5]:
            print(f"  row {idx}: {'; '.join(issues)}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Reuters RapidAPI output against the project news schema."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-01-31")
    parser.add_argument("--query-mode", choices=["strict", "broad"], default="strict")
    parser.add_argument("--query")
    parser.add_argument("--keyword", default="semiconductor")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=10)
    parser.add_argument(
        "--no-relevance-filter",
        action="store_true",
        help="Keep all Reuters rows in the normalized sample, even if they fail project relevance filters.",
    )
    parser.add_argument("--output", default="artifacts/reuters/reuters_format_sample.jsonl")
    parser.add_argument("--raw-output", default="artifacts/reuters/reuters_raw_sample.jsonl")
    args = parser.parse_args()

    test_reuters_format(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        query_mode=args.query_mode,
        query=args.query,
        keyword=args.keyword,
        pages=args.pages,
        max_records=args.max_records,
        filter_relevance=not args.no_relevance_filter,
        output_path=args.output,
        raw_output_path=args.raw_output,
    )


if __name__ == "__main__":
    main()
