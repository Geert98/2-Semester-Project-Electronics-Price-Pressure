from __future__ import annotations

import logging
import json
import os
import time
from datetime import UTC, datetime

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.storage import load_dataframe_from_mongo, upsert_dataframe_to_mongo
from src.utils import load_config, setup_env

logger = logging.getLogger(__name__)

RULES_ENRICHMENT_VERSION = "rules_v2"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"

RELEVANCE_TERMS = [
    "semiconductor",
    "semiconductors",
    "microchip",
    "microchips",
    "dram",
    "nand",
    "ssd",
]

GENERIC_RELEVANCE_TERMS = [
    "chip",
    "chips",
    "memory",
    "electronics",
]

UPWARD_PRESSURE_TERMS = [
    "shortage",
    "shortages",
    "supply disruption",
    "supply chain pressure",
    "tariff",
    "tariffs",
    "export control",
    "export controls",
    "sanction",
    "sanctions",
    "chip ban",
    "production halt",
    "capacity constraint",
    "lead time",
    "price increase",
    "rising prices",
    "higher prices",
    "demand surge",
    "ai demand",
    "inventory drawdown",
]

DOWNWARD_PRESSURE_TERMS = [
    "oversupply",
    "glut",
    "inventory surplus",
    "weak demand",
    "price cut",
    "price cuts",
    "falling prices",
    "lower prices",
    "downturn",
    "slowdown",
    "excess inventory",
]

TOPIC_TERMS = {
    "shortage": ["shortage", "shortages", "capacity constraint", "lead time"],
    "tariff": ["tariff", "tariffs", "duties"],
    "demand": ["demand", "ai demand", "demand surge", "data center"],
    "supply_chain": ["supply chain", "logistics", "shipping", "disruption"],
    "export_controls": ["export control", "export controls", "sanction", "sanctions", "chip ban"],
    "oversupply": ["oversupply", "glut", "excess inventory", "inventory surplus"],
}


def _count_terms(text: str, terms: list[str]) -> int:
    if not isinstance(text, str) or not text:
        return 0
    normalized = text.lower()
    return sum(1 for term in terms if term in normalized)


def _bounded_relevance_score(text: str) -> float:
    specific_hits = _count_terms(text, RELEVANCE_TERMS)
    generic_hits = _count_terms(text, GENERIC_RELEVANCE_TERMS)
    pressure_hits = _count_terms(text, UPWARD_PRESSURE_TERMS + DOWNWARD_PRESSURE_TERMS)
    generic_score = 0.5 if generic_hits and pressure_hits else 0.0
    return min((specific_hits + generic_score) / 2, 1.0)


def _prefilter_relevance_score(text: str) -> float:
    """
    Conservative prefilter used before spending LLM tokens.

    Ambiguous terms such as "memory", "chip", and "electronics" only count
    when they appear together with price, supply, demand, tariff, or shortage
    language.
    """
    if not isinstance(text, str) or not text:
        return 0.0

    specific_hits = _count_terms(text, RELEVANCE_TERMS)
    generic_hits = _count_terms(text, GENERIC_RELEVANCE_TERMS)
    pressure_hits = _count_terms(text, UPWARD_PRESSURE_TERMS + DOWNWARD_PRESSURE_TERMS)

    if specific_hits:
        return 1.0
    if generic_hits and pressure_hits:
        return 0.75
    return 0.0


def _direction_from_score(score: int) -> str:
    if score > 0:
        return "upward"
    if score < 0:
        return "downward"
    return "neutral"


def _rules_enrichment(article: pd.Series, analyzer: SentimentIntensityAnalyzer, max_chars: int) -> dict:
    """
    Produce structured price-pressure labels from article text.

    This is the deterministic fallback. The output schema is intentionally the
    same shape an LLM enrichment step should produce, so it can be replaced
    without changing downstream feature engineering.
    """
    clean_text = str(article.get("clean_text") or "")
    title = str(article.get("title") or "")
    text_for_scoring = clean_text[:max_chars]

    upward_count = _count_terms(text_for_scoring, UPWARD_PRESSURE_TERMS)
    downward_count = _count_terms(text_for_scoring, DOWNWARD_PRESSURE_TERMS)
    relevance_score = _bounded_relevance_score(text_for_scoring)
    is_relevant = relevance_score >= 0.5
    direction_score = upward_count - downward_count if is_relevant else 0

    topic_counts = {
        topic: _count_terms(text_for_scoring, terms)
        for topic, terms in TOPIC_TERMS.items()
    }
    primary_topic = max(topic_counts, key=topic_counts.get)
    if topic_counts[primary_topic] == 0 or not is_relevant:
        primary_topic = "other"

    sentiment = analyzer.polarity_scores(text_for_scoring)["compound"] if text_for_scoring else 0.0

    return {
        "url": article.get("url"),
        "title": title,
        "source": article.get("source"),
        "published_at": article.get("published_at"),
        "month": article.get("month"),
        "content_char_count": len(clean_text),
        "scored_char_count": len(text_for_scoring),
        "is_relevant": is_relevant,
        "relevance_score": relevance_score,
        "sentiment": sentiment,
        "is_negative": sentiment < 0,
        "price_pressure_direction": _direction_from_score(direction_score),
        "price_pressure_direction_score": direction_score,
        "price_pressure_strength": min(abs(direction_score), 5),
        "primary_topic": primary_topic,
        "ai_backend": "rules",
        "enrichment_version": RULES_ENRICHMENT_VERSION,
        "reason_short": "Deterministic keyword fallback.",
        "enriched_at": datetime.now(UTC),
        **{f"topic_{topic}_count": count for topic, count in topic_counts.items()},
        **{f"topic_{topic}_flag": int(count > 0) for topic, count in topic_counts.items()},
    }


ARTICLE_ENRICHMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_relevant": {"type": "boolean"},
        "relevance_score": {"type": "number"},
        "sentiment_score": {"type": "number"},
        "price_pressure_direction": {
            "type": "string",
            "enum": ["upward", "downward", "neutral"],
        },
        "price_pressure_strength": {"type": "integer"},
        "primary_topic": {
            "type": "string",
            "enum": [
                "shortage",
                "tariff",
                "demand",
                "supply_chain",
                "export_controls",
                "oversupply",
                "other",
            ],
        },
        "reason_short": {"type": "string"},
    },
    "required": [
        "is_relevant",
        "relevance_score",
        "sentiment_score",
        "price_pressure_direction",
        "price_pressure_strength",
        "primary_topic",
        "reason_short",
    ],
    "additionalProperties": False,
}


def _extract_response_text(payload: dict) -> str:
    output_items = payload.get("output", [])
    for item in output_items:
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"]
    if payload.get("output_text"):
        return payload["output_text"]
    raise ValueError(f"OpenAI response did not contain output text: {payload}")


def _retry_after_seconds(response: requests.Response, fallback_sleep: float) -> float:
    retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if retry_after:
        try:
            return max(float(retry_after), fallback_sleep)
        except ValueError:
            pass

    reset_after = (
        response.headers.get("x-ratelimit-reset-requests")
        or response.headers.get("x-ratelimit-reset-tokens")
    )
    if reset_after:
        try:
            return max(float(reset_after), fallback_sleep)
        except ValueError:
            pass

    return fallback_sleep


class RateLimitExceededError(RuntimeError):
    pass


def _openai_enrichment(
    article: pd.Series,
    api_key: str,
    model: str,
    max_chars: int,
    timeout_seconds: float,
    max_retries: int,
    base_retry_sleep: float,
) -> dict:
    clean_text = str(article.get("clean_text") or "")
    title = str(article.get("title") or "")
    source = str(article.get("source") or "")
    published_at = str(article.get("published_at") or "")
    text_for_ai = clean_text[:max_chars]

    prompt = (
        "Classify this article for an electronics and semiconductor price-pressure model.\n"
        "Focus on whether the article is relevant to semiconductor/electronics prices, "
        "and whether it implies upward, downward, or neutral price pressure. "
        "Do not treat generic negative news as upward price pressure unless it affects "
        "electronics supply, demand, trade restrictions, inventories, or production costs.\n\n"
        f"Title: {title}\n"
        f"Source: {source}\n"
        f"Published at: {published_at}\n"
        f"Article text:\n{text_for_ai}"
    )

    request_payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are a precise data-labeling assistant. Return only the structured "
                    "classification requested by the JSON schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "article_price_pressure_enrichment",
                "strict": True,
                "schema": ARTICLE_ENRICHMENT_SCHEMA,
            }
        },
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                OPENAI_RESPONSES_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
                timeout=timeout_seconds,
            )
            if response.status_code == 429 and attempt < max_retries:
                time.sleep(_retry_after_seconds(response, base_retry_sleep * (2**attempt)))
                continue
            response.raise_for_status()
            parsed = json.loads(_extract_response_text(response.json()))
            primary_topic = parsed["primary_topic"]

            return {
                "url": article.get("url"),
                "title": title,
                "source": article.get("source"),
                "published_at": article.get("published_at"),
                "month": article.get("month"),
                "content_char_count": len(clean_text),
                "scored_char_count": len(text_for_ai),
                "is_relevant": bool(parsed["is_relevant"]),
                "relevance_score": float(parsed["relevance_score"]),
                "sentiment": float(parsed["sentiment_score"]),
                "is_negative": float(parsed["sentiment_score"]) < 0,
                "price_pressure_direction": parsed["price_pressure_direction"],
                "price_pressure_direction_score": {
                    "upward": 1,
                    "neutral": 0,
                    "downward": -1,
                }[parsed["price_pressure_direction"]],
                "price_pressure_strength": int(parsed["price_pressure_strength"]),
                "primary_topic": primary_topic,
                "ai_backend": "openai",
                "enrichment_version": f"openai_{model}_v1",
                "reason_short": parsed["reason_short"],
                "enriched_at": datetime.now(UTC),
                **{f"topic_{topic}_count": int(topic == primary_topic) for topic in TOPIC_TERMS},
                **{f"topic_{topic}_flag": int(topic == primary_topic) for topic in TOPIC_TERMS},
            }
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(base_retry_sleep * (2**attempt))

    raise RuntimeError(f"OpenAI enrichment failed after retries: {last_error}") from last_error


def _groq_enrichment(
    article: pd.Series,
    api_key: str,
    model: str,
    max_chars: int,
    timeout_seconds: float,
    max_retries: int,
    base_retry_sleep: float,
) -> dict:
    clean_text = str(article.get("clean_text") or "")
    title = str(article.get("title") or "")
    source = str(article.get("source") or "")
    published_at = str(article.get("published_at") or "")
    text_for_ai = clean_text[:max_chars]

    prompt = (
        "Classify this article for an electronics and semiconductor price-pressure model.\n"
        "Focus on whether the article is relevant to semiconductor/electronics prices, "
        "and whether it implies upward, downward, or neutral price pressure. "
        "Do not treat generic negative news as upward price pressure unless it affects "
        "electronics supply, demand, trade restrictions, inventories, or production costs.\n\n"
        f"Title: {title}\n"
        f"Source: {source}\n"
        f"Published at: {published_at}\n"
        f"Article text:\n{text_for_ai}"
    )

    request_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise data-labeling assistant. Return only the structured "
                    "classification requested by the JSON schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "article_price_pressure_enrichment",
                "strict": True,
                "schema": ARTICLE_ENRICHMENT_SCHEMA,
            },
        },
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
                timeout=timeout_seconds,
            )
            if response.status_code == 429 and attempt < max_retries:
                sleep_for = _retry_after_seconds(response, base_retry_sleep * (2**attempt))
                logger.warning("Groq rate limit hit. Sleeping %.1f seconds before retry.", sleep_for)
                time.sleep(sleep_for)
                continue
            if response.status_code == 429:
                raise RateLimitExceededError(response.text)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            primary_topic = parsed["primary_topic"]

            return {
                "url": article.get("url"),
                "title": title,
                "source": article.get("source"),
                "published_at": article.get("published_at"),
                "month": article.get("month"),
                "content_char_count": len(clean_text),
                "scored_char_count": len(text_for_ai),
                "is_relevant": bool(parsed["is_relevant"]),
                "relevance_score": float(parsed["relevance_score"]),
                "sentiment": float(parsed["sentiment_score"]),
                "is_negative": float(parsed["sentiment_score"]) < 0,
                "price_pressure_direction": parsed["price_pressure_direction"],
                "price_pressure_direction_score": {
                    "upward": 1,
                    "neutral": 0,
                    "downward": -1,
                }[parsed["price_pressure_direction"]],
                "price_pressure_strength": int(parsed["price_pressure_strength"]),
                "primary_topic": primary_topic,
                "ai_backend": "groq",
                "enrichment_version": f"groq_{model}_v1",
                "reason_short": parsed["reason_short"],
                "enriched_at": datetime.now(UTC),
                **{f"topic_{topic}_count": int(topic == primary_topic) for topic in TOPIC_TERMS},
                **{f"topic_{topic}_flag": int(topic == primary_topic) for topic in TOPIC_TERMS},
            }
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(base_retry_sleep * (2**attempt))

    raise RuntimeError(f"Groq enrichment failed after retries: {last_error}") from last_error


def enrich_news_ai(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Enrich cleaned articles with article-level relevance and price-pressure labels.
    """
    setup_env()
    config = load_config(config_path)
    mongo_cfg = config["storage"]["mongo"]
    ai_cfg = config.get("ai_enrichment", {})

    if not bool(ai_cfg.get("enabled", True)):
        logger.info("AI enrichment is disabled in config.")
        return pd.DataFrame()

    backend = ai_cfg.get("backend", "rules")
    if backend not in {"rules", "openai", "groq"}:
        raise ValueError(
            f"Unsupported ai_enrichment.backend={backend!r}. "
            "Supported values are 'rules', 'openai', and 'groq'."
        )

    source_collection = mongo_cfg["test_clean_news_collection"]
    target_collection = mongo_cfg["test_enriched_news_collection"]
    max_articles = int(ai_cfg.get("max_articles_per_run", 500))
    max_chars = int(ai_cfg.get("max_chars_per_article", 20000))
    model = str(ai_cfg.get("model", "gpt-4o-mini"))
    timeout_seconds = float(ai_cfg.get("request_timeout_seconds", 60))
    max_retries = int(ai_cfg.get("max_retries", 2))
    request_sleep_seconds = float(ai_cfg.get("request_sleep_seconds", 0))
    stop_on_rate_limit = bool(ai_cfg.get("stop_on_rate_limit", True))
    min_prefilter_relevance_score = float(ai_cfg.get("min_prefilter_relevance_score", 0))
    enrichment_version = RULES_ENRICHMENT_VERSION
    api_key = ""

    if backend in {"openai", "groq"}:
        default_api_key_env = "OPENAI_API_KEY" if backend == "openai" else "GROQ_API_KEY"
        api_key_env = ai_cfg.get("api_key_env", default_api_key_env)
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"ai_enrichment.backend is {backend!r}, but {api_key_env} is not set. "
                "Add it to your environment or .env file."
            )
        enrichment_version = f"{backend}_{model}_v1"

    clean_df = load_dataframe_from_mongo(config, source_collection, sort_by="published_at")
    if clean_df.empty:
        logger.warning("No cleaned articles found in MongoDB collection %s", source_collection)
        return pd.DataFrame()

    existing_df = load_dataframe_from_mongo(config, target_collection)
    if {"url", "enrichment_version"}.issubset(existing_df.columns):
        current_existing = existing_df[existing_df["enrichment_version"] == enrichment_version]
        existing_urls = set(current_existing["url"].dropna())
    else:
        existing_urls = set()

    clean_df = clean_df.dropna(subset=["url"]).drop_duplicates(subset=["url"], keep="last")
    pending_df = clean_df[~clean_df["url"].isin(existing_urls)].copy()
    if backend in {"openai", "groq"} and min_prefilter_relevance_score > 0:
        pending_df["prefilter_relevance_score"] = pending_df["clean_text"].fillna("").apply(
            lambda text: _prefilter_relevance_score(str(text)[:max_chars])
        )
        pending_df = pending_df[pending_df["prefilter_relevance_score"] >= min_prefilter_relevance_score]

    pending_df = pending_df.head(max_articles)
    if pending_df.empty:
        logger.info("No new articles require enrichment.")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()
    enriched_rows = []
    for _, article in pending_df.iterrows():
        if backend == "openai":
            enriched_rows.append(
                _openai_enrichment(
                    article,
                    api_key=api_key,
                    model=model,
                    max_chars=max_chars,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    base_retry_sleep=request_sleep_seconds or 1,
                )
            )
        elif backend == "groq":
            try:
                enriched_rows.append(
                    _groq_enrichment(
                        article,
                        api_key=api_key,
                        model=model,
                        max_chars=max_chars,
                        timeout_seconds=timeout_seconds,
                        max_retries=max_retries,
                        base_retry_sleep=request_sleep_seconds or 1,
                    )
                )
            except RateLimitExceededError as exc:
                logger.warning("Stopping Groq enrichment early because rate limit persisted: %s", exc)
                if stop_on_rate_limit:
                    break
                raise
        else:
            enriched_rows.append(_rules_enrichment(article, analyzer, max_chars=max_chars))
        if request_sleep_seconds > 0:
            time.sleep(request_sleep_seconds)

    if not enriched_rows:
        logger.warning("No articles were enriched in this run.")
        return pd.DataFrame()
    enriched_df = pd.DataFrame(enriched_rows)

    changed_count = upsert_dataframe_to_mongo(
        enriched_df,
        config,
        target_collection,
        key_columns=["url"],
    )
    logger.info(
        "Upserted enriched news to MongoDB collection %s (%s rows, %s changed)",
        target_collection,
        len(enriched_df),
        changed_count,
    )
    return enriched_df


if __name__ == "__main__":
    result = enrich_news_ai()
    print(f"Enriched rows: {len(result)}")
    if not result.empty:
        print(
            result[
                [
                    "title",
                    "is_relevant",
                    "relevance_score",
                    "price_pressure_direction",
                    "price_pressure_strength",
                    "primary_topic",
                    "ai_backend",
                ]
            ].to_string(index=False)
        )
