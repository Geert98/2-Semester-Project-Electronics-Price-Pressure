from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.storage import load_dataframe_from_mongo
from src.utils import load_config, setup_env

STRONG_DOMAIN_TERMS = [
    "semiconductor",
    "semiconductors",
    "microchip",
    "microchips",
    "dram",
    "nand",
    "ssd",
]

GENERIC_DOMAIN_TERMS = [
    "chip",
    "chips",
    "memory",
    "electronics",
]

PRESSURE_TERMS = [
    "price",
    "prices",
    "shortage",
    "shortages",
    "supply",
    "demand",
    "tariff",
    "tariffs",
    "export control",
    "export controls",
    "sanction",
    "sanctions",
    "oversupply",
    "inventory",
]

NON_ELECTRONIC_CHIP_PATTERNS = [
    "fish and chips",
    "beer, chips",
    "potato chips",
    "poker chips",
    "microchipped mouthguard",
    "microchipped mouthguards",
    "mouthguard",
    "mouthguards",
]


def _json_default(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _contains_any(text: str, terms: list[str]) -> bool:
    normalized = text.lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized) for term in terms)


def _contains_non_electronic_chip_context(text: str) -> bool:
    normalized = text.lower()
    return any(pattern in normalized for pattern in NON_ELECTRONIC_CHIP_PATTERNS)


def _prefilter_score(title: str, text: str) -> float:
    title = title or ""
    text = text or ""
    score = 0.0

    if _contains_any(title, STRONG_DOMAIN_TERMS):
        score += 3.0
    if _contains_any(title, GENERIC_DOMAIN_TERMS) and _contains_any(title, PRESSURE_TERMS):
        score += 2.0
    if _contains_any(text, STRONG_DOMAIN_TERMS):
        score += 1.0
    if _contains_any(text, GENERIC_DOMAIN_TERMS) and _contains_any(text, PRESSURE_TERMS):
        score += 0.5

    if _contains_non_electronic_chip_context(f"{title} {text[:1000]}"):
        score -= 3.0

    return score


def export_news_for_ai_lab(
    config_path: str = "configs/config.yaml",
    output_path: str = "ai_lab/upload_bundle/news_to_enrich.jsonl",
    limit: int | None = None,
    max_chars: int = 12000,
    prefilter: bool = True,
) -> Path:
    setup_env()
    config = load_config(config_path)
    mongo_cfg = config["storage"]["mongo"]

    source_collection = mongo_cfg["test_clean_news_collection"]
    enriched_collection = mongo_cfg["test_enriched_news_collection"]

    clean_df = load_dataframe_from_mongo(config, source_collection, sort_by="published_at")
    enriched_df = load_dataframe_from_mongo(config, enriched_collection)

    if clean_df.empty:
        raise ValueError(f"No articles found in MongoDB collection: {source_collection}")

    existing_urls = set(enriched_df["url"].dropna()) if "url" in enriched_df.columns else set()
    clean_df = clean_df.dropna(subset=["url"]).drop_duplicates(subset=["url"], keep="last")
    pending_df = clean_df[~clean_df["url"].isin(existing_urls)].copy()
    if prefilter:
        pending_df["prefilter_score"] = pending_df.apply(
            lambda row: _prefilter_score(
                str(row.get("title") or ""),
                str(row.get("clean_text") or "")[:max_chars],
            ),
            axis=1,
        )
        pending_df = pending_df[pending_df["prefilter_score"] >= 1.0]
        pending_df = pending_df.sort_values(["prefilter_score", "published_at"], ascending=[False, True])
    if limit is not None:
        pending_df = pending_df.head(limit)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for _, row in pending_df.iterrows():
            clean_text = str(row.get("clean_text") or "")
            record = {
                "url": row.get("url"),
                "title": row.get("title"),
                "source": row.get("source"),
                "published_at": _json_default(row.get("published_at")),
                "month": _json_default(row.get("month")),
                "text": clean_text[:max_chars],
                "content_char_count": len(clean_text),
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Exported {len(pending_df)} articles to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cleaned news articles for AI-LAB batch enrichment.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--no-prefilter", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    export_cfg = config.get("ai_lab_export", {})

    export_news_for_ai_lab(
        config_path=args.config,
        output_path=args.output or export_cfg.get("output_path", "ai_lab/upload_bundle/news_to_enrich.jsonl"),
        limit=args.limit if args.limit is not None else export_cfg.get("limit"),
        max_chars=args.max_chars if args.max_chars is not None else export_cfg.get("max_chars", 12000),
        prefilter=False if args.no_prefilter else export_cfg.get("prefilter", True),
    )


if __name__ == "__main__":
    main()
