from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.storage import load_dataframe_from_mongo
from src.utils import load_config, setup_env


def _json_default(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def export_news_for_ai_lab(
    config_path: str = "configs/config.yaml",
    output_path: str = "artifacts/ai_lab/news_to_enrich.jsonl",
    limit: int | None = 100,
    max_chars: int = 12000,
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
    parser.add_argument("--output", default="artifacts/ai_lab/news_to_enrich.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=12000)
    args = parser.parse_args()

    export_news_for_ai_lab(
        config_path=args.config,
        output_path=args.output,
        limit=args.limit,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
