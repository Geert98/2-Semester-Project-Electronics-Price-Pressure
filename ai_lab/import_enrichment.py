from __future__ import annotations

import argparse
import json
from glob import glob
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

from src.storage import get_mongo_settings, upsert_dataframe_to_mongo
from src.utils import load_config, setup_env


TOPICS = [
    "shortage",
    "tariff",
    "demand",
    "supply_chain",
    "export_controls",
    "oversupply",
]


def _direction_score(direction: str) -> int:
    return {"upward": 1, "neutral": 0, "downward": -1}.get(direction, 0)


def _normalize_record(record: dict) -> dict:
    direction = str(record.get("price_pressure_direction", "neutral")).lower()
    primary_topic = str(record.get("primary_topic", "other")).lower()
    sentiment = float(record.get("sentiment", record.get("sentiment_score", 0)) or 0)

    normalized = {
        "url": record.get("url"),
        "title": record.get("title"),
        "source": record.get("source"),
        "published_at": record.get("published_at"),
        "month": record.get("month"),
        "content_char_count": int(record.get("content_char_count", 0) or 0),
        "scored_char_count": int(record.get("scored_char_count", 0) or 0),
        "is_relevant": bool(record.get("is_relevant", False)),
        "relevance_score": float(record.get("relevance_score", 0) or 0),
        "sentiment": sentiment,
        "is_negative": sentiment < 0,
        "price_pressure_direction": direction,
        "price_pressure_direction_score": _direction_score(direction),
        "price_pressure_strength": int(record.get("price_pressure_strength", 0) or 0),
        "primary_topic": primary_topic,
        "ai_backend": record.get("ai_backend", "aau_ai_lab"),
        "enrichment_version": record.get("enrichment_version", "aau_ai_lab_v1"),
        "reason_short": record.get("reason_short", ""),
        "enriched_at": record.get("enriched_at", datetime.now(UTC)),
    }

    for topic in TOPICS:
        normalized[f"topic_{topic}_count"] = int(primary_topic == topic)
        normalized[f"topic_{topic}_flag"] = int(primary_topic == topic)

    return normalized


def _resolve_input_files(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_dir():
        batch_files = sorted(path.glob("news_enriched_*.jsonl"))
        if batch_files:
            return batch_files
        single_file = path / "news_enriched.jsonl"
        return [single_file] if single_file.exists() else []
    if any(char in input_path for char in "*?[]"):
        return [Path(match) for match in sorted(glob(input_path))]
    return [path]


def import_ai_lab_enrichment(
    config_path: str = "configs/config.yaml",
    input_path: str = "ai_lab/upload_bundle/results",
    replace: bool = False,
) -> pd.DataFrame:
    setup_env()
    config = load_config(config_path)
    target_collection = config["storage"]["mongo"]["test_enriched_news_collection"]

    input_files = _resolve_input_files(input_path)
    if not input_files:
        raise FileNotFoundError(f"Missing AI-LAB enrichment output: {input_path}")

    records = []
    for path in input_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(_normalize_record(json.loads(line)))

    df = pd.DataFrame(records)
    if df.empty:
        print("No enrichment records found.")
        return df

    if replace:
        uri, db_name = get_mongo_settings(config)
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        try:
            client.admin.command("ping")
            delete_result = client[db_name][target_collection].delete_many({})
            print(f"Deleted {delete_result.deleted_count} existing records from {target_collection}")
        finally:
            client.close()

    changed_count = upsert_dataframe_to_mongo(
        df,
        config,
        target_collection,
        key_columns=["url"],
    )
    print(f"Imported {len(df)} enrichment records from {len(input_files)} file(s) into {target_collection} ({changed_count} changed)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Import AI-LAB JSONL enrichment output into MongoDB.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", default="ai_lab/upload_bundle/results")
    parser.add_argument("--replace", action="store_true", help="Delete existing target collection contents before importing.")
    args = parser.parse_args()

    import_ai_lab_enrichment(config_path=args.config, input_path=args.input, replace=args.replace)


if __name__ == "__main__":
    main()
