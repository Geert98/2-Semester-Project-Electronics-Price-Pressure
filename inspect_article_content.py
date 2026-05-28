from __future__ import annotations

import argparse
from pathlib import Path

from pymongo import MongoClient

from src.storage import get_mongo_settings
from src.utils import load_config, setup_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample news articles from MongoDB and dump their content to a text file."
    )
    parser.add_argument(
        "--provider",
        choices=["guardian", "newsapi"],
        required=True,
        help="Which provider's rows to inspect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of rows to sample from MongoDB.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/article_content_dump.txt",
        help="Path to the output text file.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the project config file.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="MongoDB collection to read from. Defaults to test_news_collection in the config.",
    )
    return parser.parse_args()


def resolve_collection_name(config: dict, collection_override: str | None) -> str:
    if collection_override:
        return collection_override

    mongo_cfg = config["storage"]["mongo"]
    return mongo_cfg.get("test_news_collection") or mongo_cfg["raw_news_collection"]


def fetch_articles(config: dict, collection_name: str, provider: str, limit: int) -> list[dict]:
    uri, db_name = get_mongo_settings(config)
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        collection = client[db_name][collection_name]
        cursor = (
            collection.find(
                {"provider": provider},
                {
                    "_id": 0,
                    "provider": 1,
                    "title": 1,
                    "url": 1,
                    "source": 1,
                    "seen_date": 1,
                    "published_at": 1,
                    "language": 1,
                    "content": 1,
                },
            )
            .sort([("published_at", -1), ("seen_date", -1)])
            .limit(max(1, limit))
        )
        return list(cursor)
    finally:
        client.close()


def format_article(index: int, article: dict) -> str:
    content = article.get("content") or ""
    lines = [
        f"ARTICLE {index}",
        f"provider: {article.get('provider', '')}",
        f"title: {article.get('title', '')}",
        f"source: {article.get('source', '')}",
        f"url: {article.get('url', '')}",
        f"seen_date: {article.get('seen_date', '')}",
        f"published_at: {article.get('published_at', '')}",
        f"language: {article.get('language', '')}",
        "content:",
        content,
        "",
        "=" * 80,
        "",
    ]
    return "\n".join(str(line) for line in lines)


def write_dump(output_path: str, articles: list[dict], provider: str, collection_name: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        f"provider: {provider}",
        f"collection: {collection_name}",
        f"article_count: {len(articles)}",
        "",
        "=" * 80,
        "",
    ]

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header))
        for index, article in enumerate(articles, start=1):
            handle.write(format_article(index, article))

    return path


def main() -> None:
    setup_env()
    args = parse_args()
    config = load_config(args.config)
    collection_name = resolve_collection_name(config, args.collection)

    articles = fetch_articles(
        config=config,
        collection_name=collection_name,
        provider=args.provider,
        limit=args.limit,
    )

    output_path = write_dump(
        output_path=args.output,
        articles=articles,
        provider=args.provider,
        collection_name=collection_name,
    )

    print(f"Wrote {len(articles)} articles to {output_path}")
    if not articles:
        print("No matching rows were found for that provider.")


if __name__ == "__main__":
    main()
