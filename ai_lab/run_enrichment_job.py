from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import requests


SCHEMA_KEYS = [
    "is_relevant",
    "relevance_score",
    "sentiment_score",
    "price_pressure_direction",
    "price_pressure_strength",
    "primary_topic",
    "reason_short",
]


def _prompt(record: dict) -> str:
    return (
        "Classify this article for an electronics and semiconductor price-pressure model.\n"
        "Return JSON only with these keys: "
        + ", ".join(SCHEMA_KEYS)
        + ".\n"
        "Rules:\n"
        "- is_relevant: true only if the article is about semiconductor/electronics prices, supply, demand, tariffs, export controls, inventories, or production.\n"
        "- price_pressure_direction must be upward, downward, or neutral.\n"
        "- price_pressure_strength must be an integer from 0 to 5.\n"
        "- primary_topic must be shortage, tariff, demand, supply_chain, export_controls, oversupply, or other.\n"
        "- Do not treat generic negative news as upward pressure unless it affects electronics/semiconductor markets.\n\n"
        f"Title: {record.get('title', '')}\n"
        f"Source: {record.get('source', '')}\n"
        f"Published at: {record.get('published_at', '')}\n"
        f"Article text:\n{record.get('text', '')}"
    )


def _call_openai_compatible(base_url: str, api_key: str, model: str, record: dict, timeout: int) -> dict:
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise data labeling assistant. Return valid JSON only.",
                },
                {"role": "user", "content": _prompt(record)},
            ],
            "temperature": 0,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def enrich_file(input_path: str, output_path: str, model: str, base_url: str, api_key: str, timeout: int) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0

    with input_file.open("r", encoding="utf-8") as src, output_file.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            print(f"Processing line {line_number}: {record.get('title', '')[:100]}", flush=True)
            try:
                parsed = _call_openai_compatible(base_url, api_key, model, record, timeout)
                output = {
                    "url": record.get("url"),
                    "title": record.get("title"),
                    "source": record.get("source"),
                    "published_at": record.get("published_at"),
                    "month": record.get("month"),
                    "content_char_count": record.get("content_char_count", 0),
                    "scored_char_count": len(record.get("text", "")),
                    "ai_backend": "aau_ai_lab",
                    "enrichment_version": f"aau_ai_lab_{model}_v1",
                    "enriched_at": datetime.now(UTC).isoformat(),
                    **parsed,
                }
                dst.write(json.dumps(output, ensure_ascii=True) + "\n")
                dst.flush()
                success_count += 1
            except Exception as exc:
                failure_count += 1
                print(f"Failed line {line_number}: {exc}", flush=True)
            time.sleep(0.1)

    print(f"Finished enrichment. success={success_count}, failures={failure_count}", flush=True)
    if success_count == 0 and failure_count > 0:
        raise RuntimeError("All enrichment calls failed. Check model, endpoint, and job logs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI-LAB article enrichment using an OpenAI-compatible endpoint.")
    parser.add_argument("--input", default="news_to_enrich.jsonl")
    parser.add_argument("--output", default="news_enriched.jsonl")
    parser.add_argument("--model", default=os.getenv("AI_LAB_MODEL", "CHANGE_ME"))
    parser.add_argument("--base-url", default=os.getenv("AI_LAB_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("AI_LAB_API_KEY", "EMPTY"))
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    enrich_file(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
