from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime

from pymongo import MongoClient

from src.enrich_hackernews_links import enrich_hackernews_links
from src.utils import load_config, setup_env


def _mongo_settings(config: dict) -> tuple[str, str]:
    mongo_cfg = config["storage"]["mongo"]
    uri = os.getenv("MONGO_URI") or mongo_cfg.get("uri", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME") or mongo_cfg.get("database", "electronics_price_pressure")
    return uri, db_name


def _status_counts(config: dict) -> dict[str, int]:
    mongo_cfg = config["storage"]["mongo"]
    uri, db_name = _mongo_settings(config)
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        collection = client[db_name][mongo_cfg["test_news_collection"]]
        docs = collection.find(
            {"provider": "hackernews"},
            {"_id": 0, "linked_fetch_status": 1},
        )
        counts: dict[str, int] = defaultdict(int)
        total = 0
        for doc in docs:
            total += 1
            status = doc.get("linked_fetch_status") or "missing"
            counts[status] += 1
        counts["total"] = total
        return dict(counts)
    finally:
        client.close()


def _print_status(label: str, status: dict[str, int]) -> None:
    ordered = {
        key: status[key]
        for key in sorted(status)
        if key != "total"
    }
    ordered = {"total": status.get("total", 0), **ordered}
    print(f"{label}: {json.dumps(ordered, sort_keys=True)}", flush=True)


def _run_module(module: str) -> None:
    print(f"Running {module} at {datetime.now(UTC).isoformat()}", flush=True)
    subprocess.run([sys.executable, "-B", "-m", module], check=True)


def run_backfill(
    config_path: str,
    batch_size: int,
    timeout_seconds: float,
    min_interval_seconds: float,
    progress_every: int,
    max_batches: int | None,
    run_postprocess: bool,
) -> None:
    setup_env()
    config = load_config(config_path)

    print(f"HN link backfill started at {datetime.now(UTC).isoformat()}", flush=True)
    status = _status_counts(config)
    _print_status("Initial status", status)

    batch = 0
    while status.get("missing", 0) > 0:
        if max_batches is not None and batch >= max_batches:
            print(f"Stopping after configured max_batches={max_batches}", flush=True)
            break

        batch += 1
        limit = min(batch_size, status.get("missing", batch_size))
        print(
            f"Starting batch {batch}: limit={limit}, "
            f"timeout_seconds={timeout_seconds}, min_interval_seconds={min_interval_seconds}",
            flush=True,
        )

        stats = enrich_hackernews_links(
            config_path=config_path,
            limit=limit,
            retry_failed=False,
            timeout_seconds_override=timeout_seconds,
            min_interval_seconds_override=min_interval_seconds,
            progress_every=progress_every,
        )
        print(f"Batch {batch} result: {json.dumps(stats, sort_keys=True)}", flush=True)

        if stats.get("checked", 0) == 0:
            print("No rows checked in batch; stopping.", flush=True)
            break

        status = _status_counts(config)
        _print_status(f"Status after batch {batch}", status)

    if run_postprocess:
        _run_module("src.preprocess")
        _run_module("src.feature_engineering")

    final_status = _status_counts(config)
    _print_status("Final status", final_status)
    print(f"HN link backfill finished at {datetime.now(UTC).isoformat()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hacker News linked-content batches until done.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--timeout-seconds", type=float, default=8)
    parser.add_argument("--min-interval-seconds", type=float, default=0.75)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args()

    run_backfill(
        config_path=args.config,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        min_interval_seconds=args.min_interval_seconds,
        progress_every=args.progress_every,
        max_batches=args.max_batches,
        run_postprocess=args.postprocess,
    )


if __name__ == "__main__":
    main()
