from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.operations import UpdateOne


def _get_storage_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("storage", {})


def get_mongo_settings(config: dict[str, Any]) -> tuple[str, str]:
    storage_cfg = _get_storage_config(config).get("mongo", {})
    default_uri = storage_cfg.get("uri", "mongodb://localhost:27017")
    default_db_name = storage_cfg.get("database", "electronics_price_pressure")
    uri = os.getenv("MONGO_URI") or default_uri
    db_name = os.getenv("MONGO_DB_NAME") or default_db_name
    return uri, db_name


def get_sqlite_path(config: dict[str, Any]) -> Path:
    storage_cfg = _get_storage_config(config).get("sqlite", {})
    default_path = storage_cfg.get("fred_db_path", "data/db/fred.sqlite3")
    return Path(os.getenv("FRED_SQLITE_PATH", default_path))


def _to_mongo_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    return value


def dataframe_to_mongo_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        records.append({key: _to_mongo_value(value) for key, value in record.items()})
    return records


def _deduplicate_collection_by_keys(collection: Any, key_columns: list[str]) -> int:
    group_id = {column: f"${column}" for column in key_columns}
    pipeline = [
        {"$match": {column: {"$nin": [None, ""]} for column in key_columns}},
        {"$group": {"_id": group_id, "ids": {"$push": "$_id"}, "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}},
    ]

    deleted_count = 0
    for duplicate_group in collection.aggregate(pipeline):
        ids_to_delete = duplicate_group["ids"][1:]
        if ids_to_delete:
            result = collection.delete_many({"_id": {"$in": ids_to_delete}})
            deleted_count += result.deleted_count

    return deleted_count


def save_dataframe_to_mongo(df: pd.DataFrame, config: dict[str, Any], collection_name: str) -> None:
    uri, db_name = get_mongo_settings(config)
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        collection = client[db_name][collection_name]
        collection.delete_many({})

        if not df.empty:
            collection.insert_many(dataframe_to_mongo_records(df))
    finally:
        client.close()


def upsert_dataframe_to_mongo(
    df: pd.DataFrame,
    config: dict[str, Any],
    collection_name: str,
    key_columns: list[str],
) -> int:
    uri, db_name = get_mongo_settings(config)
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        collection = client[db_name][collection_name]

        if df.empty:
            return 0

        unique_df = df.dropna(subset=key_columns).drop_duplicates(
            subset=key_columns,
            keep="last",
        )
        records = dataframe_to_mongo_records(unique_df)
        operations = []
        for record in records:
            filter_query = {column: record[column] for column in key_columns}
            operations.append(UpdateOne(filter_query, {"$set": record}, upsert=True))

        if not operations:
            return 0

        _deduplicate_collection_by_keys(collection, key_columns)
        collection.create_index([(column, 1) for column in key_columns], unique=True)
        result = collection.bulk_write(operations, ordered=False)
        return result.upserted_count + result.modified_count
    finally:
        client.close()


def load_dataframe_from_mongo(
    config: dict[str, Any],
    collection_name: str,
    sort_by: str | None = None,
) -> pd.DataFrame:
    uri, db_name = get_mongo_settings(config)
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        records = list(client[db_name][collection_name].find({}, {"_id": 0}))
    finally:
        client.close()

    df = pd.DataFrame(records)
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)
    return df


def save_dataframe_to_sqlite(df: pd.DataFrame, sqlite_path: Path, table_name: str) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        df.to_sql(table_name, connection, if_exists="replace", index=False)


def load_dataframe_from_sqlite(sqlite_path: Path, table_name: str) -> pd.DataFrame:
    if not sqlite_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(sqlite_path) as connection:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
        except Exception:
            return pd.DataFrame()
