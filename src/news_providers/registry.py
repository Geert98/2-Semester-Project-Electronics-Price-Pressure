from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.news_providers.gdelt import append_gdelt_rows
from src.news_providers.guardian import append_guardian_rows
from src.news_providers.newsapi import append_newsapi_rows
from src.news_providers.newsdata import append_newsdata_rows


ProviderRunner = Callable[[list[dict]], None]


def build_provider_runners(
    news_cfg: dict,
    query: str,
    max_records: int,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_retries: int,
    retry_base_sleep: float,
    sleep_seconds: float,
) -> list[tuple[str, ProviderRunner]]:
    runners: list[tuple[str, ProviderRunner]] = []

    if bool(news_cfg.get("use_gdelt", True)):
        runners.append(
            (
                "gdelt",
                lambda rows: append_gdelt_rows(
                    rows=rows,
                    query=query,
                    max_records=max_records,
                    windows=windows,
                    max_retries=max_retries,
                    retry_base_sleep=retry_base_sleep,
                    sleep_seconds=sleep_seconds,
                ),
            )
        )

    if bool(news_cfg.get("use_guardian", False)):
        runners.append(
            (
                "guardian",
                lambda rows: append_guardian_rows(
                    rows=rows,
                    news_cfg=news_cfg,
                    query=query,
                    max_records=max_records,
                    windows=windows,
                    max_retries=max_retries,
                    retry_base_sleep=retry_base_sleep,
                ),
            )
        )

    if bool(news_cfg.get("use_newsapi", False)):
        runners.append(
            (
                "newsapi",
                lambda rows: append_newsapi_rows(
                    rows=rows,
                    news_cfg=news_cfg,
                    query=query,
                    max_records=max_records,
                    max_retries=max_retries,
                    retry_base_sleep=retry_base_sleep,
                ),
            )
        )

    if bool(news_cfg.get("use_newsdata", False)):
        runners.append(
            (
                "newsdata",
                lambda rows: append_newsdata_rows(
                    rows=rows,
                    news_cfg=news_cfg,
                    query=query,
                    max_records=max_records,
                    windows=windows,
                    max_retries=max_retries,
                    retry_base_sleep=retry_base_sleep,
                ),
            )
        )

    return runners
