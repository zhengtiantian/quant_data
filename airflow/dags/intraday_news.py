"""
Intraday news collection DAG (host-based BashOperator).

Schedule: every 30 minutes (the four collectors run in parallel)
Replaces: ENABLE_FINNHUB_JOB / ENABLE_NEWSAPI_JOB / ENABLE_YAHOO_JOB in task.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG

from _host_common import host_task

default_args = {
    "owner": "quant",
    "retries": 0,
}

with DAG(
    dag_id="quant_intraday_news",
    default_args=default_args,
    description="Every 30 min: Finnhub + NewsAPI + Yahoo + Asia RSS news collection",
    schedule="*/30 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "news", "intraday"],
) as dag:

    finnhub = host_task(
        "finnhub_news",
        "collectors/news/finnhub/collector.py",
        execution_timeout=timedelta(minutes=10),
    )

    newsapi = host_task(
        "newsapi_news",
        "collectors/news/newsapi/collector.py",
        execution_timeout=timedelta(minutes=10),
    )

    yahoo = host_task(
        "yahoo_news",
        "collectors/news/yahoo/collector.py",
        execution_timeout=timedelta(minutes=10),
    )

    # Korean / Taiwanese / Japanese English-language outlets (2026-11-10): news on SK hynix, TSMC and
    # the like often never reaches the US-centric feeds above.
    asia_rss = host_task(
        "asia_rss_news",
        "collectors/news/asia_rss/collector.py",
        execution_timeout=timedelta(minutes=10),
    )

    [finnhub, newsapi, yahoo, asia_rss]
