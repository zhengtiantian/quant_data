"""
News validation audit DAG (host-based BashOperator).

Schedule: None (manual trigger only)
  Audits relevance and content quality of collected news articles
  (sample per symbol, ambiguous-ticker spot checks, domain distribution).
  Report is printed to the task log.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG

from _host_common import host_task

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="quant_news_validation",
    default_args=default_args,
    description="On-demand audit of news article relevance and content quality",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "news", "validation"],
) as dag:

    host_task(
        "news_validation_audit",
        "research/quality/news_validation_audit.py",
        execution_timeout=timedelta(minutes=30),
    )
