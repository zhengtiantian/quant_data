"""
Retail sentiment DAG (host-based BashOperator).

Schedule: 20:30 daily
Replaces: ENABLE_RETAIL_JOB in task.py
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
    dag_id="quant_retail_sentiment",
    default_args=default_args,
    description="Daily 20:30: retail investor sentiment collection",
    schedule="30 20 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "retail", "sentiment"],
) as dag:

    host_task(
        "retail_sentiment",
        "retail_collector/collector.py",
        execution_timeout=timedelta(minutes=15),
    )
