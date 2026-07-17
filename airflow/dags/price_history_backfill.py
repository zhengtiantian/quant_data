"""
Price history backfill DAG (host-based BashOperator).

Schedule: 06:30 daily — ensures 10-year daily price history is complete.
Replaces: ENABLE_PRICE_HISTORY_BACKFILL_JOB in task.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG

from _host_common import host_task

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

with DAG(
    dag_id="price_history_backfill",
    default_args=default_args,
    description="Daily 06:30: ensure 10-year price history complete for all symbols",
    schedule="30 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "prices", "backfill"],
) as dag:

    host_task(
        "price_history_backfill",
        "stock_collector/price_collector/10y_1d_history_collector.py",
        execution_timeout=timedelta(hours=2),
    )
