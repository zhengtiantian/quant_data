"""
GDELT daily backfill DAG (host-based BashOperator).

Schedule: 05:15 daily — processes all pending batches and exits when done.
Replaces: ENABLE_DAILY_GDELT_BACKFILL_JOB in task.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG

from _host_common import ROOT, PYTHON, GDELT_ENV
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}

with DAG(
    dag_id="quant_gdelt_backfill",
    default_args=default_args,
    description="Daily GDELT GKG backfill: seed pending batches → process → exit",
    schedule="15 5 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "gdelt", "backfill"],
) as dag:

    BashOperator(
        task_id="gdelt_backfill",
        bash_command=f"cd {ROOT} && {PYTHON} {ROOT}/news_collectors/gdelt/historical_collector.py",
        env=GDELT_ENV,
        append_env=True,
        execution_timeout=timedelta(hours=10),
    )
