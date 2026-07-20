"""
Backfill step 5/6: Snorkel weak-supervision label merge (host-based BashOperator).

Schedule: None (manual trigger only)
Standalone — run after both LLM passes (steps 3 and 4) have produced labels.
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
    dag_id="backfill_5_snorkel_merge",
    default_args=default_args,
    description="Backfill step 5/6: Snorkel weak-supervision label merge (manual, standalone)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "backfill", "news", "step5"],
) as dag:

    host_task(
        "snorkel_merge",
        "research/snorkel_label_merge.py",
        execution_timeout=timedelta(hours=1),
    )
