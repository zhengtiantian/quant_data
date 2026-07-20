"""
GDELT batch self-healing verification DAG (host-based BashOperator).

Schedule: weekly, Sunday 04:00 — before quant_weekly_pipeline (06:00).

Why this exists: claim_next_batch() only reclaims 'pending'/'failed'/stuck-'running'
batches — a batch that finished with a handful of individual file misses (transient
404/timeout during its one download attempt) is marked 'done' and never revisited,
so gaps inside already-completed batches accumulate silently forever. This DAG
re-derives each 'done' batch's expected file timestamps and checks them against
gkg_index, reopening (status='pending') any batch found incomplete.

Note: GDELT collection is manual-only (backfill_1_gdelt_collect, no daily auto
run) — batches reopened here just sit as 'pending' until you trigger it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG

from _host_common import GDELT_ENV, host_task

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

with DAG(
    dag_id="gdelt_batch_verify",
    default_args=default_args,
    description="Weekly: scan 'done' GDELT batches for internal gaps and reopen incomplete ones",
    schedule="0 4 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "gdelt", "verify"],
) as dag:

    host_task(
        "verify_batches",
        "news_collectors/gdelt/historical_collector.py",
        extra_env={**GDELT_ENV, "VERIFY_MODE": "true"},
        execution_timeout=timedelta(hours=1),
    )
