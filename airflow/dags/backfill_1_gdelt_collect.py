"""
Backfill step 1/6: GDELT historical GKG collection (host-based BashOperator).

Schedule: None (manual trigger only) — no automatic daily collection.
Trigger this on demand whenever you want to pull new/missing GDELT GKG
files. Standalone — does not auto-trigger step 2. Run this, check the
log/Mongo counts, then manually trigger backfill_2_company_match when ready.

Note: the START_DATE param below is accepted but not currently honored by
historical_collector.py (get_gkg_file_urls() always scans 2016-01-01 →
today; already-completed batches are skipped via the MySQL queue's
INSERT IGNORE regardless of the requested start date).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from _host_common import ROOT, PYTHON, GDELT_ENV

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="backfill_1_gdelt_collect",
    default_args=default_args,
    description="Backfill step 1/6: GDELT historical GKG collection (manual, standalone)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "backfill", "news", "step1"],
    params={"start_date": "2016-01-01"},
) as dag:

    BashOperator(
        task_id="gdelt_collect",
        bash_command=f"cd {ROOT} && {PYTHON} {ROOT}/news_collectors/gdelt/historical_collector.py",
        env={
            **GDELT_ENV,
            "START_DATE": "{{ dag_run.conf.get('start_date', '2016-01-01') }}",
        },
        append_env=True,
        execution_timeout=timedelta(hours=10),
    )
