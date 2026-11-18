"""
Filing changes DAG (host-based BashOperator), PROJECT_PLAN Q.17.5.

Each weekday evening: for the core names, the watchlist and the trade log, the latest 10-K / 10-Q from
EDGAR, its risk factors (Item 1A) against the same form a year earlier, and the flag Q.17.5 tested
(research/models/filing_changes.py -> /Volumes/Data4T/filing_changes/latest.json, read by quant_ai).
Only new filings are downloaded.

Schedule: 22:45 UTC = 17:45 New York (18:45 while New York is on daylight time), after the SEC's
17:30 cut-off for same-day acceptance.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from _host_common import ROOT, PYTHON, BASE_ENV, notify_failure

with DAG(
    dag_id="filing_changes",
    default_args={"owner": "quant", "retries": 1, "retry_delay": timedelta(minutes=20)},
    description="Weekdays 22:45 UTC: latest 10-K / 10-Q risk factors against a year earlier (Q.17.5 flag)",
    schedule="45 22 * * 1-5",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "sec", "filings"],
) as dag:
    BashOperator(
        task_id="risk_factor_changes",
        on_failure_callback=notify_failure,
        bash_command=f"cd {ROOT} && {PYTHON} {ROOT}/research/models/filing_changes.py",
        env=BASE_ENV,
        append_env=True,
        execution_timeout=timedelta(minutes=50),
    )
