"""
Market snapshots DAG (host-based BashOperator), PROJECT_PLAN Q.16.9.

Each weekday after the close: the option market (implied volatility, skew, straddle, put / call volume
and open interest) and analysts' rating counts for the 102 research stocks and the holdings, kept daily
from 2026-11-02 because no free history exists (collectors/market/options_analyst_snapshot.py).

Schedule: 21:15 UTC = 16:15 New York (17:15 while New York is on daylight time).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from _host_common import ROOT, PYTHON, BASE_ENV, notify_failure

with DAG(
    dag_id="market_snapshots",
    default_args={"owner": "quant", "retries": 1, "retry_delay": timedelta(minutes=15)},
    description="Weekdays 21:15 UTC: option-market and analyst-rating snapshot (102 stocks + holdings)",
    schedule="15 21 * * 1-5",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "snapshot", "options", "analysts"],
) as dag:
    BashOperator(
        task_id="options_analysts",
        on_failure_callback=notify_failure,
        bash_command=f"cd {ROOT} && {PYTHON} {ROOT}/collectors/market/options_analyst_snapshot.py",
        env=BASE_ENV,
        append_env=True,
        execution_timeout=timedelta(minutes=30),
    )
