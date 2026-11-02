"""
Daily advice DAG (host-based BashOperator), PROJECT_PLAN Q.16.

Three times each weekday the advice models (research/models/daily_advice.py, retrained on Sundays by
weekly_model_training) forecast each holding's next 5 / 20 / 60 sessions from fresh daily bars and
the news store, into /Volumes/Data4T/daily_advice/latest.json, which quant_ai's daily action shows.

Schedule (UTC): 13:30 = 08:30 New York before the open, 17:30 = 12:30 midday, 21:30 = 16:30 after
the close (an hour later while New York is on daylight time).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from _host_common import ROOT, PYTHON, BASE_ENV, notify_failure

with DAG(
    dag_id="daily_advice",
    default_args={"owner": "quant", "retries": 1, "retry_delay": timedelta(minutes=10)},
    description="Weekdays 13:30 / 17:30 / 21:30 UTC: model advice per holding for the daily action",
    schedule="30 13,17,21 * * 1-5",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "advice", "model"],
) as dag:
    BashOperator(
        task_id="predict",
        on_failure_callback=notify_failure,
        bash_command=f"cd {ROOT} && {PYTHON} {ROOT}/research/models/daily_advice.py predict $ADVICE_SYMBOLS",
        env={**BASE_ENV, "ADVICE_SYMBOLS": "INTC SKHY TSM STX GOOGL"},
        append_env=True,
        execution_timeout=timedelta(minutes=20),
    )
