"""
Weekly 13F institutional holdings DAG (host-based BashOperator).

Schedule: Sunday 06:00 — fetches the two most recent 13F-HR filings via
edgartools, derives inst_holding_pct / inst_holding_pct_chg (strongest
single factor, 60d IC +0.20).

Standalone — does not chain into model training. weekly_model_training
runs separately at 07:00, after this has had time to finish.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG

from _host_common import host_task

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}

with DAG(
    dag_id="weekly_inst13f_holdings",
    default_args=default_args,
    description="Weekly Sunday 06:00: SEC EDGAR 13F institutional holdings update",
    schedule="0 6 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "weekly", "13f"],
) as dag:

    host_task(
        "inst_13f_holdings",
        "collectors/inst_13f/collector.py",
        execution_timeout=timedelta(minutes=30),
    )
