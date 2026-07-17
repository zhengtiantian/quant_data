"""
Weekly pipeline DAG (host-based BashOperator).

Schedule: Sunday 06:00
  inst_13f_holdings → model_training
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG

from _host_common import host_task, BASE_ENV
from airflow.operators.bash import BashOperator

ROOT = "/Users/xiz/Quant_trade/quant_data"
PYTHON = f"{ROOT}/.venv311/bin/python"

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}

with DAG(
    dag_id="quant_weekly_pipeline",
    default_args=default_args,
    description="Weekly Sunday: 13F holdings update + model retraining",
    schedule="0 6 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "weekly"],
) as dag:

    inst13f = host_task(
        "inst_13f_holdings",
        "inst_13f_collector/collector.py",
        execution_timeout=timedelta(minutes=30),
    )

    model_train = BashOperator(
        task_id="train_baseline_models",
        bash_command=f"cd {ROOT} && {PYTHON} {ROOT}/research/train_baseline_models.py --collection daily_symbol_features",
        env={
            **BASE_ENV,
            "FEATURE_OUTPUT_COLLECTION": "daily_symbol_features",
            "MLFLOW_TRACKING_URI": "http://127.0.0.1:15050",
        },
        append_env=True,
        execution_timeout=timedelta(hours=4),
    )

    inst13f >> model_train
