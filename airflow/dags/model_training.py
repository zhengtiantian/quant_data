"""
Weekly model training DAG (host-based BashOperator).

Schedule: Sunday 02:00 — Ridge + LightGBM walk-forward CV, results saved to MongoDB + MLflow.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from _host_common import ROOT, PYTHON, BASE_ENV

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}

with DAG(
    dag_id="quant_model_training",
    default_args=default_args,
    description="Weekly baseline model training (Ridge + LightGBM walk-forward CV)",
    schedule="0 2 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "model", "training"],
) as dag:

    BashOperator(
        task_id="train_baseline_models",
        bash_command=(
            f"cd {ROOT} && {PYTHON} {ROOT}/research/train_baseline_models.py"
            " --collection daily_symbol_features"
        ),
        env={
            **BASE_ENV,
            "FEATURE_OUTPUT_COLLECTION": "daily_symbol_features",
            "MLFLOW_TRACKING_URI": "http://127.0.0.1:15050",
        },
        append_env=True,
        execution_timeout=timedelta(hours=4),
    )
