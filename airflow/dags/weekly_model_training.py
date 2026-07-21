"""
Weekly model training DAG (host-based BashOperator).

Schedule: Sunday 07:00 — Ridge + LightGBM walk-forward CV on
daily_symbol_features, results saved to MongoDB + MLflow.

Runs 1 hour after weekly_inst13f_holdings (06:00, ~30min timeout) so this
week's 13F refresh is already in the feature set being trained on. Standalone
DAG, not chained — the two are time-sequenced via schedule, not a task
dependency, so a slow/failed 13F run does not block training from firing.
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
    dag_id="weekly_model_training",
    default_args=default_args,
    description="Weekly Sunday 07:00: baseline model training (Ridge + LightGBM walk-forward CV)",
    schedule="0 7 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "weekly", "model", "training"],
) as dag:

    BashOperator(
        task_id="train_baseline_models",
        bash_command=(
            f"cd {ROOT} && {PYTHON} {ROOT}/research/models/train_baseline_models.py"
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
