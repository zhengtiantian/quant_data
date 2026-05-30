"""
Weekly model training DAG.

Runs every Sunday at 02:00 to retrain baseline models on the latest features.
Uses the daily_symbol_features collection (includes LLM features when available).

Tasks:
  train_baseline_models — Ridge + LightGBM walk-forward CV, saves results to MongoDB
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

QUANT_IMAGE = os.getenv("QUANT_IMAGE", "xiz001/quant_data:9df26b0")
NETWORK = "quant_docker_project-net"
QUANT_DATA_PATH = "/Users/xiz/Quant_trade/quant_data"

BASE_ENV = {
    "MONGO_URI": "mongodb://root:root@mongo6:27017/quant_data?authSource=admin",
    "LOCAL_MONGO_URI": "mongodb://root:root@mongo6:27017/quant_data?authSource=admin",
}

SOURCE_MOUNT = Mount(
    source=QUANT_DATA_PATH,
    target="/app",
    type="bind",
    read_only=False,
)

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=4),
}

with DAG(
    dag_id="quant_model_training",
    default_args=default_args,
    description="Weekly baseline model training (Ridge + LightGBM walk-forward CV)",
    schedule="0 2 * * 0",  # Sunday 02:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["quant", "model", "training"],
) as dag:

    DockerOperator(
        task_id="train_baseline_models",
        image=QUANT_IMAGE,
        command="python /app/research/train_baseline_models.py --collection daily_symbol_features",
        environment={
            **BASE_ENV,
            "FEATURE_OUTPUT_COLLECTION": "daily_symbol_features",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        },
        mounts=[SOURCE_MOUNT],
        network_mode=NETWORK,
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mount_tmp_dir=False,
    )
