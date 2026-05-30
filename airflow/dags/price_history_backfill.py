"""
Price history backfill DAG.

Runs daily at 06:30 to ensure 10-year price history is complete.
Replaces: scheduler/task.py ENABLE_PRICE_HISTORY_BACKFILL_JOB
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

QUANT_IMAGE = "xiz001/quant_data:9df26b0"
NETWORK = "quant_docker_project-net"
QUANT_DATA_PATH = "/Users/xiz/Quant_trade/quant_data"

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

with DAG(
    dag_id="price_history_backfill",
    default_args=default_args,
    description="Ensure 10-year daily price history is complete for all symbols",
    schedule="30 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["quant", "prices"],
) as dag:

    DockerOperator(
        task_id="price_history_backfill",
        image=QUANT_IMAGE,
        command="python /app/stock_collector/price_collector/10y_1d_history_collector.py",
        environment={
            "MONGO_URI": "mongodb://root:root@mongo6:27017/quant_data?authSource=admin",
            "LOCAL_MONGO_URI": "mongodb://root:root@mongo6:27017/quant_data?authSource=admin",
        },
        mounts=[Mount(source=QUANT_DATA_PATH, target="/app", type="bind")],
        network_mode=NETWORK,
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mount_tmp_dir=False,
        execution_timeout=timedelta(hours=2),
    )
