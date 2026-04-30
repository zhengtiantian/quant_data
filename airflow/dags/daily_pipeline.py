"""
Daily pipeline DAG.

Runs every day starting at 05:00:
  gdelt_backfill ──┐
  finnhub_news  ───┤──▶ price_update ──▶ feature_build
  newsapi_news  ───┤
  yahoo_news    ───┘

Replaces: scheduler/task.py (daily jobs)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

QUANT_IMAGE = "xiz001/quant_data:5af1779"
NETWORK = "quant_docker_project-net"
QUANT_DATA_PATH = "/Users/xiz/Quant_trade/quant_data"

BASE_ENV = {
    "MONGO_URI": "mongodb://root:root@mongo6:27017/quant_data?authSource=admin",
    "LOCAL_MONGO_URI": "mongodb://root:root@mongo6:27017/quant_data?authSource=admin",
    "MYSQL_HOST": "mysql8",
    "MYSQL_PORT": "3306",
    "MYSQL_USER": "root",
    "MYSQL_PASSWORD": "root",
    "MYSQL_DATABASE": "workflow",
    "USE_MYSQL_BATCH_QUEUE": "true",
}

SOURCE_MOUNT = Mount(
    source=QUANT_DATA_PATH,
    target="/app",
    type="bind",
    read_only=False,
)

default_args = {
    "owner": "quant",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=3),
}


def make_task(task_id: str, script: str, extra_env: dict | None = None, **kwargs):
    env = {**BASE_ENV, **(extra_env or {})}
    return DockerOperator(
        task_id=task_id,
        image=QUANT_IMAGE,
        command=f"python /app/{script}",
        environment=env,
        mounts=[SOURCE_MOUNT],
        network_mode=NETWORK,
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        **kwargs,
    )


with DAG(
    dag_id="quant_daily_pipeline",
    default_args=default_args,
    description="Daily news collection → price update → feature build",
    schedule="0 5 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["quant", "pipeline"],
) as dag:

    gdelt = make_task(
        "gdelt_backfill",
        "news_collectors/gdelt/historical_collector.py",
        extra_env={"RESET_ALL_RUNNING_ON_START": "false"},
        execution_timeout=timedelta(hours=2),
    )

    finnhub = make_task(
        "finnhub_news",
        "news_collectors/finnhub/collector.py",
        execution_timeout=timedelta(minutes=30),
    )

    newsapi = make_task(
        "newsapi_news",
        "news_collectors/newsapi/collector.py",
        execution_timeout=timedelta(minutes=30),
    )

    yahoo = make_task(
        "yahoo_news",
        "news_collectors/yahoo/collector.py",
        execution_timeout=timedelta(minutes=30),
    )

    price = make_task(
        "daily_price_update",
        "stock_collector/price_collector/collector.py",
        execution_timeout=timedelta(minutes=30),
    )

    features = make_task(
        "feature_build",
        "research/daily_symbol_features.py",
        execution_timeout=timedelta(hours=2),
    )

    [gdelt, finnhub, newsapi, yahoo] >> price >> features
