"""
Historical News Article Backfill DAG.

Full pipeline:
  gdelt_collect → company_match → llm_pass_a ─┐
                                  llm_pass_b ─┴→ snorkel_merge → feature_rebuild

Schedule: None (manual trigger only)

Trigger with date range:
  airflow dags trigger news_history_backfill \\
    --conf '{"start_date": "2023-01-01"}'
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
    "MYSQL_HOST": "mysql8",
    "MYSQL_PORT": "3306",
    "MYSQL_USER": "root",
    "MYSQL_PASSWORD": "root",
    "MYSQL_DATABASE": "workflow",
    "USE_MYSQL_BATCH_QUEUE": "true",
    "GDELT_CACHE_DIR": "/Volumes/Data4T/docker-volumes/gdelt_cache",
    "GKG_TMP_DIR": "/Volumes/Data4T/gdelt_tmp",
    "RESET_ALL_RUNNING_ON_START": "false",
}

SOURCE_MOUNT = Mount(
    source=QUANT_DATA_PATH,
    target="/app",
    type="bind",
    read_only=False,
)

GDELT_MOUNT = Mount(
    source="/Volumes/Data4T/docker-volumes/gdelt_cache",
    target="/Volumes/Data4T/docker-volumes/gdelt_cache",
    type="bind",
    read_only=False,
)

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def make_task(task_id: str, script: str, extra_env: dict | None = None,
              timeout: timedelta = timedelta(hours=4)):
    return DockerOperator(
        task_id=task_id,
        image=QUANT_IMAGE,
        command=f"python /app/{script}",
        environment={**BASE_ENV, **(extra_env or {})},
        mounts=[SOURCE_MOUNT, GDELT_MOUNT],
        network_mode=NETWORK,
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        execution_timeout=timeout,
    )


with DAG(
    dag_id="news_history_backfill",
    default_args=default_args,
    description="On-demand: GDELT collect → company match → LLM label → feature rebuild",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["quant", "backfill", "news"],
    params={
        "start_date": "2016-01-01",
    },
) as dag:

    gdelt_collect = make_task(
        task_id="gdelt_collect",
        script="news_collectors/gdelt/historical_collector.py",
        extra_env={
            "START_DATE": "{{ dag_run.conf.get('start_date', '2016-01-01') }}",
            "RESET_ALL_RUNNING_ON_START": "false",
            "USE_GKG_MONGO": "true",
        },
        timeout=timedelta(hours=8),
    )

    company_match = make_task(
        task_id="company_match",
        script="research/slm_company_match_v2.py",
        extra_env={
            "DST_COLLECTION": "news_articles_company_matched_v2",
        },
        timeout=timedelta(hours=4),
    )

    llm_pass_a = make_task(
        task_id="llm_enrich_pass_a",
        script="research/llm_enrich_articles.py",
        extra_env={
            "LLM_PASS": "a",
            "OLLAMA_MODEL": "gemma3:4b-q4_K_M",
            "OLLAMA_API": "http://host.docker.internal:11434",
        },
        timeout=timedelta(hours=6),
    )

    llm_pass_b = make_task(
        task_id="llm_enrich_pass_b",
        script="research/llm_enrich_articles.py",
        extra_env={
            "LLM_PASS": "b",
            "OLLAMA_MODEL": "qwen3:4b-q4_K_M",
            "OLLAMA_API": "http://host.docker.internal:11434",
        },
        timeout=timedelta(hours=6),
    )

    snorkel = make_task(
        task_id="snorkel_merge",
        script="research/snorkel_label_merge.py",
        timeout=timedelta(hours=1),
    )

    feature_rebuild = make_task(
        task_id="feature_rebuild",
        script="research/daily_symbol_features.py",
        extra_env={
            "FEATURE_OUTPUT_COLLECTION": "daily_symbol_features_company_matched_v2",
            "FEATURE_LLM_COLLECTION": "news_articles_company_matched_v2",
        },
        timeout=timedelta(hours=3),
    )

    gdelt_collect >> company_match >> [llm_pass_a, llm_pass_b] >> snorkel >> feature_rebuild
