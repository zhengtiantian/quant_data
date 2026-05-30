"""
LLM enrichment pipeline DAG.

Runs daily at 09:00 (after news collection completes):
  company_match_v2 ──▶ llm_enrich_a ──┐
                    └─▶ llm_enrich_b ──┴──▶ snorkel_merge ──▶ feature_rebuild

Requires LM Studio running on Windows at SLM_API_URL (192.168.31.226:1234).
All steps are resumable: already-processed docs are skipped automatically.
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
SLM_API_URL = os.getenv("SLM_API_URL", "http://192.168.31.226:1234/v1")

BASE_ENV = {
    "MONGO_URI": "mongodb://root:root@mongo6:27017/quant_data?authSource=admin",
    "LOCAL_MONGO_URI": "mongodb://root:root@mongo6:27017/quant_data?authSource=admin",
    "SLM_API_URL": SLM_API_URL,
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
    "retry_delay": timedelta(minutes=15),
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
    dag_id="quant_llm_enrichment",
    default_args=default_args,
    description="Company match v2 → LLM sentiment passes → label merge → feature rebuild",
    schedule="0 9 * * 1-5",  # weekdays only; LM Studio may be offline on weekends
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["quant", "llm", "enrichment"],
) as dag:

    company_match = make_task(
        "company_match_v2",
        "research/slm_company_match_v2.py",
        extra_env={
            "SLM_MODELS": os.getenv("SLM_MODELS", "qwen3.5-4b,qwen3.5-4b:2"),
            "V2_WORKERS": "8",
        },
        execution_timeout=timedelta(hours=3),
    )

    llm_enrich_a = make_task(
        "llm_enrich_pass_a",
        "research/llm_enrich_articles.py",
        extra_env={
            "ENRICH_PASS": "A",
            "SLM_MODEL_A": os.getenv("SLM_MODEL_A", "google/gemma-4-e4b"),
            "SLM_MODEL_A2": os.getenv("SLM_MODEL_A2", "google/gemma-4-e4b:2"),
            "ENRICH_WORKERS": "8",
        },
        execution_timeout=timedelta(hours=3),
    )

    llm_enrich_b = make_task(
        "llm_enrich_pass_b",
        "research/llm_enrich_articles.py",
        extra_env={
            "ENRICH_PASS": "B",
            "SLM_MODEL_B": os.getenv("SLM_MODEL_B", "qwen/qwen3.5-9b"),
            "ENRICH_WORKERS": "8",
        },
        execution_timeout=timedelta(hours=3),
    )

    snorkel_merge = make_task(
        "snorkel_label_merge",
        "research/snorkel_label_merge.py",
        execution_timeout=timedelta(hours=1),
    )

    feature_rebuild = make_task(
        "feature_rebuild_with_llm",
        "research/daily_symbol_features.py",
        extra_env={
            "FEATURE_OUTPUT_COLLECTION": "daily_symbol_features",
            "FEATURE_LLM_COLLECTION": "news_articles_company_matched_v2",
        },
        execution_timeout=timedelta(hours=2),
    )

    company_match >> [llm_enrich_a, llm_enrich_b] >> snorkel_merge >> feature_rebuild
