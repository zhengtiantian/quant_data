"""
LLM enrichment pipeline DAG (host-based BashOperator).

Schedule: weekdays 09:00
  company_match_v2 → [llm_enrich_a, llm_enrich_b] → snorkel_merge → feature_rebuild
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG

from _host_common import host_task

SLM_API_URL = os.getenv("SLM_API_URL", "http://127.0.0.1:1234/v1")

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

with DAG(
    dag_id="quant_llm_enrichment",
    default_args=default_args,
    description="Company match → LLM sentiment (pass A+B) → label merge → feature rebuild",
    schedule="0 9 * * 1-5",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "llm", "enrichment"],
) as dag:

    company_match = host_task(
        "company_match_v2",
        "research/slm_company_match_v2.py",
        extra_env={
            "SLM_API_URL": SLM_API_URL,
            "SLM_MODELS": os.getenv("SLM_MODELS", "qwen3.5-4b,qwen3.5-4b:2"),
            "V2_WORKERS": "8",
        },
        execution_timeout=timedelta(hours=3),
    )

    llm_a = host_task(
        "llm_enrich_pass_a",
        "research/llm_enrich_articles.py",
        extra_env={
            "SLM_API_URL": SLM_API_URL,
            "ENRICH_PASS": "A",
            "SLM_MODEL_A": os.getenv("SLM_MODEL_A", "google/gemma-4-e4b"),
            "ENRICH_WORKERS": "8",
        },
        execution_timeout=timedelta(hours=3),
    )

    llm_b = host_task(
        "llm_enrich_pass_b",
        "research/llm_enrich_articles.py",
        extra_env={
            "SLM_API_URL": SLM_API_URL,
            "ENRICH_PASS": "B",
            "SLM_MODEL_B": os.getenv("SLM_MODEL_B", "qwen/qwen3.5-9b"),
            "ENRICH_WORKERS": "8",
        },
        execution_timeout=timedelta(hours=3),
    )

    snorkel = host_task(
        "snorkel_label_merge",
        "research/snorkel_label_merge.py",
        execution_timeout=timedelta(hours=1),
    )

    feature_rebuild = host_task(
        "feature_rebuild_with_llm",
        "research/daily_symbol_features.py",
        extra_env={
            "FEATURE_OUTPUT_COLLECTION": "daily_symbol_features",
            "FEATURE_LLM_COLLECTION": "news_articles_company_matched_v2",
        },
        execution_timeout=timedelta(hours=2),
    )

    company_match >> [llm_a, llm_b] >> snorkel >> feature_rebuild
