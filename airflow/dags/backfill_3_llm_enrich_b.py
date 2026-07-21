"""
Backfill step 3/5: LLM enrichment pass B (host-based BashOperator).

Schedule: None (manual trigger only)
Standalone — independent of pass A (backfill_2_llm_enrich_a); run after
step 1 (backfill_1_collect_and_match). snorkel_merge (step 4) expects both
passes present, but each can be triggered and inspected on its own first.
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
    dag_id="backfill_3_llm_enrich_b",
    default_args=default_args,
    description="Backfill step 3/5: LLM enrichment pass B (manual, standalone)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "backfill", "news", "step3"],
) as dag:

    host_task(
        "llm_enrich_pass_b",
        "research/labeling/llm_enrich_articles.py",
        extra_env={"SLM_API_URL": SLM_API_URL, "ENRICH_PASS": "B"},
        execution_timeout=timedelta(hours=6),
    )
