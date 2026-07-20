"""
Backfill step 3/6: LLM enrichment pass A (host-based BashOperator).

Schedule: None (manual trigger only)
Standalone — run after step 2 (backfill_2_company_match). Independent of
pass B (backfill_4_llm_enrich_b); the two can run in either order or skip
one entirely if only single-pass labels are needed.
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
    dag_id="backfill_3_llm_enrich_a",
    default_args=default_args,
    description="Backfill step 3/6: LLM enrichment pass A (manual, standalone)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "backfill", "news", "step3"],
) as dag:

    host_task(
        "llm_enrich_pass_a",
        "research/llm_enrich_articles.py",
        extra_env={"SLM_API_URL": SLM_API_URL, "ENRICH_PASS": "A"},
        execution_timeout=timedelta(hours=6),
    )
