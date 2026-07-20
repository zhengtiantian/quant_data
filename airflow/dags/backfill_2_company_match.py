"""
Backfill step 2/6: SLM company-match verification (host-based BashOperator).

Schedule: None (manual trigger only)
Standalone — run after step 1 (backfill_1_gdelt_collect) has produced fresh
news_articles. There is no automatic daily GDELT collection; step 1 must be
triggered manually first. Does not auto-trigger step 3.
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
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="backfill_2_company_match",
    default_args=default_args,
    description="Backfill step 2/6: SLM company-match verification (manual, standalone)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "backfill", "news", "step2"],
) as dag:

    host_task(
        "company_match",
        "research/slm_company_match_v2.py",
        extra_env={
            "SLM_API_URL": SLM_API_URL,
            "DST_COLLECTION": "news_articles_company_matched_v2",
            "SLM_MODELS": os.getenv("SLM_MODELS", "qwen3.5-4b,qwen3.5-4b:2"),
            "V2_WORKERS": "8",
        },
        execution_timeout=timedelta(hours=4),
    )
