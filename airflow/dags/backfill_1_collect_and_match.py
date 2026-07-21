"""
Backfill step 1/5: GDELT collection + SLM company match (host-based, merged).

Schedule: None (manual trigger only). Two chained tasks:
    gdelt_collect >> company_match

Step 1a collects fresh GKG / news_articles from GDELT (2018-01-01 onward),
then step 1b runs SLM company-match on the newly-ingested articles
(incremental via the _id cursor — only articles past the last-processed id
are matched). Standalone — does not auto-trigger step 2.

Merged from the former backfill_1_gdelt_collect + backfill_2_company_match.
company_match runs at V2_WORKERS=32 code-level concurrency; make sure the
LM Studio server's max-concurrency is set >= 32 or the extra threads queue.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from _host_common import ROOT, PYTHON, GDELT_ENV, host_task

SLM_API_URL = os.getenv("SLM_API_URL", "http://127.0.0.1:1234/v1")

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="backfill_1_collect_and_match",
    default_args=default_args,
    description="Backfill step 1/5: GDELT collection + SLM company match (manual, standalone)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "backfill", "news", "step1"],
    params={"start_date": "2018-01-01"},
) as dag:

    gdelt_collect = BashOperator(
        task_id="gdelt_collect",
        bash_command=f"cd {ROOT} && {PYTHON} {ROOT}/collectors/news/gdelt/historical_collector.py",
        env={
            **GDELT_ENV,
            "START_DATE": "{{ dag_run.conf.get('start_date', '2018-01-01') }}",
        },
        append_env=True,
        execution_timeout=timedelta(hours=10),
    )

    company_match = host_task(
        "company_match",
        "research/labeling/slm_company_match_v2.py",
        extra_env={
            "SLM_API_URL": SLM_API_URL,
            "DST_COLLECTION": "news_articles_company_matched_v2",
            "SLM_MODELS": os.getenv("SLM_MODELS", "qwen3.5-4b,qwen3.5-4b:2"),
            "V2_WORKERS": "32",
        },
        execution_timeout=timedelta(hours=4),
    )

    gdelt_collect >> company_match
