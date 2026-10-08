"""
Backfill step 1/2: GDELT collection + SLM company match (host-based, merged).

Schedule: daily 02:00 UTC (03:00 Irish summer time), since 2026-10-08. Before that it was
manual-only and silently fell 2.5 months behind (last run 2026-07-21). Tasks:
    gdelt_collect >> [company_match >> trigger_enrich, gkg_export]
trigger_enrich starts backfill_2_enrich_and_features, so one daily run goes
collect -> match -> enrich -> merge -> features -> Qdrant index, finishing well before
daily_signal_pipeline at 07:30 UTC. A day's increment is ~96 GKG files.

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
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from _host_common import ROOT, PYTHON, BASE_ENV, GDELT_ENV, host_task

SLM_API_URL = os.getenv("SLM_API_URL", "http://127.0.0.1:1234/v1")

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="backfill_1_collect_and_match",
    default_args=default_args,
    description="Daily step 1/2: GDELT collection + SLM company match + Parquet export, then triggers step 2",
    schedule="0 2 * * *",
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
        # A 2.5-month gap is ~7,500 GKG files; importing them into gkg_index (325GB $text
        # index) runs well past 10 hours. The collector resumes, but a timeout mid-run
        # only wastes the retry.
        execution_timeout=timedelta(hours=48),
    )

    company_match = host_task(
        "company_match",
        "research/labeling/slm_company_match_v2.py",
        extra_env={
            "SLM_API_URL": SLM_API_URL,
            "DST_COLLECTION": "news_articles_company_matched_v2",
            "SLM_MODELS": os.getenv("SLM_MODELS", "qwen3.5-4b-mlx"),  # one loaded instance
            "V2_WORKERS": "32",
        },
        execution_timeout=timedelta(hours=4),
    )

    # Refresh the Parquet copy of gkg_index (Stage 6B) for every month touched by this run.
    # Re-exports whole months from MongoDB, starting at the last month already exported.
    gkg_export = BashOperator(
        task_id="gkg_export",
        bash_command=f"cd {ROOT} && {PYTHON} {ROOT}/tools/gkg_parquet.py export --since-last",
        env={**BASE_ENV, "GKG_PARQUET_DIR": os.getenv("GKG_PARQUET_DIR", "/Volumes/Data4T/gkg_parquet")},
        append_env=True,
        execution_timeout=timedelta(hours=2),
    )

    trigger_enrich = TriggerDagRunOperator(
        task_id="trigger_enrich",
        trigger_dag_id="backfill_2_enrich_and_features",
        wait_for_completion=False,
    )

    gdelt_collect >> [company_match, gkg_export]
    company_match >> trigger_enrich
