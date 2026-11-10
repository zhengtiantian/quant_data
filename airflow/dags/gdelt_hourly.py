"""
GDELT hourly increment (host-based), added 2026-11-10.

backfill_1_collect_and_match collects GDELT once a day at 04:00 UTC, so a story published in the
Asian morning (Morgan Stanley's SK hynix target cut, 2026-11-10) reached the database up to a day
later. GDELT publishes a GKG file every 15 minutes; this DAG takes the new ones every hour:
    gdelt_collect (masterfilelist by HTTP Range) >> company_match (v2, incremental _id cursor)
The batch queue's signature check reopens the growing last batch, so the hourly and the daily run
share one queue. Hours 03-06 UTC are left to the daily chain (backfill_1 -> backfill_2 ->
daily_signal_pipeline). Labelling, features and the index stay with the nightly chain.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from _host_common import ROOT, PYTHON, GDELT_ENV, host_task, lm_command, notify_failure

SLM_API_URL = os.getenv("SLM_API_URL", "http://127.0.0.1:1234/v1")

with DAG(
    dag_id="gdelt_hourly",
    default_args={"owner": "quant", "retries": 0},
    description="Hourly: new GDELT GKG files + SLM company match (daily chain keeps 03-06 UTC)",
    schedule="20 0-2,7-23 * * *",
    start_date=datetime(2026, 11, 10),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    tags=["quant", "news", "gdelt", "intraday"],
) as dag:

    gdelt_collect = BashOperator(
        task_id="gdelt_collect",
        on_failure_callback=notify_failure,
        bash_command=f"cd {ROOT} && " + lm_command(
            f"{PYTHON} {ROOT}/collectors/news/gdelt/historical_collector.py", GDELT_ENV["SLM_MODEL"],
            workers_env=("SLM_MAX_CONCURRENCY/6",), ids_env="SLM_MODELS"),
        env={**GDELT_ENV, "GDELT_MASTER_INCREMENTAL": "true", "START_DATE": "2018-01-01"},
        append_env=True,
        execution_timeout=timedelta(minutes=50),
    )

    company_match = host_task(
        "company_match",
        "research/labeling/slm_company_match_v2.py",
        extra_env={
            "SLM_API_URL": SLM_API_URL,
            "DST_COLLECTION": "news_articles_company_matched_v2",
            "SLM_MODELS": os.getenv("SLM_MODELS", "qwen3.5-4b-mlx"),
            "V2_WORKERS": "32",
        },
        execution_timeout=timedelta(minutes=40),
        lm_model=os.getenv("SLM_MODELS", "qwen3.5-4b-mlx"),
        lm_workers_env=("SLM_MAX_CONCURRENCY",),
        lm_ids_env="SLM_MODELS",
    )

    gdelt_collect >> company_match
