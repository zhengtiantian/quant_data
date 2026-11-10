"""
GDELT hourly increment (host-based), added 2026-11-10.

backfill_1_collect_and_match collects GDELT once a day at 04:00 UTC, so a story published in the
Asian morning (Morgan Stanley's SK hynix target cut, 2026-11-10) reached the database up to a day
later. GDELT publishes a GKG file every 15 minutes; this DAG takes the new ones every hour:
    gdelt_collect (masterfilelist by HTTP Range) >> company_match (v2 relevance student, _id cursor)
The batch queue's signature check reopens the growing last batch, so the hourly and the daily run
share one queue. Hours 03-06 UTC are left to the daily chain (backfill_1 -> backfill_2 ->
daily_signal_pipeline). Labelling, features and the index stay with the nightly chain.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from _host_common import ROOT, PYTHON, GDELT_ENV, lm_command, notify_failure, student_task

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

    company_match = student_task(
        "company_match",
        "research/labeling/slm_company_match_v2.py",
        name="relevance-student",
        extra_env={"DST_COLLECTION": "news_articles_company_matched_v2", "V2_ENGINE": "student"},
        execution_timeout=timedelta(minutes=40),
    )

    gdelt_collect >> company_match
