"""
Historical News Article Backfill DAG (host-based BashOperator).

Schedule: None (manual trigger only)

Trigger example:
  airflow dags trigger news_history_backfill \\
    --conf '{"start_date": "2023-01-01"}'

Pipeline:
  gdelt_collect → company_match → [llm_pass_a, llm_pass_b] → snorkel_merge → feature_rebuild
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from _host_common import ROOT, PYTHON, BASE_ENV, GDELT_ENV

SLM_API_URL = os.getenv("SLM_API_URL", "http://127.0.0.1:1234/v1")

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def make_task(task_id: str, script: str, extra_env: dict | None = None,
              timeout: timedelta = timedelta(hours=4)) -> BashOperator:
    env = {**BASE_ENV, **(extra_env or {})}
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {ROOT} && {PYTHON} {ROOT}/{script}",
        env=env,
        append_env=True,
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
    params={"start_date": "2016-01-01"},
) as dag:

    gdelt_collect = BashOperator(
        task_id="gdelt_collect",
        bash_command=f"cd {ROOT} && {PYTHON} {ROOT}/news_collectors/gdelt/historical_collector.py",
        env={
            **GDELT_ENV,
            "START_DATE": "{{ dag_run.conf.get('start_date', '2016-01-01') }}",
            "USE_GKG_MONGO": "true",
        },
        append_env=True,
        execution_timeout=timedelta(hours=8),
    )

    company_match = make_task(
        "company_match",
        "research/slm_company_match_v2.py",
        extra_env={
            "SLM_API_URL": SLM_API_URL,
            "DST_COLLECTION": "news_articles_company_matched_v2",
        },
        timeout=timedelta(hours=4),
    )

    llm_a = make_task(
        "llm_enrich_pass_a",
        "research/llm_enrich_articles.py",
        extra_env={"SLM_API_URL": SLM_API_URL, "ENRICH_PASS": "A"},
        timeout=timedelta(hours=6),
    )

    llm_b = make_task(
        "llm_enrich_pass_b",
        "research/llm_enrich_articles.py",
        extra_env={"SLM_API_URL": SLM_API_URL, "ENRICH_PASS": "B"},
        timeout=timedelta(hours=6),
    )

    snorkel = make_task(
        "snorkel_merge",
        "research/snorkel_label_merge.py",
        timeout=timedelta(hours=1),
    )

    feature_rebuild = make_task(
        "feature_rebuild",
        "research/daily_symbol_features.py",
        extra_env={
            "FEATURE_OUTPUT_COLLECTION": "daily_symbol_features",
            "FEATURE_LLM_COLLECTION": "news_articles_company_matched_v2",
        },
        timeout=timedelta(hours=3),
    )

    gdelt_collect >> company_match >> [llm_a, llm_b] >> snorkel >> feature_rebuild
