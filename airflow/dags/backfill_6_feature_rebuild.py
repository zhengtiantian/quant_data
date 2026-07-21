"""
Backfill step 6/6: daily symbol feature rebuild with LLM labels (host-based BashOperator).

Schedule: None (manual trigger only)
Standalone — run after step 5 (backfill_5_snorkel_merge). After this,
optionally trigger the separate quant_news_validation DAG to audit quality.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG

from _host_common import host_task

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="backfill_6_feature_rebuild",
    default_args=default_args,
    description="Backfill step 6/6: daily symbol feature rebuild with LLM labels (manual, standalone)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "backfill", "news", "step6"],
) as dag:

    host_task(
        "feature_rebuild",
        "research/features/daily_symbol_features.py",
        extra_env={
            "FEATURE_OUTPUT_COLLECTION": "daily_symbol_features",
            "FEATURE_LLM_COLLECTION": "news_articles_company_matched_v2",
        },
        execution_timeout=timedelta(hours=3),
    )
