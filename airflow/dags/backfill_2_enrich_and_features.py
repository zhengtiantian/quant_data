"""
Backfill step 2/2: LLM enrichment (A + B) -> merge -> feature rebuild (host-based, merged).

Schedule: None (manual trigger only). Chained tasks:
    llm_enrich_pass_a >> llm_enrich_pass_b >> merge_ab_labels >> feature_rebuild

Merged from the former backfill_2_llm_enrich_a + backfill_3_llm_enrich_b +
backfill_4_snorkel_merge + backfill_5_feature_rebuild. The snorkel step was
dropped: with only two label sources a probabilistic label model adds nothing,
so merge_ab_labels.py combines pass A/B directly into llm_sentiment_final +
llm_disagreement (both read by feature rebuild).

All steps are incremental:
  - enrich A/B skip docs that already have their target field
  - merge only fills docs missing final/disagreement
  - feature rebuild recomputes as needed

Models (deploy both in LM Studio):
  - Pass A -> gemma-4-e4b   (SLM_MODEL_A)
  - Pass B -> qwen3.5-9b    (SLM_MODEL_B)
Passes run sequentially (A then B), so LM Studio only needs one model loaded at
a time. If it has memory for both at once, A and B can be parallelised — change
the tail to: [enrich_a, enrich_b] >> merge >> feature.
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
    dag_id="backfill_2_enrich_and_features",
    default_args=default_args,
    description="Backfill step 2/2: LLM enrich A/B + merge + feature rebuild (manual, standalone)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "backfill", "news", "step2"],
) as dag:

    enrich_a = host_task(
        "llm_enrich_pass_a",
        "research/labeling/llm_enrich_articles.py",
        # model ids must match what LM Studio actually serves (see /v1/models)
        extra_env={
            "SLM_API_URL": SLM_API_URL,
            "ENRICH_PASS": "A",
            "SLM_MODEL_A": os.getenv("SLM_MODEL_A", "gemma-4-e4b-it-mlx"),
        },
        execution_timeout=timedelta(hours=6),
    )

    enrich_b = host_task(
        "llm_enrich_pass_b",
        "research/labeling/llm_enrich_articles.py",
        extra_env={
            "SLM_API_URL": SLM_API_URL,
            "ENRICH_PASS": "B",
            "SLM_MODEL_B": os.getenv("SLM_MODEL_B", "qwen3.5-9b-mlx"),
        },
        execution_timeout=timedelta(hours=6),
    )

    merge_ab = host_task(
        "merge_ab_labels",
        "research/labeling/merge_ab_labels.py",
        execution_timeout=timedelta(minutes=30),
    )

    feature_rebuild = host_task(
        "feature_rebuild",
        "research/features/daily_symbol_features.py",
        extra_env={
            "FEATURE_OUTPUT_COLLECTION": "daily_symbol_features",
            "FEATURE_LLM_COLLECTION": "news_articles_company_matched_v2",
        },
        execution_timeout=timedelta(hours=3),
    )

    enrich_a >> enrich_b >> merge_ab >> feature_rebuild
