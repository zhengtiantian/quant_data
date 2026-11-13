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

from _host_common import BASE_ENV, ROOT, host_task, notify_failure, student_task
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

QUANT_AI = os.path.join(os.path.dirname(ROOT), "quant_ai")

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

    # 2026-11-10: both passes are labelled by the distilled students (student_enrich_articles.py):
    # A = Gemma-4-E2B taught by gemma-4-e4b, B = Qwen3-1.7B taught by qwen3.5-9b, ~20 articles/s each.
    # The teacher script llm_enrich_articles.py stays for relabelling and for training new students.
    enrich_a = student_task(
        "llm_enrich_pass_a",
        "research/labeling/student_enrich_articles.py",
        name="label-a-student",
        extra_env={"ENRICH_PASS": "A"},
        execution_timeout=timedelta(hours=6),
    )

    enrich_b = student_task(
        "llm_enrich_pass_b",
        "research/labeling/student_enrich_articles.py",
        name="label-b-student",
        extra_env={"ENRICH_PASS": "B"},
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

    # Keep the dense leg of hybrid search current (PROJECT_PLAN R.12): embed the newly matched
    # articles into Qdrant news_v1. index_news.py resumes from its _id checkpoint, so a daily run
    # only embeds that day's articles. Runs after enrichment because the payload carries labels.
    index_news = BashOperator(
        task_id="index_news_qdrant",
        on_failure_callback=notify_failure,
        bash_command=f"cd {QUANT_AI} && {QUANT_AI}/.venv/bin/python tools/index_news.py",
        env={
            **BASE_ENV,
            "LM_STUDIO_URL": SLM_API_URL,
            "EMBED_MODEL": os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"),
            "QDRANT_URL": os.getenv("QDRANT_URL", "http://localhost:26333"),
        },
        append_env=True,
        execution_timeout=timedelta(hours=2),
    )

    # Next link of the daily chain (see backfill_1): signals start as soon as this run is done.
    trigger_signals = TriggerDagRunOperator(
        task_id="trigger_signals",
        trigger_dag_id="daily_signal_pipeline",
        wait_for_completion=False,
        trigger_rule="all_done",
    )

    # 2026-11-10, the user: labelling paused while the distilled students replace the teachers. While the
    # flag file exists both passes exit at once (before lm_session loads a model) and the rest of the chain
    # runs as usual; the articles they skip are picked up by the first run after the file is removed.
    pause_file = os.getenv("LABELING_PAUSE_FILE", os.path.expanduser("~/Library/Application Support/quant/labeling.paused"))
    for t in (enrich_a, enrich_b):
        t.bash_command = (f'if [ -e "{pause_file}" ]; then echo "labelling paused: {pause_file}"; exit 0; fi; '
                          + t.bash_command)

    enrich_a >> enrich_b >> merge_ab >> feature_rebuild >> index_news >> trigger_signals

    # 2026-11-13 (N.9): shadow run of the GD1 students on the last 7 days' new articles. Writes only
    # news_articles.student_gd1, which nothing reads yet; a side branch, so a failure never holds up
    # the chain, and it queues for the GPU lock like the other student tasks (~2 min a night).
    gd1_shadow = student_task(
        "gd1_shadow",
        "research/labeling/student_gd1_shadow.py",
        name="gd1-shadow",
        execution_timeout=timedelta(hours=1),
    )
    gd1_shadow.retries = 0
    gd1_shadow.trigger_rule = "all_done"
    enrich_b >> gd1_shadow
