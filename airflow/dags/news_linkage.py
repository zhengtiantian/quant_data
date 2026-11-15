"""
News linkage (PROJECT_PLAN N.6 step 5, 2026-11-15): every 30 minutes.

research/news_linkage/live.py groups the followed symbols' news into storylines, decides which fired
(a burst against the storyline's own baseline, a new storyline on several sites, a leadership event) and
has Claude judge the combined impact of what fired; quant_ai shows the latest judgment in the holdings
page's Action cell, as a hint that never changes the advice.

At :15 and :45: the intraday collectors run at :00 and :30, and gdelt_hourly's relevance match (which files
every new article into news_articles_company_matched_v2) starts at :20. Claude is called through the CLI on
the user's subscription, never the API: ANTHROPIC_API_KEY is removed from the task's environment as well as
from the CLI's. Embeddings come from LM Studio's embedding model, which lm_session keeps loaded for search,
so the task takes no LM Studio lock. It runs under quant_ai's interpreter: the judge reuses quant_ai's
retrieval (news_rag) and injection guard.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from _host_common import BASE_ENV, ROOT, notify_failure

QUANT_AI_PYTHON = "/Users/xiz/Quant_trade/quant_ai/.venv/bin/python"

with DAG(
    dag_id="news_linkage",
    default_args={"owner": "quant", "retries": 0},
    description="Every 30 min: storylines -> triggers -> Claude (CLI) judgment, shown as a hint on holdings",
    schedule="15,45 * * * *",
    start_date=datetime(2026, 11, 15),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    tags=["quant", "news", "intraday", "n6"],
) as dag:

    BashOperator(
        task_id="judge_storylines",
        bash_command=f"cd {ROOT}/research/news_linkage && env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN "
                     f"{QUANT_AI_PYTHON} live.py",
        env=BASE_ENV,
        append_env=True,
        execution_timeout=timedelta(minutes=28),
        on_failure_callback=notify_failure,
    )
