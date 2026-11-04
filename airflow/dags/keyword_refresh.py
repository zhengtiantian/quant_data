"""
Weekly keyword refresh for the company rules (2026-11-04, approved by the user), tools/keyword_refresh.py.

keyword_refresh_weekly   Saturday 08:00-08:30 Irish time (the universe backfill's Saturday window starts at
                         08:30 for it): the week's GKG headlines -> new product / brand / subsidiary /
                         executive names for the 102 core and 490 universe companies, checked by the local
                         model (~2 minutes for ~200 candidates on qwen3.5-9b-mlx, 2026-11-04) and by
                         co-mention. Nothing is written to the rules: the proposals wait on quant_ui's
                         Keywords tab.
keyword_refresh_apply    hourly at :55 from 08:55 to 23:55: the user's decisions from that tab into the rule
                         files (backed up first; a decision changed to reject takes the term out again).
                         No model, a few seconds.
Past news for an added keyword is searched at the start of the next universe backfill window
(backfill_universe_news.py window -> keyword_refresh.py rescan), which holds LM Studio.
Both created paused; unpaused 2026-11-04 at the user's go-ahead.
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG

from _host_common import host_task

START = pendulum.datetime(2026, 11, 1, tz="Europe/Dublin")

with DAG(
    dag_id="keyword_refresh_weekly",
    default_args={"owner": "quant", "retries": 0},
    description="Saturday 08:00 Irish time: propose new company keywords for review in quant_ui",
    schedule="0 8 * * 6",
    start_date=START,
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    tags=["quant", "news", "rules"],
) as weekly:
    host_task(
        "propose",
        "tools/keyword_refresh.py run --until 08:28",
        execution_timeout=timedelta(minutes=35),
        lm_model="qwen3.5-9b-mlx",
        lm_workers_env=("LM_WORKERS",),
    )

with DAG(
    dag_id="keyword_refresh_apply",
    default_args={"owner": "quant", "retries": 0},
    description="Hourly 08:55-23:55 Irish time: approved keywords from quant_ui into the rule files",
    schedule="55 8-23 * * *",
    start_date=START,
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    tags=["quant", "news", "rules"],
) as hourly:
    host_task("apply", "tools/keyword_refresh.py apply", execution_timeout=timedelta(minutes=5))
