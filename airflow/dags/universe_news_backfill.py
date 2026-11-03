"""
Universe news backfill in the hours the user gave (2026-11-03), PROJECT_PLAN Q.16.

tools/backfill_universe_news.py window: scan, filter and the headline labels for the ~490 universe
companies since 2021, resumable, stopping before the window ends. Two windows a day, Irish local
time (the DAG's timezone, so they do not move an hour with daylight saving):
  11:30-15:45   after job_agent's 11:00 run, ending before its 16:00 one
  16:30-23:00   after job_agent's 16:00 run, before the 04:00 nightly chain
Both go through tools/lm_session.py, so a window that opens while job_agent still holds LM Studio
waits for it. ~1.43M distinct headlines at ~1.37/s on qwen3.5-9b-mlx, 10.75 hours a day: ~53k a day, about
27 days, newest month first (2024-2026 in about 11 days).
Created paused: unpaused by hand once the user has approved the start.
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG

from _host_common import host_task

WINDOWS = {"1130": ("30 11 * * *", "11:30", "15:45", timedelta(hours=4, minutes=55)),
           "1630": ("30 16 * * *", "16:30", "23:00", timedelta(hours=7, minutes=10))}

for name, (cron, start, until, timeout) in WINDOWS.items():
    with DAG(
        dag_id=f"universe_news_backfill_{name}",
        default_args={"owner": "quant", "retries": 0},
        description=f"Daily {start}-{until} Irish time: universe news scan / filter / headline labels",
        schedule=cron,
        start_date=pendulum.datetime(2026, 11, 1, tz="Europe/Dublin"),
        catchup=False,
        max_active_runs=1,
        is_paused_upon_creation=True,
        tags=["quant", "news", "universe", "backfill"],
    ) as dag:
        host_task(
            "window",
            f"tools/backfill_universe_news.py window --until {until}",
            execution_timeout=timeout,
            lm_model="qwen3.5-9b-mlx",
            lm_workers_env=("LM_WORKERS",),
        )
    globals()[dag.dag_id] = dag
