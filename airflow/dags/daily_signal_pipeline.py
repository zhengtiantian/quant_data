"""
Daily signal pipeline DAG (host-based BashOperator).

Schedule: 07:30 daily
  [daily_price, premarket, analyst, macro]
      → daily_features → score_signals → [track_positions, backtest] → data_quality
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG

from _host_common import host_task, notify_failure, wait_for_night_chain
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "quant",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="daily_signal_pipeline",
    default_args=default_args,
    description="Daily, after the nightly news chain: price → premarket → analyst → macro → features → signals → positions → DQ → job_agent",
    # No clock of its own since 2026-10-24: backfill_2 triggers it when the nightly chain is done
    # (02:00 UTC backfill_1 -> backfill_2 -> here), and it starts job_agent when it is done. It
    # used to run at 07:30 UTC; nothing it collects is fresher at 07:30 than at 03:00 (US
    # pre-market opens at 08:00 UTC; the pre-market collector reads the past 7 days).
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "pipeline", "daily"],
) as dag:

    daily_price = host_task(
        "daily_price",
        "collectors/stock/price/collector.py",
        execution_timeout=timedelta(minutes=30),
    )

    premarket = host_task(
        "premarket_signals",
        "collectors/premarket/collector.py",
        extra_env={"PREMARKET_PERIOD": "5d"},
        execution_timeout=timedelta(minutes=20),
    )

    analyst = host_task(
        "analyst_consensus",
        "collectors/analyst/collector.py",
        execution_timeout=timedelta(minutes=20),
    )

    macro = host_task(
        "macro_indicators",
        "collectors/macro/collector.py",
        extra_env={"MACRO_PERIOD": "1mo"},
        execution_timeout=timedelta(minutes=20),
    )

    features = host_task(
        "daily_features",
        "research/features/daily_symbol_features.py",
        extra_env={
            "FEATURE_OUTPUT_COLLECTION": "daily_symbol_features",
            "FEATURE_LLM_COLLECTION": "news_articles_company_matched_v2",
        },
        execution_timeout=timedelta(hours=2),
    )

    signals = host_task(
        "score_signals",
        "research/signals/score_daily_signals.py",
        extra_env={"SIGNAL_TOP_N": "10"},
        execution_timeout=timedelta(minutes=15),
    )

    positions = host_task(
        "track_positions",
        "research/signals/track_positions.py",
        execution_timeout=timedelta(minutes=15),
    )

    backtest = host_task(
        "backtest_portfolio",
        "research/backtest/backtest_portfolio.py",
        execution_timeout=timedelta(minutes=30),
    )

    data_quality = host_task(
        "data_quality_check",
        "research/quality/data_quality_check.py",
        execution_timeout=timedelta(minutes=10),
    )

    wait = wait_for_night_chain()          # news features come from the nightly chain
    wait >> [daily_price, premarket, analyst, macro]
    [daily_price, premarket, analyst, macro] >> features >> signals >> [positions, backtest] >> data_quality

    # Last link: job_agent's daily scrape (its own launchd job, which keeps a 16:00 run of its own).
    # kickstart returns at once; job_agent queues for LM Studio through its own lock.
    data_quality >> BashOperator(
        task_id="start_job_agent",
        bash_command='launchctl kickstart "gui/$(id -u)/com.xiangyu.jobagent"',
        trigger_rule="all_done",
        on_failure_callback=notify_failure,
    )
