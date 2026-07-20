"""
Daily signal pipeline DAG (host-based BashOperator).

Schedule: 07:30 daily
  [daily_price, premarket, analyst, macro]
      → daily_features → score_signals → [track_positions, backtest] → data_quality
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
    dag_id="daily_signal_pipeline",
    default_args=default_args,
    description="Daily 07:30: price → premarket → analyst → macro → features → signals → positions → DQ",
    schedule="30 7 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "pipeline", "daily"],
) as dag:

    daily_price = host_task(
        "daily_price",
        "stock_collector/price_collector/collector.py",
        execution_timeout=timedelta(minutes=30),
    )

    premarket = host_task(
        "premarket_signals",
        "premarket_collector/collector.py",
        extra_env={"PREMARKET_PERIOD": "5d"},
        execution_timeout=timedelta(minutes=20),
    )

    analyst = host_task(
        "analyst_consensus",
        "analyst_collector/collector.py",
        execution_timeout=timedelta(minutes=20),
    )

    macro = host_task(
        "macro_indicators",
        "macro_collector/collector.py",
        extra_env={"MACRO_PERIOD": "1mo"},
        execution_timeout=timedelta(minutes=20),
    )

    features = host_task(
        "daily_features",
        "research/daily_symbol_features.py",
        extra_env={
            "FEATURE_OUTPUT_COLLECTION": "daily_symbol_features",
            "FEATURE_LLM_COLLECTION": "news_articles_company_matched_v2",
        },
        execution_timeout=timedelta(hours=2),
    )

    signals = host_task(
        "score_signals",
        "research/score_daily_signals.py",
        extra_env={"SIGNAL_TOP_N": "10"},
        execution_timeout=timedelta(minutes=15),
    )

    positions = host_task(
        "track_positions",
        "research/track_positions.py",
        execution_timeout=timedelta(minutes=15),
    )

    backtest = host_task(
        "backtest_portfolio",
        "research/backtest_portfolio.py",
        execution_timeout=timedelta(minutes=30),
    )

    data_quality = host_task(
        "data_quality_check",
        "research/data_quality_check.py",
        execution_timeout=timedelta(minutes=10),
    )

    [daily_price, premarket, analyst, macro] >> features >> signals >> [positions, backtest] >> data_quality
