#!/usr/bin/env python3
"""Scheduler for daily news collection, optional backfill, and feature builds."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import schedule
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)

PYTHON_BIN = sys.executable
LOG_FILE = ROOT / "pipeline_scheduler.log"
LOCAL_MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)

JOB_LOCKS: dict[str, threading.Lock] = {}
MANAGED_PROCESSES: dict[str, subprocess.Popen] = {}
STREAM_THREADS: dict[str, threading.Thread] = {}


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() == "true"


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def get_lock(job_name: str) -> threading.Lock:
    if job_name not in JOB_LOCKS:
        JOB_LOCKS[job_name] = threading.Lock()
    return JOB_LOCKS[job_name]


def effective_mongo_uri() -> str | None:
    uri = os.getenv("MONGO_URI")
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.hostname not in {"mongo6", "mongo", "mongodb"}:
        return uri
    fallback = urlparse(LOCAL_MONGO_URI)
    return urlunparse(
        (
            parsed.scheme or fallback.scheme,
            fallback.netloc,
            parsed.path or fallback.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def run_script_once(job_name: str, relative_path: str, timeout: int | None = None, extra_env: dict | None = None) -> None:
    lock = get_lock(job_name)
    if not lock.acquire(blocking=False):
        logging.info("%s skipped: previous run still in progress", job_name)
        return

    script_path = ROOT / relative_path
    env = os.environ.copy()
    mongo_uri = effective_mongo_uri()
    if mongo_uri:
        env["MONGO_URI"] = mongo_uri
        env.setdefault("LOCAL_MONGO_URI", LOCAL_MONGO_URI)
    if extra_env:
        env.update(extra_env)

    try:
        logging.info("Start %s -> %s", job_name, relative_path)
        result = subprocess.run(
            [PYTHON_BIN, str(script_path)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.stdout.strip():
            logging.info("%s stdout:\n%s", job_name, result.stdout.strip())
        if result.stderr.strip():
            logging.warning("%s stderr:\n%s", job_name, result.stderr.strip())
        if result.returncode != 0:
            logging.error("%s failed with exit code %s", job_name, result.returncode)
        else:
            logging.info("%s finished successfully", job_name)
    except subprocess.TimeoutExpired:
        logging.error("%s timed out after %ss", job_name, timeout)
    except Exception:
        logging.exception("%s crashed", job_name)
    finally:
        lock.release()


def ensure_long_running(job_name: str, relative_path: str, extra_env: dict | None = None) -> None:
    proc = MANAGED_PROCESSES.get(job_name)
    if proc and proc.poll() is None:
        logging.info("%s already running (pid=%s)", job_name, proc.pid)
        return

    script_path = ROOT / relative_path
    env = os.environ.copy()
    mongo_uri = effective_mongo_uri()
    if mongo_uri:
        env["MONGO_URI"] = mongo_uri
        env.setdefault("LOCAL_MONGO_URI", LOCAL_MONGO_URI)
    if extra_env:
        env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")

    logging.info("Launching %s -> %s", job_name, relative_path)
    proc = subprocess.Popen(
        [PYTHON_BIN, str(script_path)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    MANAGED_PROCESSES[job_name] = proc
    if proc.stdout:
        thread = threading.Thread(
            target=stream_managed_output,
            args=(job_name, proc),
            name=f"{job_name}-stdout",
            daemon=True,
        )
        STREAM_THREADS[job_name] = thread
        thread.start()


def stream_managed_output(job_name: str, proc: subprocess.Popen) -> None:
    stdout = proc.stdout
    if stdout is None:
        return

    try:
        for line in iter(stdout.readline, ""):
            rendered = line.rstrip()
            if rendered:
                logging.info("%s | %s", job_name, rendered)
    except Exception:
        logging.exception("%s output stream crashed", job_name)
    finally:
        try:
            stdout.close()
        except Exception:
            pass


def poll_managed_processes() -> None:
    for job_name, proc in list(MANAGED_PROCESSES.items()):
        if proc.poll() is None:
            continue
        thread = STREAM_THREADS.pop(job_name, None)
        if thread:
            thread.join(timeout=1)
        logging.info("%s exited with code %s", job_name, proc.returncode)
        del MANAGED_PROCESSES[job_name]


def configure_schedule() -> None:
    if env_bool("ENABLE_FINNHUB_JOB", "true"):
        schedule.every(env_int("FINNHUB_JOB_MINUTES", 30)).minutes.do(
            run_script_once,
            "finnhub_news",
            "news_collectors/finnhub/collector.py",
            env_int("FINNHUB_JOB_TIMEOUT", 600),
        )

    if env_bool("ENABLE_NEWSAPI_JOB", "true"):
        schedule.every(env_int("NEWSAPI_JOB_MINUTES", 60)).minutes.do(
            run_script_once,
            "newsapi_news",
            "news_collectors/newsapi/collector.py",
            env_int("NEWSAPI_JOB_TIMEOUT", 600),
        )

    if env_bool("ENABLE_YAHOO_JOB", "true"):
        schedule.every(env_int("YAHOO_JOB_MINUTES", 60)).minutes.do(
            run_script_once,
            "yahoo_news",
            "news_collectors/yahoo/collector.py",
            env_int("YAHOO_JOB_TIMEOUT", 600),
        )

    if env_bool("ENABLE_GDELT_LIVE_JOB", "false"):
        schedule.every(env_int("GDELT_LIVE_JOB_MINUTES", 120)).minutes.do(
            run_script_once,
            "gdelt_live_news",
            "news_collectors/gdelt/collector.py",
            env_int("GDELT_LIVE_JOB_TIMEOUT", 1200),
        )

    if env_bool("ENABLE_DAILY_PRICE_JOB", "true"):
        schedule.every().day.at(os.getenv("DAILY_PRICE_JOB_TIME", "07:30")).do(
            run_script_once,
            "daily_price_quotes",
            "stock_collector/price_collector/collector.py",
            env_int("DAILY_PRICE_JOB_TIMEOUT", 1800),
        )

    if env_bool("ENABLE_PRICE_HISTORY_BACKFILL_JOB", "false"):
        schedule.every().day.at(os.getenv("PRICE_HISTORY_BACKFILL_JOB_TIME", "06:30")).do(
            run_script_once,
            "historical_daily_prices",
            "stock_collector/price_collector/10y_1d_history_collector.py",
            env_int("PRICE_HISTORY_BACKFILL_JOB_TIMEOUT", 7200),
        )

    if env_bool("ENABLE_DAILY_GDELT_BACKFILL_JOB", "true"):
        schedule.every().day.at(os.getenv("DAILY_GDELT_BACKFILL_JOB_TIME", "05:15")).do(
            ensure_long_running,
            "gdelt_backfill",
            "news_collectors/gdelt/historical_collector.py",
            {
                "RESET_ALL_RUNNING_ON_START": "false",
                "USE_MYSQL_BATCH_QUEUE": os.getenv("USE_MYSQL_BATCH_QUEUE", "true"),
            },
        )

    if env_bool("ENABLE_PREMARKET_JOB", "true"):
        schedule.every().day.at(os.getenv("PREMARKET_JOB_TIME", "07:45")).do(
            run_script_once,
            "premarket_signals",
            "premarket_collector/collector.py",
            env_int("PREMARKET_JOB_TIMEOUT", 900),
            {"PREMARKET_PERIOD": os.getenv("PREMARKET_PERIOD", "5d")},
        )

    if env_bool("ENABLE_INST13F_JOB", "true"):
        schedule.every().sunday.at(os.getenv("INST13F_JOB_TIME", "06:00")).do(
            run_script_once,
            "inst_13f_holdings",
            "inst_13f_collector/collector.py",
            env_int("INST13F_JOB_TIMEOUT", 1800),
        )

    if env_bool("ENABLE_ANALYST_JOB", "true"):
        schedule.every().day.at(os.getenv("ANALYST_JOB_TIME", "07:48")).do(
            run_script_once,
            "analyst_consensus",
            "analyst_collector/collector.py",
            env_int("ANALYST_JOB_TIMEOUT", 600),
        )

    if env_bool("ENABLE_MACRO_JOB", "true"):
        schedule.every().day.at(os.getenv("MACRO_JOB_TIME", "07:50")).do(
            run_script_once,
            "macro_indicators",
            "macro_collector/collector.py",
            env_int("MACRO_JOB_TIMEOUT", 600),
            {"MACRO_PERIOD": os.getenv("MACRO_PERIOD", "1mo")},
        )

    if env_bool("ENABLE_DAILY_FEATURE_JOB", "true"):
        schedule.every().day.at(os.getenv("DAILY_FEATURE_JOB_TIME", "08:00")).do(
            run_script_once,
            "daily_symbol_features",
            "research/daily_symbol_features.py",
            env_int("DAILY_FEATURE_JOB_TIMEOUT", 7200),
            {
                "FEATURE_OUTPUT_COLLECTION": os.getenv("FEATURE_OUTPUT_COLLECTION", "daily_symbol_features"),
                "FEATURE_LLM_COLLECTION": os.getenv("FEATURE_LLM_COLLECTION", "news_articles_company_matched_v2"),
            },
        )

    if env_bool("ENABLE_DAILY_SIGNAL_JOB", "true"):
        schedule.every().day.at(os.getenv("DAILY_SIGNAL_JOB_TIME", "08:30")).do(
            run_script_once,
            "score_daily_signals",
            "research/score_daily_signals.py",
            env_int("DAILY_SIGNAL_JOB_TIMEOUT", 600),
            {"SIGNAL_TOP_N": os.getenv("SIGNAL_TOP_N", "10")},
        )

    if env_bool("ENABLE_DAILY_POSITION_JOB", "true"):
        schedule.every().day.at(os.getenv("DAILY_POSITION_JOB_TIME", "08:40")).do(
            run_script_once,
            "track_positions",
            "research/track_positions.py",
            env_int("DAILY_POSITION_JOB_TIMEOUT", 600),
        )

    if env_bool("ENABLE_DATA_QUALITY_JOB", "true"):
        schedule.every().day.at(os.getenv("DATA_QUALITY_JOB_TIME", "09:00")).do(
            run_script_once,
            "data_quality_check",
            "research/data_quality_check.py",
            env_int("DATA_QUALITY_JOB_TIMEOUT", 300),
        )

    if env_bool("ENABLE_DAILY_BACKTEST_JOB", "true"):
        schedule.every().day.at(os.getenv("DAILY_BACKTEST_JOB_TIME", "08:45")).do(
            run_script_once,
            "backtest_portfolio",
            "research/backtest_portfolio.py",
            env_int("DAILY_BACKTEST_JOB_TIMEOUT", 1800),
        )

    if env_bool("ENABLE_GDELT_BACKFILL_JOB", "false"):
        schedule.every(env_int("GDELT_BACKFILL_CHECK_MINUTES", 30)).minutes.do(
            ensure_long_running,
            "gdelt_backfill",
            "news_collectors/gdelt/historical_collector.py",
            {
                "RESET_ALL_RUNNING_ON_START": "false",
                "USE_MYSQL_BATCH_QUEUE": os.getenv("USE_MYSQL_BATCH_QUEUE", "true"),
            },
        )


def run_startup_jobs() -> None:
    if env_bool("RUN_STARTUP_INCREMENTAL_JOBS", "false"):
        run_script_once("finnhub_news", "news_collectors/finnhub/collector.py", env_int("FINNHUB_JOB_TIMEOUT", 600))
        run_script_once("newsapi_news", "news_collectors/newsapi/collector.py", env_int("NEWSAPI_JOB_TIMEOUT", 600))
        run_script_once("yahoo_news", "news_collectors/yahoo/collector.py", env_int("YAHOO_JOB_TIMEOUT", 600))

    if env_bool("RUN_STARTUP_FEATURE_BUILD", "false"):
        run_script_once(
            "daily_symbol_features",
            "research/daily_symbol_features.py",
            env_int("DAILY_FEATURE_JOB_TIMEOUT", 7200),
        )

    if env_bool("RUN_STARTUP_PRICE_HISTORY_BACKFILL", "true"):
        run_script_once(
            "historical_daily_prices",
            "stock_collector/price_collector/10y_1d_history_collector.py",
            env_int("PRICE_HISTORY_BACKFILL_JOB_TIMEOUT", 7200),
        )

    if env_bool("RUN_STARTUP_GDELT_BACKFILL", "true"):
        ensure_long_running(
            "gdelt_backfill",
            "news_collectors/gdelt/historical_collector.py",
            {
                "RESET_ALL_RUNNING_ON_START": "false",
                "USE_MYSQL_BATCH_QUEUE": os.getenv("USE_MYSQL_BATCH_QUEUE", "true"),
            },
        )


def main_loop() -> None:
    logging.info("=== Quant data pipeline scheduler started ===")
    logging.info("Using Python: %s", PYTHON_BIN)
    configure_schedule()
    run_startup_jobs()

    while True:
        poll_managed_processes()
        schedule.run_pending()
        time.sleep(5)


if __name__ == "__main__":
    main_loop()
