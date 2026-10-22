"""Shared constants and helpers for host-based BashOperator DAGs."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow.operators.bash import BashOperator

ROOT = "/Users/xiz/Quant_trade/quant_data"
PYTHON = f"{ROOT}/.venv311/bin/python"


def _load_dotenv_value(key: str) -> str | None:
    """Minimal .env reader (no python-dotenv dependency in .venv-airflow)."""
    env_path = Path(ROOT) / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip()
    return None


LOCAL_MONGO = os.environ.get("HOST_LOCAL_MONGO_URI") or _load_dotenv_value("HOST_LOCAL_MONGO_URI")
if not LOCAL_MONGO:
    raise RuntimeError("HOST_LOCAL_MONGO_URI not set (env var or quant_data/.env)")

BASE_ENV = {
    "MONGO_URI": LOCAL_MONGO,
    "LOCAL_MONGO_URI": LOCAL_MONGO,
    "MYSQL_HOST": "127.0.0.1",
    "MYSQL_PORT": "23306",
    "MYSQL_USER": "root",
    "MYSQL_PASSWORD": os.environ.get("HOST_MYSQL_PASSWORD") or _load_dotenv_value("HOST_MYSQL_PASSWORD") or "",
    "MYSQL_DATABASE": "workflow",
    "PYTHONUNBUFFERED": "1",
    # LM Studio model id for the SLM news filter / company match. On the M5 Max the model
    # loads as "qwen3.5-4b-mlx" (one instance); the old "qwen3.5-4b" alias no longer exists.
    "SLM_MODEL": os.getenv("SLM_MODEL", "qwen3.5-4b-mlx"),
    "SLM_MODELS": os.getenv("SLM_MODELS", "qwen3.5-4b-mlx"),
}

GDELT_ENV = {
    **BASE_ENV,
    "GDELT_CACHE_DIR": "/Volumes/Data4T/docker-volumes/gdelt_cache",
    "GKG_TMP_DIR": "/Volumes/Data4T/gdelt_tmp",
    "USE_MYSQL_BATCH_QUEUE": "true",
    "RESET_ALL_RUNNING_ON_START": "false",
    # M5 Pro/48G local LM Studio measured ~8-12 concurrent request ceiling
    # after raising LM Studio's max-concurrency setting (was ~2 before).
    # 6 leaves headroom since each worker's internal thread pool also
    # issues multiple concurrent SLM calls on its own.
    "BATCH_WORKERS": "6",
}


FAILURE_LOG = f"{ROOT}/airflow_task_failures.log"


def notify_failure(context) -> None:
    """A task failed for good (retries used up): a macOS notification, and one line in
    FAILURE_LOG. The pipeline was down for a week in October 2026 without anyone noticing;
    quant_ai's freshness watchdog covers the case where nothing runs at all."""
    ti = context.get("task_instance")
    dag_id, task_id = getattr(ti, "dag_id", "?"), getattr(ti, "task_id", "?")
    err = str(context.get("exception") or "")[:200]
    try:
        with open(FAILURE_LOG, "a") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "dag": dag_id, "task": task_id,
                                "try": getattr(ti, "try_number", None), "error": err}) + "\n")
    except OSError:
        pass
    script = f"display notification {json.dumps(err or 'see the Airflow log')} with title {json.dumps(f'Airflow failed: {dag_id}.{task_id}')}"
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001 - alerting must never raise inside a callback
        pass


def host_task(
    task_id: str,
    script: str,
    extra_env: dict | None = None,
    execution_timeout: timedelta = timedelta(hours=1),
) -> BashOperator:
    env = {**BASE_ENV, **(extra_env or {})}
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {ROOT} && {PYTHON} {ROOT}/{script}",
        env=env,
        append_env=True,
        execution_timeout=execution_timeout,
        on_failure_callback=notify_failure,
    )
