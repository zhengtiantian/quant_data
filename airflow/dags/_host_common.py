"""Shared constants and helpers for host-based BashOperator DAGs."""

from __future__ import annotations

import os
from datetime import timedelta
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
}

GDELT_ENV = {
    **BASE_ENV,
    "GDELT_CACHE_DIR": "/Volumes/Data4T/docker-volumes/gdelt_cache",
    "GKG_TMP_DIR": "/Volumes/Data4T/gdelt_tmp",
    "USE_MYSQL_BATCH_QUEUE": "true",
    "RESET_ALL_RUNNING_ON_START": "false",
}


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
    )
