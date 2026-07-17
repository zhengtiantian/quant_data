"""Shared constants and helpers for host-based BashOperator DAGs."""

from __future__ import annotations

from datetime import timedelta

from airflow.operators.bash import BashOperator

ROOT = "/Users/xiz/Quant_trade/quant_data"
PYTHON = f"{ROOT}/.venv311/bin/python"

LOCAL_MONGO = "mongodb://root:root@127.0.0.1:37018/quant_data?authSource=admin"

BASE_ENV = {
    "MONGO_URI": LOCAL_MONGO,
    "LOCAL_MONGO_URI": LOCAL_MONGO,
    "MYSQL_HOST": "127.0.0.1",
    "MYSQL_PORT": "23306",
    "MYSQL_USER": "root",
    "MYSQL_PASSWORD": "root",
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
