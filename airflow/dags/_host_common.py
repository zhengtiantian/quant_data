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
    _notify(f"Airflow failed: {dag_id}.{task_id}", err or "see the Airflow log")


def _notify(title: str, message: str) -> None:
    script = f"display notification {json.dumps(message)} with title {json.dumps(title)}"
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001 - alerting must never raise inside a callback
        pass


# The nightly news chain: backfill_1 (02:00 UTC) triggers backfill_2 when company match is done.
# Whatever reads its output (news features) waits for it to finish instead of racing it: on
# 2026-10-24 a catch-up ran until 15:00, long past the 07:30 signals.
NIGHT_CHAIN = ("backfill_1_collect_and_match", "backfill_2_enrich_and_features")
NIGHT_CHAIN_MAX_WAIT = timedelta(hours=6)


def _night_chain_busy() -> list[str]:
    from airflow.models import DagRun
    from airflow.utils.state import DagRunState
    return [f"{r.dag_id} {r.run_id}" for d in NIGHT_CHAIN for st in (DagRunState.RUNNING, DagRunState.QUEUED)
            for r in DagRun.find(dag_id=d, state=st)]


def wait_for_night_chain(on_timeout: str = "proceed"):
    """First task of a DAG that must run after the nightly chain. Checks every 5 minutes.
    on_timeout="proceed": after NIGHT_CHAIN_MAX_WAIT go ahead anyway with a notification (a day
    without signals is worse than signals on yesterday's news features). on_timeout="skip": skip
    this run (gdelt_batch_verify must not reopen batches while the collector is filling them)."""
    from airflow.sensors.python import PythonSensor

    def done(dag_run=None, **_):
        busy = _night_chain_busy()
        if not busy:
            return True
        waited = datetime.now(dag_run.start_date.tzinfo) - dag_run.start_date if dag_run else timedelta(0)
        print(f"nightly chain still running after {waited}: {busy}")
        if on_timeout == "proceed" and waited >= NIGHT_CHAIN_MAX_WAIT:
            _notify(f"Airflow: {dag_run.dag_id} went ahead", f"nightly chain still running after {waited}: {busy[0]}")
            return True
        return False

    return PythonSensor(
        task_id="wait_for_night_chain",
        python_callable=done,
        mode="reschedule",
        poke_interval=300,
        timeout=(NIGHT_CHAIN_MAX_WAIT + timedelta(minutes=30)).total_seconds(),
        soft_fail=on_timeout == "skip",
        retries=0,
        on_failure_callback=notify_failure,
    )


def lm_command(command: str, model: str, workers_env: tuple = (), ids_env: str | None = None,
               second_id_env: str | None = None) -> str:
    """Run `command` through tools/lm_session.py: wait for LM Studio to be free (one LLM task at
    a time, job_agent included), load `model` with the slots this machine measured for it, set
    the task's worker variables to match, unload it afterwards."""
    opts = [f"--model {model}"] + [f"--workers-env {w}" for w in workers_env]
    if ids_env:
        opts.append(f"--ids-env {ids_env}")
    if second_id_env:
        opts.append(f"--second-id-env {second_id_env}")
    return f"{PYTHON} {ROOT}/tools/lm_session.py run {' '.join(opts)} -- {command}"


def host_task(
    task_id: str,
    script: str,
    extra_env: dict | None = None,
    execution_timeout: timedelta = timedelta(hours=1),
    lm_model: str | None = None,
    lm_workers_env: tuple = (),
    lm_ids_env: str | None = None,
    lm_second_id_env: str | None = None,
) -> BashOperator:
    env = {**BASE_ENV, **(extra_env or {})}
    command = f"{PYTHON} {ROOT}/{script}"
    if lm_model:
        command = lm_command(command, lm_model, lm_workers_env, lm_ids_env, lm_second_id_env)
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {ROOT} && {command}",
        env=env,
        append_env=True,
        execution_timeout=execution_timeout,
        on_failure_callback=notify_failure,
    )
