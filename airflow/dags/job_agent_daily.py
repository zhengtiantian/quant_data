"""
Job Agent Unified Schedules (host-based BashOperator), migrated from launchd com.xiangyu.jobagent.

Staggered multi-region schedule (Europe/Dublin local time, auto-adjusting for GMT/IST):
  1. job_agent_morning   (11:00 daily, 19:00 Beijing):
       materialize_drive_files
         >> ireland_morning_scan          (Ireland full: LinkedIn + Indeed + 200+ ATS + LLM judge + report; auto 7d full sweep)
         >> china_fulltime_daily          (China full: 201 foreign MNC portals + Tencent/JD + LinkedIn CN + Indeed CN/HK + communities)
         >> china_parttime_daily          (China remote part-time: Eleduck + V2EX + GitHub + Indeed + LinkedIn + judge)
         >> eu_me_official_ats_morning    (EU/UK + Middle East + Remote fast official-ATS-only sync, 0 LinkedIn calls)
         >> backup_db
  2. job_agent_afternoon (16:00 daily):
       materialize_drive_files
         >> ireland_afternoon_scan        (Ireland afternoon full scan + LLM judge + report)
         >> eu_me_remote_daily            (EU/UK + Middle East + Remote full nationwide LinkedIn + ATS; Sunday 7d full sweep)
         >> china_official_ats_afternoon  (China fast 201 foreign MNC portals + Tencent/JD + communities only, 0 LinkedIn calls)
         >> backup_db

Notes:
  - Runs via local APFS bootstrapper (~/Library/Application Support/job_agent/airflow_entry.py) invoked
    directly by ~/.venvs/job_agent/bin/python3 so launchd-spawned bash never touches ~/Library/CloudStorage.
  - LLM tasks coordinate with quant_data through ~/Library/Application Support/lm_session/lmstudio.lock.
"""

from __future__ import annotations

from datetime import timedelta
import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator

from _host_common import notify_failure

JOB_PYTHON = "/Users/xiz/.venvs/job_agent/bin/python3"
ENTRY_SCRIPT = "/Users/xiz/Library/Application Support/job_agent/airflow_entry.py"
DUBLIN_TZ = pendulum.timezone("Europe/Dublin")

DEFAULT_ARGS = {
    "owner": "job_agent",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _job_task(
    task_id: str,
    step: str,
    timeout_minutes: int = 45,
    retries: int = 1,
    trigger_rule: str = "all_success",
) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f'{JOB_PYTHON} "{ENTRY_SCRIPT}" {step}',
        env={"PYTHONUNBUFFERED": "1"},
        append_env=True,
        execution_timeout=timedelta(minutes=timeout_minutes),
        retries=retries,
        on_failure_callback=notify_failure,
        trigger_rule=trigger_rule,
        pool="job_agent",
    )


with DAG(
    dag_id="job_agent_morning",
    default_args=DEFAULT_ARGS,
    description="11:00 Dublin daily: Ireland (1/2) + China Full ( Foreign/Domestic/Part-time ) + EU/ME Official ATS fast sync",
    schedule="0 11 * * *",
    start_date=pendulum.datetime(2024, 1, 1, tz=DUBLIN_TZ),
    catchup=False,
    max_active_runs=1,
    tags=["job_agent", "daily", "ireland", "china"],
) as dag_morning:
    m_mat = _job_task("materialize_drive_files", "materialize", timeout_minutes=10, retries=2)
    m_ireland = _job_task("ireland_morning_scan", "ireland", timeout_minutes=60)
    m_cn_ft = _job_task("china_fulltime_daily", "china_fulltime_all", timeout_minutes=45)
    m_cn_pt = _job_task("china_parttime_daily", "china_parttime", timeout_minutes=35)
    m_gl_fast = _job_task("eu_me_official_ats_morning", "global_official_fast", timeout_minutes=20)
    m_backup = _job_task("backup_db", "backup_db", timeout_minutes=10, trigger_rule="all_done")

    m_mat >> m_ireland >> m_cn_ft >> m_cn_pt >> m_gl_fast >> m_backup


with DAG(
    dag_id="job_agent_afternoon",
    default_args=DEFAULT_ARGS,
    description="16:00 Dublin daily: Ireland (2/2) + EU/UK & Middle East & Remote Full + China 201 Foreign Official ATS fast sync",
    schedule="0 16 * * *",
    start_date=pendulum.datetime(2024, 1, 1, tz=DUBLIN_TZ),
    catchup=False,
    max_active_runs=1,
    tags=["job_agent", "daily", "ireland", "eu_me_remote"],
) as dag_afternoon:
    a_mat = _job_task("materialize_drive_files", "materialize", timeout_minutes=10, retries=2)
    a_ireland = _job_task("ireland_afternoon_scan", "ireland", timeout_minutes=60)
    a_global = _job_task("eu_me_remote_daily", "global_all", timeout_minutes=50)
    a_cn_fast = _job_task("china_official_ats_afternoon", "china_official_fast", timeout_minutes=25)
    a_backup = _job_task("backup_db", "backup_db", timeout_minutes=10, trigger_rule="all_done")

    a_mat >> a_ireland >> a_global >> a_cn_fast >> a_backup
