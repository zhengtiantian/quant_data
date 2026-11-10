#!/usr/bin/env python3
"""The 490 universe companies' articles through the students, in order: relevance filter, then
labels A and B for the rows that passed (2026-11-10).

    universe_students.py filter | label_a | label_b | all        (all = the three in order)

Each stage runs in chunks, every chunk its own tools/lm_session.py `local` session, so the hourly
GDELT match, the nightly chain and job_agent get the lock between chunks; no chunk starts in the
quarter hour before job_agent's 11:00 / 16:00 runs. Stages resume: the filter by its own checkpoint
(company_match_jobs "universe_student_filter"), the labels by the missing label field.
The filter writes its verdict onto the rows already in the matched collection (V2_ANNOTATE).
Start it detached (double fork + setsid); progress goes to stdout.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv311" / "bin" / "python")
STUDENT_PYTHON = os.getenv("STUDENT_PYTHON", os.path.expanduser("~/.venvs/vllm-mlx/bin/python"))
MONGO = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
CHUNK = int(os.getenv("UNIVERSE_CHUNK", "20000"))
QUIET = ((10 * 60 + 45, 11 * 60 + 3), (15 * 60 + 45, 16 * 60 + 3))     # local minutes of day
JOB = "universe_student_filter"


def log(msg: str) -> None:
    print(f"[{datetime.now():%m-%d %H:%M}] {msg}", flush=True)


def wait_quiet() -> None:
    while True:
        now = datetime.now()
        m = now.hour * 60 + now.minute
        if not any(a <= m < b for a, b in QUIET):
            return
        time.sleep(60)


def db():
    from pymongo import MongoClient
    return MongoClient(MONGO)["quant_data"]


def session(name: str, script: str, env: dict) -> int:
    wait_quiet()
    cmd = [PYTHON, str(ROOT / "tools" / "lm_session.py"), "local", "--name", name, "--",
           STUDENT_PYTHON, str(ROOT / script)]
    return subprocess.call(cmd, cwd=ROOT, env={**os.environ, "LOCAL_MONGO_URI": MONGO, "PYTHONUNBUFFERED": "1", **env})


def stage_filter() -> None:
    d = db()
    env = {"V2_ENGINE": "student", "V2_ANNOTATE": "1", "V2_INCLUDE_UNIVERSE": "1", "V2_JOB_NAME": JOB,
           "V2_SOURCE_FILTER": '{"origin": "universe_backfill"}', "V2_LIMIT": str(CHUNK), "V2_PROGRESS_EVERY": "5000"}
    total = d.news_articles.count_documents({"origin": "universe_backfill"})
    while True:
        p = d.company_match_jobs.find_one({"_id": JOB}) or {}
        before = int(p.get("processed", 0))
        log(f"filter: {before:,}/{total:,} judged, {int(p.get('matched', 0)):,} kept")
        if session("universe-filter", "research/labeling/slm_company_match_v2.py", env) != 0:
            sys.exit("filter chunk failed")
        after = int((d.company_match_jobs.find_one({"_id": JOB}) or {}).get("processed", 0))
        if after - before < CHUNK:
            log(f"filter done: {after:,} judged")
            return


def stage_label(suffix: str) -> None:
    d = db()
    q = {f"llm_sentiment_{suffix}": {"$exists": False}, "company_match_v2.matched": {"$ne": False},
         "$or": [{"origin": {"$ne": "universe_backfill"}}, {"company_match_v2.engine": "student"}]}
    env = {"ENRICH_PASS": suffix.upper(), "ENRICH_INCLUDE_UNIVERSE": "1", "ENRICH_LIMIT": str(CHUNK)}
    last = None
    while True:
        left = d.news_articles_company_matched_v2.count_documents(q)
        log(f"label {suffix.upper()}: {left:,} rows to go")
        if left == 0 or left == last:          # nothing left, or only rows whose answer never parses
            return
        last = left
        if session(f"universe-label-{suffix}", "research/labeling/student_enrich_articles.py", env) != 0:
            sys.exit(f"label {suffix} chunk failed")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage not in ("filter", "label_a", "label_b", "all"):
        sys.exit(__doc__)
    if stage in ("filter", "all"):
        stage_filter()
    if stage in ("label_a", "all"):
        stage_label("a")
    if stage in ("label_b", "all"):
        stage_label("b")
    log("done")
