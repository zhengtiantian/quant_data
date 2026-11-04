"""News for the wider universe, 2021 on (PROJECT_PLAN Q.16, 2026-11-02; time windows 2026-11-03).

The user chose, until a bigger machine runs the full pipeline: the last five years only, labelled from
the GKG headline by one model. Steps, each resumable:

  scan    every month's GKG Parquet read once, every company's keywords (all it had since 2021) matched
          over it -> <work>/<SYM>/candidates.jsonl. The current month is left out (still being filled).
  filter  the collector's rules per company (universe rule files, tools/generate_company_rules.py), then
          the headline check (BACKFILL_TITLE_GATE): the company named in the headline or URL
          -> <work>/<SYM>/kept.jsonl. The SLM reviews single weak hits (BBY, PSA, WM only).
  index   every company's kept rows regrouped by month, headline extracted -> <labels>/todo/<YYYY-MM>.jsonl
  label   newest month first. Rows of one company with the same headline (syndicated copies, ~20% in
          2021-06 and 2025-09) are labelled once. llm_enrich_articles.call_llm with today's prompt, title
          only, one headline per request: on 500 headlines (2026-11-03) a short prompt ran 1.7x faster but
          agreed with it on 76% of event types against 98% for a rerun of the same prompt, and 20
          headlines per request agreed on 71%; the labels must match the ones already stored.
          -> <labels>/<YYYY-MM>.jsonl, one row per kept URL; <labels>/<YYYY-MM>.keys.jsonl while a
          month is under way
  fetch   (not scheduled) download and store in news_articles_universe
  (rescan) before the labels, each window runs tools/keyword_refresh.py rescan: past months for keywords
          added in the weekly review (2026-11-04)

  window --until HH:MM   the scheduled form (airflow/dags/universe_news_backfill.py: 11:30-15:45 and
          16:30-23:00, the hours the user gave, clear of job_agent at 11:00 / 16:00 and the 04:00 chain):
          runs whatever is unfinished in the order above and stops before HH:MM, between months
          (scan), companies (filter) or chunks of headlines (label).

Usage: .venv311/bin/python tools/backfill_universe_news.py scan|filter|index|label|fetch|status [SYMBOLS...]
       .venv311/bin/python tools/backfill_universe_news.py window --until 15:00
       (under tools/lm_session.py run --model qwen3.5-9b-mlx --workers-env LM_WORKERS -- ...)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL_MODEL = os.getenv("UNIVERSE_LABEL_MODEL", "qwen3.5-9b-mlx")
os.environ.setdefault("BACKFILL_RULES_DIR", "/Volumes/Data4T/universe/company_rules")
os.environ.setdefault("BACKFILL_COLLECTION", "news_articles_universe")
os.environ.setdefault("BACKFILL_WORK_DIR", "/Volumes/Data4T/universe/news_backfill")
os.environ.setdefault("BACKFILL_TITLE_GATE", "1")
os.environ.setdefault("MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
# the SLM filter and the labels both use the model lm_session loaded, so it is never swapped mid-run
os.environ["SLM_MODEL"] = os.environ["SLM_MODELS"] = LABEL_MODEL
os.environ.setdefault("SLM_API_URL", "http://localhost:1234/v1")
os.environ.setdefault("ENRICH_LLM_TIMEOUT", "120")
SINCE = os.getenv("UNIVERSE_NEWS_SINCE", "2021-01-01")
LABELS = Path(os.getenv("UNIVERSE_LABELS_DIR", "/Volumes/Data4T/universe/news_labels"))
WORK = Path(os.environ["BACKFILL_WORK_DIR"])
NAMES = Path("/Volumes/Data4T/universe/names.json")
WORKERS = int(os.getenv("LM_WORKERS", "8"))
MAX_TRIES = 2
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "collectors" / "news" / "gdelt"))
sys.path.insert(0, str(ROOT / "research" / "labeling"))

import backfill_symbol_news as bf  # noqa: E402


def rules():
    from special_rules import RuleManager
    return RuleManager(os.environ["BACKFILL_RULES_DIR"])


def all_keywords(rm, sym: str) -> list:
    """Every keyword the company had at some point since SINCE (renamed companies change keywords)."""
    kw, d = set(), datetime.strptime(SINCE, "%Y-%m-%d")
    while d <= datetime.now():
        kw |= set(rm.get_keywords(sym, d))
        d = d.replace(year=d.year + (d.month == 12), month=d.month % 12 + 1)
    return sorted(kw)


def this_month() -> str:
    return datetime.now().strftime("%Y-%m")


def scan(rm, syms: list, deadline: float | None = None) -> bool:
    done = WORK / "scan.done"
    if done.exists():
        return True
    ok = bf.scan_many({s: all_keywords(rm, s) for s in syms}, SINCE, deadline, before_month=this_month())
    if ok:
        done.write_text(this_month())
    return ok


def filter_all(rm, syms: list, deadline: float | None = None) -> bool:
    t0 = time.time()
    for i, s in enumerate(syms, 1):
        cand, kept, mark = WORK / s / "candidates.jsonl", WORK / s / "kept.jsonl", WORK / s / "filter.done"
        if mark.exists() or not cand.exists():
            continue
        if deadline and time.time() >= deadline:
            return False
        bf.filter_stage(s, cand, kept, WORKERS)
        kept.touch()
        mark.write_text(datetime.now().isoformat(timespec="seconds"))
        print(f"PROGRESS filter_company {i} {len(syms)} {s} {time.time() - t0:.0f}s", flush=True)
    return True


def key_of(sym: str, title: str) -> str:
    return sym + "|" + re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()[-120:]


def index(syms: list) -> None:
    """kept rows by month: {symbol, url, ts, title, key}."""
    todo = LABELS / "todo"
    if (todo / "index.done").exists():
        return
    todo.mkdir(parents=True, exist_ok=True)
    for p in todo.glob("*.jsonl"):
        p.unlink()                                  # a half-written index from an interrupted run
    files: dict = {}
    for s in syms:
        kept = WORK / s / "kept.jsonl"
        if not kept.exists():
            continue
        for line in kept.open():
            r = json.loads(line)
            title = bf.gkg_title(r["raw"])
            m = f"{str(r['ts'])[:4]}-{str(r['ts'])[4:6]}"
            if m not in files:
                files[m] = (todo / f"{m}.jsonl").open("a")
            files[m].write(json.dumps({"symbol": s, "url": r["url"], "ts": str(r["ts"]), "title": title,
                                       "key": key_of(s, title)}) + "\n")
    for f in files.values():
        f.close()
    (todo / "index.done").write_text(datetime.now().isoformat(timespec="seconds"))
    print(f"index: {len(files)} months", flush=True)


def _month_keys(month: str) -> tuple:
    rows = [json.loads(l) for l in (LABELS / "todo" / f"{month}.jsonl").open()]
    first = {}
    for r in rows:
        first.setdefault(r["key"], r)
    labelled, tries = {}, {}
    kf = LABELS / f"{month}.keys.jsonl"
    if kf.exists():
        for k in map(json.loads, kf.open()):
            if k["label"]:
                labelled[k["key"]] = k["label"]
            else:
                tries[k["key"]] = tries.get(k["key"], 0) + 1
    todo = [r for key, r in first.items() if key not in labelled and tries.get(key, 0) < MAX_TRIES]
    return rows, first, labelled, todo


def label(deadline: float | None = None) -> bool:
    import llm_enrich_articles as le
    names = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    months = sorted((p.stem for p in (LABELS / "todo").glob("*.jsonl")), reverse=True)
    t0, n_done, chunk_secs = time.time(), 0, 0.0
    for month in months:
        if (LABELS / f"{month}.jsonl").exists():
            continue
        rows, first, labelled, todo = _month_keys(month)
        print(f"label {month}: {len(rows)} rows, {len(first)} distinct headlines, {len(todo)} to label", flush=True)
        step = WORKERS * 8
        with ThreadPoolExecutor(WORKERS) as ex, (LABELS / f"{month}.keys.jsonl").open("a") as kf:
            for i in range(0, len(todo), step):
                if deadline and time.time() + max(chunk_secs, 30) >= deadline:
                    print(f"window ends: stopped in {month}; {n_done} labelled this run "
                          f"({n_done / max(1, time.time() - t0):.2f}/s)", flush=True)
                    return False
                c0, chunk = time.time(), todo[i:i + step]
                out = list(ex.map(lambda r: le.call_llm(r["title"], "", r["symbol"], (names.get(r["symbol"]) or {}).get("name") or r["symbol"],
                                                        LABEL_MODEL), chunk))
                for r, lab in zip(chunk, out):
                    kf.write(json.dumps({"key": r["key"], "label": lab}) + "\n")
                kf.flush()
                chunk_secs, n_done = time.time() - c0, n_done + len(chunk)
                print(f"PROGRESS label {month} {min(i + step, len(todo))} {len(todo)} {n_done / max(1, time.time() - t0):.2f}/s", flush=True)
        rows, first, labelled, todo = _month_keys(month)
        if todo:                                    # failures get one more try next run
            continue
        with (LABELS / f"{month}.jsonl.part").open("w") as f:
            for r in rows:
                lab = labelled.get(r["key"])
                f.write(json.dumps({"symbol": r["symbol"], "url": r["url"], "ts": r["ts"], "title": r["title"],
                                    "dup_of_key": r["key"] if first[r["key"]]["url"] != r["url"] else None,
                                    "label": lab, "model": LABEL_MODEL, "mode": "gkg_title_v1"}) + "\n")
        (LABELS / f"{month}.jsonl.part").rename(LABELS / f"{month}.jsonl")
        print(f"label {month} done", flush=True)
    return True


def status() -> dict:
    todo = LABELS / "todo"
    months = sorted(p.stem for p in todo.glob("*.jsonl")) if todo.exists() else []
    left = 0
    for m in months:
        if not (LABELS / f"{m}.jsonl").exists():
            left += len(_month_keys(m)[3])
    out = {"scan_done": (WORK / "scan.done").exists(),
           "filtered": sum(1 for _ in WORK.glob("*/filter.done")) if WORK.exists() else 0,
           "indexed": (todo / "index.done").exists(), "months": len(months),
           "months_labelled": sum((LABELS / f"{m}.jsonl").exists() for m in months), "headlines_left": left}
    print(json.dumps(out), flush=True)
    return out


def window(until: str) -> None:
    h, m = map(int, until.split(":"))
    end = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0).timestamp()
    if time.time() >= end - 120:
        print(f"started after the window ({until}); nothing to do")
        return
    rm = rules()
    syms = sorted(rm.company_configs)
    print(f"window until {until}: {len(syms)} companies, model {LABEL_MODEL}, {WORKERS} workers", flush=True)
    if scan(rm, syms, end) and filter_all(rm, syms, end):
        index(syms)
        # keywords the user added in the weekly review: their past months first (core and universe), so the
        # new universe headlines join this run's labels (tools/keyword_refresh.py rescan, 2026-11-04)
        import keyword_refresh
        try:
            go_on = keyword_refresh.rescan(end)
        except Exception as e:  # noqa: BLE001 - a failed rescan must not cost the window its labels
            print(f"rescan failed, labels go on: {type(e).__name__}: {e}", flush=True)
            go_on = True
        if go_on:
            label(end)
    status()


def main(stage: str, args: list) -> None:
    if stage == "window":
        window(args[args.index("--until") + 1] if "--until" in args else "23:59")
        return
    if stage == "status":
        status()
        return
    rm = rules()
    syms = args or sorted(rm.company_configs)
    if stage == "scan":
        scan(rm, syms)
    elif stage == "filter":
        filter_all(rm, syms)
    elif stage == "index":
        index(syms)
    elif stage == "label":
        label()
    elif stage == "fetch":
        for s in syms:
            kept = WORK / s / "kept.jsonl"
            if kept.exists():
                bf.fetch_stage(s, rm.company_configs[s].get("name") or s, kept, 12)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "status", sys.argv[2:])
