"""News for the wider universe, 2021 on (PROJECT_PLAN Q.16, 2026-11-02).

The user chose, until a bigger machine runs the full pipeline: the last five years only, and titles
labelled by one model. Steps, each resumable:

  scan    every month's GKG Parquet read once, every company's keywords matched over it (one union
          regex first, then each company's) -> <work>/<SYM>/candidates.jsonl
  filter  the collector's rules per company (universe rule files, tools/generate_company_rules.py)
  fetch   download and store in news_articles_universe (not news_articles: the nightly company match and
          two-model enrichment stay on the original companies)
  label   research/labeling/llm_enrich_articles.py, pass A, one model, titles only, on that collection

Usage: .venv311/bin/python tools/backfill_universe_news.py scan|filter|fetch|label|all [SYMBOLS...]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BACKFILL_RULES_DIR", "/Volumes/Data4T/universe/company_rules")
os.environ.setdefault("BACKFILL_COLLECTION", "news_articles_universe")
os.environ.setdefault("BACKFILL_WORK_DIR", "/Volumes/Data4T/universe/news_backfill")
os.environ.setdefault("MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
SINCE = os.getenv("UNIVERSE_NEWS_SINCE", "2021-01-01")
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "collectors" / "news" / "gdelt"))

import backfill_symbol_news as bf  # noqa: E402


def rules():
    from special_rules import RuleManager
    return RuleManager(os.environ["BACKFILL_RULES_DIR"])


def main(stage: str, syms: list) -> None:
    rm = rules()
    syms = syms or sorted(rm.company_configs)
    since_dt = datetime.strptime(SINCE, "%Y-%m-%d")
    work = Path(os.environ["BACKFILL_WORK_DIR"])
    t0 = time.time()
    if stage in ("scan", "all"):
        bf.scan_many({s: rm.get_keywords(s, since_dt) for s in syms}, SINCE)
    for i, s in enumerate(syms, 1):
        cand, kept = work / s / "candidates.jsonl", work / s / "kept.jsonl"
        if stage in ("filter", "all") and cand.exists():
            bf.filter_stage(s, cand, kept, int(os.getenv("LM_WORKERS", "8")))
        if stage in ("fetch", "all") and kept.exists():
            bf.fetch_stage(s, rm.company_configs[s].get("name") or s, kept, 12)
        if stage in ("filter", "fetch", "all"):
            print(f"PROGRESS company {i} {len(syms)} {s} {time.time() - t0:.0f}s", flush=True)
    if stage in ("label", "all"):
        env = {**os.environ, "ENRICH_COLLECTION": os.environ["BACKFILL_COLLECTION"], "ENRICH_TITLE_ONLY": "1", "ENRICH_PASS": "A",
               "SLM_MODEL_A": os.getenv("UNIVERSE_LABEL_MODEL", "qwen3.5-9b-mlx"), "SLM_MODEL_A2": "",
               "SLM_API_URL": os.getenv("SLM_API_URL", "http://localhost:1234/v1"), "LOCAL_MONGO_URI": os.environ["MONGO_URI"]}
        subprocess.run([sys.executable, str(ROOT / "research/labeling/llm_enrich_articles.py")], env=env, check=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all", sys.argv[2:])
