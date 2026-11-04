"""Check the drafted universe rules before any backfill (PROJECT_PLAN Q.16 news, 2026-11-03).

tools/generate_company_rules.py drafted 468 rule files with the local model; nothing has verified
that their keywords pick out the right company. This measures it on a sample of the GKG index:

  match   three sample months of the GKG Parquet (one union regex, then each company's), the rule
          engine on each hit, rules only (USE_SLM_FILTER=false: the single-weak-hit cases the SLM
          would review in the real filter pass here, so the noise measured is an upper bound), then the
          headline check of the universe backfill (backfill_symbol_news.title_gate).
          Per company: hits, kept, and up to SAMPLE kept rows -> <out>/matches.json
  judge   the larger local model reads each sampled row (company, the GKG text with the headline,
          the URL) and answers whether it is about this company -> <out>/judged.jsonl (resumable)
  report  per company: kept per sample month, judged precision; flagged when precision < 0.8 or
          nothing kept -> <out>/report.json, plus the estimated article count for the backfill

The 102 core rules (2026-11-04): BACKFILL_RULES_DIR=collectors/news/gdelt/company_rules RULE_CHECK_DIR=<dir>;
names come from the rule files. The nightly collector checks headlines as well since 2026-11-04
(CORE_TITLE_GATE); RULE_CHECK_TITLE_GATE=0 reproduces the collector before that.

Usage: .venv311/bin/python tools/validate_universe_rules.py match|judge|report
       (judge under tools/lm_session.py run --model qwen/qwen3.8-27b --workers-env LM_WORKERS -- ...)
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collectors" / "news" / "gdelt"))
os.environ.setdefault("USE_SLM_FILTER", "false")
sys.path.insert(0, str(ROOT / "tools"))
from special_rules.headline import short_names, title_gate  # noqa: E402  (the headline check the backfill and collector apply)

RULES_DIR = os.getenv("BACKFILL_RULES_DIR", "/Volumes/Data4T/universe/company_rules")
PARQUET_DIR = os.getenv("GKG_PARQUET_DIR", "/Volumes/Data4T/gkg_parquet")
OUT = Path(os.getenv("RULE_CHECK_DIR", "/Volumes/Data4T/universe/rule_check"))
NAMES = Path("/Volumes/Data4T/universe/names.json")
MONTHS = os.getenv("RULE_CHECK_MONTHS", "2021-06,2023-09,2025-09").split(",")
SAMPLE = int(os.getenv("RULE_CHECK_SAMPLE", "8"))
LM = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1").rstrip("/")
MODEL = os.getenv("RULE_CHECK_LLM", "qwen/qwen3.8-27b")
SINCE = "2021-01"
TITLE_GATE = os.getenv("RULE_CHECK_TITLE_GATE", "1") == "1"


def _names() -> dict:
    """names.json (universe), else each rule file's own name (core)."""
    names = json.loads(NAMES.read_text()) if NAMES.exists() and "universe" in RULES_DIR else {}
    for p in Path(RULES_DIR).glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        names.setdefault(d.get("symbol") or p.stem, {"name": d.get("name") or d.get("company_name") or p.stem})
    return names


def match() -> None:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    from special_rules import RuleManager
    rm = RuleManager(RULES_DIR)
    syms = sorted(rm.company_configs)
    out = {s: {"hits": 0, "kept": 0, "months": {}, "keywords": None, "sample": []} for s in syms}
    rng = random.Random(7)
    t0 = time.time()
    for month in MONTHS:
        mid = datetime.strptime(month + "-15", "%Y-%m-%d")
        kw = {s: [k for k in rm.get_keywords(s, mid) if len(k) > 1] for s in syms}
        rx = {s: "(?i)\\b(" + "|".join(re.escape(k) for k in v) + ")\\b" for s, v in kw.items() if v}
        union = "(?i)\\b(" + "|".join(sorted({re.escape(k) for v in kw.values() for k in v})) + ")\\b"
        t = ds.dataset(f"{PARQUET_DIR}/month={month}", format="parquet").to_table(columns=["ts", "url", "raw"])
        total = t.num_rows
        t = t.filter(pc.match_substring_regex(t["raw"], union))
        print(f"{month}: {total} rows, {t.num_rows} match some company ({time.time() - t0:.0f}s)", flush=True)
        for i, s in enumerate(syms, 1):
            out[s]["keywords"] = kw[s]
            if s not in rx:
                continue
            rows = {}
            for r in t.filter(pc.match_substring_regex(t["raw"], rx[s])).to_pylist():
                rows.setdefault(r["url"], r)
            ruled = [r for r in rows.values()
                     if rm.should_include(s, {"title": "", "content": r["raw"], "url": r["url"], "date": str(r["ts"])[:8]})]
            kept = [r for r in ruled if not TITLE_GATE or title_gate(kw[s] + short_names(rm.company_configs.get(s, {})), r["raw"], r["url"])]
            out[s]["hits"] += len(rows)
            out[s]["ruled"] = out[s].get("ruled", 0) + len(ruled)
            out[s]["kept"] += len(kept)
            out[s]["months"][month] = {"hits": len(rows), "kept": len(kept)}
            pick = rng.sample(kept, min(len(kept), max(1, SAMPLE // len(MONTHS) + 1)))
            out[s]["sample"] += [{"month": month, "url": r["url"], "text": r["raw"][-700:]} for r in pick]
            if i % 100 == 0:
                print(f"   {month} {i}/{len(syms)} ({time.time() - t0:.0f}s)", flush=True)
    for s in out:
        out[s]["sample"] = rng.sample(out[s]["sample"], min(SAMPLE, len(out[s]["sample"])))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "matches.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"match done: {len(syms)} companies, {sum(v['kept'] for v in out.values())} kept rows ({time.time() - t0:.0f}s)")


def _ask(company: str, sym: str, info: dict, row: dict) -> str:
    prompt = (f"Company: {company} (ticker {sym}; sector {info.get('sector')}; industry {info.get('industry')}).\n"
              f"URL: {row['url']}\nGDELT record (organisations, names and the headline at the end):\n{row['text']}\n\n"
              "Is this news item about this company - the company itself, its business, products, people or stock - "
              "rather than something else that shares a word with its name (a place, a person, another company, an "
              "ordinary word)? Answer with one word: yes, no, or unsure.")
    r = __import__("requests").post(f"{LM}/chat/completions", timeout=300, json={
        "model": MODEL, "temperature": 0, "max_tokens": 5,
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "<think>\n\n</think>\n\n"}]})
    a = re.sub(r"<think>.*?</think>", "", r.json()["choices"][0]["message"]["content"], flags=re.S).strip().lower()
    return "yes" if a.startswith("yes") else "no" if a.startswith("no") else "unsure"


def judge() -> None:
    m = json.loads((OUT / "matches.json").read_text())
    names = _names()
    path = OUT / "judged.jsonl"
    done = {(j["symbol"], j["url"]) for j in map(json.loads, path.open())} if path.exists() else set()
    todo = [(s, r) for s, v in m.items() for r in v["sample"] if (s, r["url"]) not in done]
    workers = int(os.getenv("LM_WORKERS", "4"))
    print(f"judge: {len(todo)} rows, {workers} workers", flush=True)
    t0 = time.time()

    def one(item):
        s, r = item
        info = names.get(s) or {}
        try:
            return s, r, _ask(info.get("name") or s, s, info, r)
        except Exception as e:  # noqa: BLE001
            return s, r, f"error {type(e).__name__}"
    with ThreadPoolExecutor(workers) as ex, path.open("a") as f:
        for i, (s, r, a) in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps({"symbol": s, "url": r["url"], "month": r["month"], "answer": a}) + "\n")
            if i % 50 == 0 or i == len(todo):
                f.flush()
                print(f"PROGRESS judge {i} {len(todo)} {time.time() - t0:.0f}s", flush=True)


def report() -> dict:
    m = json.loads((OUT / "matches.json").read_text())
    # only this sample's rows: judged.jsonl may carry answers from an earlier match (copied in to save model
    # time), and those rows may no longer be kept (2026-11-04: they made the fixed rules look worse)
    sample = {(s, r["url"]) for s, v in m.items() for r in v["sample"]}
    judged: dict = {}
    if (OUT / "judged.jsonl").exists():
        seen = set()
        for j in map(json.loads, (OUT / "judged.jsonl").open()):
            k = (j["symbol"], j["url"])
            if k in sample and k not in seen:
                seen.add(k)
                judged.setdefault(j["symbol"], []).append(j["answer"])
    months_total = len(glob.glob(f"{PARQUET_DIR}/month=*"))
    since_months = len([p for p in glob.glob(f"{PARQUET_DIR}/month=*") if p.split("=")[1] >= SINCE])
    rows, flagged = {}, []
    for s, v in m.items():
        a = judged.get(s, [])
        yes, no = a.count("yes"), a.count("no")
        prec = round(yes / (yes + no), 2) if yes + no else None
        rows[s] = {"kept_per_month": round(v["kept"] / len(MONTHS), 1), "hits": v["hits"], "kept": v["kept"],
                   "judged": len(a), "yes": yes, "no": no, "precision": prec, "keywords": v["keywords"]}
        if v["kept"] == 0 or (prec is not None and prec < 0.8):
            flagged.append(s)
    est = sum(r["kept_per_month"] for r in rows.values()) * since_months
    allj = [x for a in judged.values() for x in a]
    rep = {"months": MONTHS, "companies": len(rows), "flagged": sorted(flagged),
           "nothing_kept": sorted(s for s, r in rows.items() if r["kept"] == 0),
           "judged": len(allj), "overall_precision": round(allj.count("yes") / max(1, allj.count("yes") + allj.count("no")), 3),
           "estimated_rows_since_2021": int(est), "parquet_months": months_total, "companies_detail": rows}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "companies_detail"}, indent=1))
    return rep


if __name__ == "__main__":
    {"match": match, "judge": judge, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
