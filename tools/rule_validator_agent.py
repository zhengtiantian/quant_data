"""
Rule Validator Agent — per-symbol validation, results for manual review.
Pipeline: keyword match → SLM(rule) → URL fetch → SLM(content)
"""
import sys
import os
import random
import json
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

project_root = "/Users/xiz/Quant_trade/quant_data"
collector_dir = os.path.join(project_root, "news_collectors/gdelt")
for path in [project_root, collector_dir]:
    if path not in sys.path:
        sys.path.append(path)

os.environ["USE_SLM_FILTER"] = "true"

try:
    from historical_collector import process_file_task, fetch_article, RuleManager
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

BASE_DIR       = "/Volumes/Data24T/docker-volumes/gdelt_cache/files"
RULES_DIR      = os.path.join(project_root, "news_collectors/gdelt/company_rules")
FILES_PER_YEAR = 100
FETCH_WORKERS  = 6
MAX_FETCH      = 10   # 每个 symbol 最多抓取正文数

TECH_26 = [
    "TSM", "ASML", "CRM", "PLTR", "NOW", "ADBE", "NFLX", "UBER",
    "SNOW", "MDB", "PANW", "CRWD", "SMCI", "AMAT", "LRCX", "KLAC",
    "TXN", "ADI", "MCHP", "ORCL", "FTNT", "ABNB", "CSCO", "IBM",
    "DELL", "INTU",
]


def _all_zips_by_year():
    all_zips = glob(os.path.join(BASE_DIR, "*.gkg.csv.zip"))
    by_year = {}
    for z in all_zips:
        y = os.path.basename(z)[:4]
        if y.isdigit() and 2016 <= int(y) <= datetime.now().year:
            by_year.setdefault(y, []).append(z)
    return by_year


def _sample_files(by_year, per_year=FILES_PER_YEAR):
    files = []
    for y in sorted(by_year):
        files.extend(random.sample(by_year[y], min(per_year, len(by_year[y]))))
    return files


def _avg_date(zip_paths):
    dates = []
    for z in zip_paths:
        try:
            dates.append(datetime.strptime(os.path.basename(z)[:8], "%Y%m%d"))
        except Exception:
            pass
    return (min(dates) + (max(dates) - min(dates)) / 2) if dates else datetime.now()


def _build_keyword_map(rule_manager, symbol, avg_date):
    cfg = rule_manager.company_configs.get(symbol, {})
    name = cfg.get("company_name", symbol)
    keywords = rule_manager.get_keywords(symbol, avg_date) or [name]
    kw_map, kw_list = {}, []
    for k in keywords:
        if not isinstance(k, str) or len(k) <= 1:
            continue
        kl = k.lower()
        kw_map.setdefault(kl, []).append(symbol)
        kw_list.append(re.escape(kl))
    return kw_map, "|".join(kw_list), keywords


def validate_symbol(symbol, rule_manager, by_year):
    cfg = rule_manager.company_configs.get(symbol, {})
    company_name = cfg.get("company_name", symbol)
    company = {"symbol": symbol, "name": company_name, "cleaned_name": company_name}

    print(f"\n{'='*65}")
    print(f"  {symbol}  ({company_name})")
    print(f"{'='*65}")

    sample_files = _sample_files(by_year)
    avg_dt = _avg_date(sample_files)
    kw_map, pattern, keywords = _build_keyword_map(rule_manager, symbol, avg_dt)

    print(f"  Keywords ({len(keywords)}): {', '.join(keywords[:8])}{'...' if len(keywords)>8 else ''}")
    print(f"  Files: {len(sample_files)} ({FILES_PER_YEAR}/year)\n")

    if not pattern:
        print("  ⚠️  No keywords — skipping")
        return {"symbol": symbol, "kw_hits": 0, "confirmed": [], "status": "no_keywords"}

    # Phase 1: keyword regex + rule/SLM
    kw_hits = []
    sym_to_company = {symbol: company}
    for i, zp in enumerate(sample_files):
        res = process_file_task(zp, rule_manager, kw_map, pattern, sym_to_company, None, {})
        matches = res.get("matches", []) if res else []
        if matches:
            fname = os.path.basename(zp)
            print(f"  [{i+1:>3}/{len(sample_files)}] {fname}  → {len(matches)} hits")
            kw_hits.extend(matches)

    print(f"\n  Phase 1: {len(kw_hits)} keyword+SLM hits")

    if not kw_hits:
        print("  ⚪ No matches found across all sampled files")
        return {"symbol": symbol, "kw_hits": 0, "confirmed": [], "status": "zero_hits"}

    # Phase 2+3: URL fetch + content SLM (newest first — better URL survival rate)
    fetch_queue = sorted(kw_hits, key=lambda m: m["row"].get("Date", ""), reverse=True)
    confirmed, failed = [], 0

    def _fetch(m):
        return m, fetch_article(m["row"], m["company"])

    print(f"  Fetching {len(fetch_queue)} articles...")
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futures = {ex.submit(_fetch, m): m for m in fetch_queue}
        for fut in as_completed(futures):
            m, article = fut.result()
            if article:
                confirmed.append({
                    "title": article.get("title", ""),
                    "quality": article.get("data_quality", ""),
                    "content_len": article.get("content_length", 0),
                    "url": m["row"].get("URL", ""),
                    "date": m["row"].get("Date", ""),
                })
            else:
                failed += 1

    precision = len(confirmed) / max(len(fetch_queue), 1) * 100
    print(f"\n  Phase 2+3: {len(confirmed)} confirmed, {failed} failed ({precision:.0f}% precision)")
    print(f"\n  {'DATE':<10}  {'QUALITY':<10}  {'LEN':>5}  TITLE")
    print(f"  {'-'*60}")
    for a in sorted(confirmed, key=lambda x: x["date"]):
        title = a["title"][:45] or "(no title)"
        print(f"  {a['date']:<10}  {a['quality']:<10}  {a['content_len']:>5}  {title}")
        print(f"  {'':>32} {a['url'][:60]}")

    return {
        "symbol": symbol,
        "company_name": company_name,
        "keywords": keywords,
        "kw_hits": len(kw_hits),
        "fetch_attempted": len(fetch_queue),
        "confirmed": confirmed,
        "failed": failed,
        "precision_pct": round(precision, 1),
        "status": "done",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", nargs="?", default=None, help="Single symbol to validate")
    args = parser.parse_args()

    by_year = _all_zips_by_year()
    years = sorted(by_year.keys())
    print(f"📦 {sum(len(v) for v in by_year.values())} GKG files  |  years {years[0]}–{years[-1]}")

    rule_manager = RuleManager(RULES_DIR)
    all_results = {}

    symbols = [args.symbol.upper()] if args.symbol else TECH_26
    print(f"📋 Validating: {symbols}\n")

    for symbol in symbols:
        result = validate_symbol(symbol, rule_manager, by_year)
        all_results[symbol] = result

    # Summary table
    print(f"\n\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'SYM':<8} {'KW_HITS':>8} {'CONFIRMED':>10} {'PREC%':>7}  STATUS")
    print(f"  {'-'*55}")
    for sym, r in all_results.items():
        kw    = r.get("kw_hits", 0)
        conf  = len(r.get("confirmed", []))
        prec  = r.get("precision_pct", 0)
        stat  = r.get("status", "—")
        flag  = "⚪" if kw == 0 else ("✅" if conf > 0 else "❌")
        print(f"  {sym:<8} {kw:>8} {conf:>10} {prec:>7}%  {flag} {stat}")

    out = os.path.join(project_root, "audit_validation_latest.json")
    with open(out, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": all_results},
                  f, indent=4, ensure_ascii=False)
    print(f"\n💾 Saved to: {out}")


if __name__ == "__main__":
    main()
