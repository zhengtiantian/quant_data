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

project_root = "/Users/xiz/Quant_trade/quant_data"
collector_dir = os.path.join(project_root, "news_collectors/gdelt")
for path in [project_root, collector_dir]:
    if path not in sys.path:
        sys.path.append(path)

os.environ["USE_SLM_FILTER"] = "true"

try:
    import historical_collector as _hc
    from historical_collector import process_file_task, fetch_article, RuleManager
    from special_rules.ambiguous_names import print_filter_summary
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

BASE_DIR       = "/Volumes/Data24T/docker-volumes/gdelt_cache/files"
BASE_DIR2      = "/Volumes/Data6T/gdelt_cache/files"   # odd batches
RULES_DIR      = os.path.join(project_root, "news_collectors/gdelt/company_rules")
FILES_PER_YEAR = 30    # files sampled per calendar year (10yr × 30 = ~300 total)
FETCH_WORKERS  = 6
MAX_FETCH      = 10   # Max article body fetches per symbol

TECH_26 = [
    "TSM", "ASML", "CRM", "PLTR", "NOW", "ADBE", "NFLX", "UBER",
    "SNOW", "MDB", "PANW", "CRWD", "SMCI", "AMAT", "LRCX", "KLAC",
    "TXN", "ADI", "MCHP", "ORCL", "FTNT", "ABNB", "CSCO", "IBM",
    "DELL", "INTU",
]

NEW_60 = [
    "SHOP", "NET", "ZS", "HUBS", "WDAY", "VEEV", "TEAM", "TTD", "OKTA", "APP",
    "RBLX", "COIN", "TWLO", "DUOL", "CFLT", "GTLB", "MNDY", "S",
    "LLY", "JNJ", "AMGN", "GILD", "REGN", "VRTX", "ISRG", "UNH", "MRNA",
    "ABBV", "PFE", "MDT", "SYK", "DXCM", "ILMN",
    "V", "MA", "PYPL", "GS", "JPM", "MS", "BLK", "SCHW", "AXP", "COF",
    "DIS", "SNAP", "SPOT", "RDDT", "PINS",
    "NKE", "HD", "SBUX", "MCD", "TGT",
    "CAT", "HON", "RTX", "LMT", "GE", "DE", "BA",
]


def _register_files(csv_paths):
    """Register only the sampled CSV paths into _hc._filename_to_batch.

    process_file_task does os.path.basename(url) then calls _file_cached(basename),
    which needs _filename_to_batch to resolve the batch dir. We populate just the
    ~100 sampled files instead of all 188K, so this is instant.
    """
    for full_path in csv_paths:
        fname    = os.path.basename(full_path)
        batch_id = int(os.path.basename(os.path.dirname(full_path)))
        _hc._filename_to_batch[fname] = batch_id


def _all_zips_by_year():
    """Quickly discover batch dirs; use only the first filename per batch for year detection.

    Returns {year: [batch_dir_path, ...]} — individual files are listed on demand
    inside _sample_files so we never scan all 188K files upfront.
    """
    by_year = {}
    for base in [BASE_DIR, BASE_DIR2]:
        if not os.path.exists(base):
            continue
        try:
            entries = list(os.scandir(base))
        except PermissionError as e:
            print(f"  ⚠️  Skipping {base}: {e}")
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            # Peek at first CSV to learn the year — fast, no full listing
            try:
                first = next(
                    f for f in os.listdir(entry.path) if f.endswith(".gkg.csv")
                )
            except StopIteration:
                continue
            y = first[:4]
            if y.isdigit() and 2015 <= int(y) <= datetime.now().year + 1:
                by_year.setdefault(y, []).append(entry.path)
    return by_year


def _sample_files(by_year, per_year=FILES_PER_YEAR):
    """Sample `per_year` CSV files per year. by_year maps year → [batch_dir_paths]."""
    files = []
    for y in sorted(by_year):
        batch_dirs = by_year[y]
        sampled_dirs = random.sample(batch_dirs, min(per_year, len(batch_dirs)))
        for d in sampled_dirs:
            # Pick one random CSV from each sampled batch dir
            csvs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".gkg.csv")]
            if csvs:
                files.append(random.choice(csvs))
    return files


def _avg_date(csv_paths):
    dates = []
    for z in csv_paths:
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


def validate_symbol(symbol, rule_manager, by_year, files_per_year=FILES_PER_YEAR):
    import ahocorasick
    cfg = rule_manager.company_configs.get(symbol, {})
    company_name = cfg.get("name", symbol)
    company = {"symbol": symbol, "name": company_name, "cleaned_name": company_name}

    print(f"\n{'='*65}")
    print(f"  {symbol}  ({company_name})")
    print(f"{'='*65}")

    sample_files = _sample_files(by_year, per_year=files_per_year)
    _register_files(sample_files)   # inject batch_id so _file_cached works
    avg_dt = _avg_date(sample_files)
    kw_map, _, keywords = _build_keyword_map(rule_manager, symbol, avg_dt)

    print(f"  Keywords ({len(keywords)}): {', '.join(keywords[:8])}{'...' if len(keywords)>8 else ''}")
    print(f"  Files: {len(sample_files)} ({files_per_year}/year)\n")

    if not kw_map:
        print("  ⚠️  No keywords — skipping")
        return {"symbol": symbol, "kw_hits": 0, "confirmed": [], "status": "no_keywords"}

    # Build Aho-Corasick automaton (current process_file_task signature)
    ac = ahocorasick.Automaton()
    for kl in kw_map:
        ac.add_word(kl, kl)
    ac.make_automaton()

    # Phase 1: keyword AC match → rule/SLM filter
    kw_hits = []
    sym_to_company = {symbol: company}
    for i, zp in enumerate(sample_files):
        res = process_file_task(zp, rule_manager, ac, kw_map, sym_to_company, None, {})
        matches = res.get("matches", []) if res else []
        if matches:
            fname = os.path.basename(zp)
            print(f"  [{i+1:>3}/{len(sample_files)}] {fname}  → {len(matches)} hits")
            kw_hits.extend(matches)

    print_filter_summary()
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
    parser.add_argument("symbol", nargs="*", default=None, help="One or more symbols to validate")
    parser.add_argument("--new", action="store_true", help="Validate all 60 new symbols")
    parser.add_argument("--tech", action="store_true", help="Validate original TECH_26")
    parser.add_argument("--files-per-year", type=int, default=FILES_PER_YEAR,
                        help=f"CSV files sampled per calendar year (default {FILES_PER_YEAR})")
    parser.add_argument("--start-year", type=int, default=None,
                        help="Only sample files from this year onwards (e.g. 2018)")
    args = parser.parse_args()

    files_per_year = args.files_per_year

    print("🔍 Discovering batch dirs by year...")
    by_year = _all_zips_by_year()
    if args.start_year:
        by_year = {y: v for y, v in by_year.items() if int(y) >= args.start_year}
        print(f"⚙️  start-year filter: {args.start_year}+")
    years = sorted(by_year.keys())
    total_batches = sum(len(v) for v in by_year.values())
    print(f"📦 {total_batches} batch dirs  |  years {years[0]}–{years[-1]}")
    if files_per_year != FILES_PER_YEAR:
        print(f"⚙️  files-per-year overridden to {files_per_year}")

    rule_manager = RuleManager(RULES_DIR)
    all_results = {}

    if args.symbol:
        symbols = [s.upper() for s in args.symbol]
    elif args.new:
        symbols = NEW_60
    elif args.tech:
        symbols = TECH_26
    else:
        symbols = NEW_60   # default to new symbols
    print(f"📋 Validating {len(symbols)} symbols: {symbols[:6]}{'...' if len(symbols)>6 else ''}\n")

    for symbol in symbols:
        result = validate_symbol(symbol, rule_manager, by_year, files_per_year=files_per_year)
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
