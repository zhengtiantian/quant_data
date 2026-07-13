"""
F.9 Rule Optimization Agent
Iterative loop: sample confirmed articles → LLM judge (TP/FP) → diagnose FP patterns
→ propose regex rules → write patches → repeat until convergence.

Usage:
  # Dry run: evaluate only, no changes applied
  python tools/rule_optimizer.py --symbols MS ISRG AAPL --dry-run

  # Full loop: evaluate + auto-apply low-risk patches
  python tools/rule_optimizer.py --symbols MS ISRG --max-rounds 10 --auto-apply

  # Use Claude as judge (requires ANTHROPIC_API_KEY)
  ANTHROPIC_API_KEY=sk-... python tools/rule_optimizer.py --judge claude --symbols MS

  # Resume from a specific round (skip re-judging; just re-propose based on last results)
  python tools/rule_optimizer.py --symbols MS --from-round 3
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient

sys.path.insert(0, os.path.dirname(__file__))
from llm_judge import LLMJudge

# ── Config ─────────────────────────────────────────────────────────────────
MONGO_URI      = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME        = "quant_data"
ARTICLE_COL    = "news_articles_company_matched_v2"
RUNS_COL       = "rule_optimization_runs"
PATCHES_FILE   = os.path.join(os.path.dirname(__file__), "rule_optimizer_patches.json")

DEFAULT_YEARS      = list(range(2016, 2026))   # 10 years
DEFAULT_N_PER_YEAR = 10                         # articles sampled per (symbol, year)
DEFAULT_MAX_ROUNDS = 10
CONVERGENCE_DELTA  = 0.01                       # stop when precision improves < 1%
MIN_FPS_TO_PROPOSE = 2                          # min FP examples needed to propose a pattern

# ── MongoDB helpers ─────────────────────────────────────────────────────────

def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return client[DB_NAME]


def get_company_name(db, symbol: str) -> str:
    doc = db["stock_universe"].find_one({"symbol": symbol}, {"name": 1})
    if doc and doc.get("name"):
        return doc["name"]
    # Fallback: query articles collection
    doc = db[ARTICLE_COL].find_one({"symbol": symbol}, {"company_name": 1})
    return (doc or {}).get("company_name", symbol)


def sample_articles(db, symbol: str, years: list[int], n_per_year: int) -> list[dict]:
    """Stratified sample: n_per_year articles per year, prefer data_quality='full'."""
    col = db[ARTICLE_COL]
    results = []
    for year in years:
        year_start = f"{year}-01-01"
        year_end   = f"{year + 1}-01-01"
        # Try full articles first
        pipeline = [
            {"$match": {
                "symbol": symbol,
                "data_quality": "full",
                "publishedAt": {"$gte": year_start, "$lt": year_end},
                "title":   {"$exists": True, "$ne": ""},
                "content": {"$exists": True, "$ne": ""},
            }},
            {"$sample": {"size": n_per_year}},
            {"$project": {"_id": 1, "title": 1, "content": 1, "publishedAt": 1,
                          "url": 1, "data_quality": 1}},
        ]
        batch = list(col.aggregate(pipeline))
        # If not enough full articles, top up with title_only
        if len(batch) < n_per_year:
            have = {str(d["_id"]) for d in batch}
            gap = n_per_year - len(batch)
            pipeline2 = [
                {"$match": {
                    "symbol": symbol,
                    "data_quality": {"$in": ["title_only", "url_only"]},
                    "publishedAt": {"$gte": year_start, "$lt": year_end},
                    "title": {"$exists": True, "$ne": ""},
                }},
                {"$sample": {"size": gap * 2}},
                {"$project": {"_id": 1, "title": 1, "content": 1, "publishedAt": 1,
                              "url": 1, "data_quality": 1}},
            ]
            extras = [d for d in col.aggregate(pipeline2) if str(d["_id"]) not in have]
            batch.extend(extras[:gap])
        for doc in batch:
            doc["year"] = year
            doc["_id"] = str(doc["_id"])
        results.extend(batch)
    return results


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(verdicts: list[dict]) -> dict:
    judged  = [v for v in verdicts if v.get("verdict") in ("TP", "FP")]
    tp      = sum(1 for v in judged if v["verdict"] == "TP")
    fp      = sum(1 for v in judged if v["verdict"] == "FP")
    total   = tp + fp
    precision = tp / total if total else 0.0
    fp_by_type = Counter(
        v.get("fp_type", "UNKNOWN") for v in judged if v["verdict"] == "FP"
    )
    fp_by_symbol = defaultdict(int)
    for v in judged:
        if v["verdict"] == "FP":
            fp_by_symbol[v.get("symbol", "?")] += 1
    return {
        "total_judged": total,
        "uncertain":    len(verdicts) - total,
        "tp": tp,
        "fp": fp,
        "precision": round(precision, 4),
        "fp_breakdown": dict(fp_by_type),
        "fp_by_symbol": dict(fp_by_symbol),
    }


# ── Patches file ─────────────────────────────────────────────────────────────

def load_patches() -> dict:
    if not os.path.exists(PATCHES_FILE):
        return {"version": 0, "STATIC_KILL_PATTERNS": {}, "CONTEXTUAL_REJECT_PATTERNS": {}}
    with open(PATCHES_FILE) as f:
        return json.load(f)


def save_patches(patches: dict):
    patches["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(PATCHES_FILE, "w") as f:
        json.dump(patches, f, indent=2, ensure_ascii=False)
    print(f"  💾 Patches written → {PATCHES_FILE}  (v{patches['version']})")


def apply_patch(change: dict, patches: dict, auto_apply_risk: Optional[str] = None) -> bool:
    """Merge one proposed change into patches dict. Returns True if applied."""
    risk = change.get("risk", "medium")
    if auto_apply_risk and risk not in _risk_levels_up_to(auto_apply_risk):
        return False
    target  = change.get("target_dict", "CONTEXTUAL_REJECT_PATTERNS")
    symbol  = change["symbol"]
    pattern = change.get("pattern", "")
    if not pattern:
        return False
    bucket = patches.setdefault(target, {})
    existing = bucket.setdefault(symbol, [])
    if pattern not in existing:
        existing.append(pattern)
    return True


def _risk_levels_up_to(max_risk: str) -> set[str]:
    order = ["low", "medium", "high"]
    idx = order.index(max_risk) if max_risk in order else 0
    return set(order[: idx + 1])


# ── MongoDB round storage ─────────────────────────────────────────────────────

def save_round(db, round_n: int, symbols: list[str], metrics: dict,
               verdicts: list[dict], proposed: list[dict], applied: list[dict],
               judge_model: str):
    doc = {
        "round":          round_n,
        "started_at":     datetime.now(timezone.utc).isoformat(),
        "symbols":        symbols,
        "judge_model":    judge_model,
        "metrics":        metrics,
        "proposed_changes": proposed,
        "applied_changes":  applied,
        "fp_samples": [
            {
                "symbol":   v.get("symbol"),
                "year":     v.get("year"),
                "title":    (v.get("title") or "")[:120],
                "fp_type":  v.get("fp_type"),
                "reason":   v.get("reason"),
                "confidence": v.get("confidence"),
            }
            for v in verdicts if v.get("verdict") == "FP"
        ][:50],
    }
    db[RUNS_COL].insert_one(doc)
    print(f"  📊 Round {round_n} saved to MongoDB ({RUNS_COL})")


def dump_round_json(round_n: int, metrics: dict, proposed: list[dict]):
    """Also write a human-readable JSON summary alongside the script."""
    out_path = os.path.join(
        os.path.dirname(__file__),
        f"audit_optimizer_round_{round_n:02d}.json",
    )
    with open(out_path, "w") as f:
        json.dump({"round": round_n, "metrics": metrics, "proposed": proposed},
                  f, indent=2, ensure_ascii=False)
    print(f"  📄 Round summary → {out_path}")


# ── Main optimization loop ────────────────────────────────────────────────────

def run(
    symbols: list[str],
    years: list[int],
    n_per_year: int,
    max_rounds: int,
    dry_run: bool,
    judge_model: str,
    auto_apply_risk: Optional[str],
):
    db    = get_db()
    judge = LLMJudge(model=judge_model)

    company_names = {s: get_company_name(db, s) for s in symbols}
    print(f"\n🎯 Symbols: {symbols}")
    print(f"📅 Years: {years[0]}–{years[-1]}  ({n_per_year}/year/symbol → "
          f"~{len(years) * n_per_year * len(symbols)} articles/round)")
    print(f"🔄 Max rounds: {max_rounds}  |  dry-run: {dry_run}  "
          f"|  auto-apply risk ≤ {auto_apply_risk or 'none'}\n")

    prev_precision: Optional[float] = None
    patches = load_patches()

    for round_n in range(1, max_rounds + 1):
        t0 = time.time()
        print(f"{'='*65}")
        print(f"  ROUND {round_n}")
        print(f"{'='*65}")

        # ── 1. Sample & judge ──────────────────────────────────────────────
        all_verdicts: list[dict] = []

        for symbol in symbols:
            company_name = company_names[symbol]
            articles = sample_articles(db, symbol, years, n_per_year)
            print(f"\n  [{symbol}] {company_name} — {len(articles)} articles sampled")
            judged = 0
            for art in articles:
                result = judge.evaluate(art, symbol, company_name)
                if result is None:
                    continue
                result["symbol"]  = symbol
                result["year"]    = art.get("year")
                result["title"]   = art.get("title", "")
                result["article_id"] = art["_id"]
                all_verdicts.append(result)
                judged += 1
                if judged % 20 == 0:
                    print(f"    … judged {judged}/{len(articles)}")

            sym_metrics = compute_metrics([v for v in all_verdicts if v.get("symbol") == symbol])
            print(f"    Precision: {sym_metrics['precision']:.1%}  "
                  f"TP={sym_metrics['tp']}  FP={sym_metrics['fp']}  "
                  f"FP types: {sym_metrics['fp_breakdown']}")

        # ── 2. Compute overall metrics ─────────────────────────────────────
        metrics = compute_metrics(all_verdicts)
        elapsed = round(time.time() - t0, 1)
        print(f"\n  📊 Round {round_n} overall:")
        print(f"     Precision = {metrics['precision']:.1%}  "
              f"(TP={metrics['tp']}, FP={metrics['fp']}, "
              f"uncertain={metrics['uncertain']})")
        print(f"     FP breakdown: {metrics['fp_breakdown']}")
        print(f"     Time: {elapsed}s")

        # ── 3. Propose rule changes for each (symbol, fp_type) group ──────
        proposed_changes: list[dict] = []
        fps = [v for v in all_verdicts if v.get("verdict") == "FP"]

        if fps and not dry_run:
            # group FPs by (symbol, fp_type)
            groups: dict[tuple, list] = defaultdict(list)
            for v in fps:
                key = (v.get("symbol", ""), v.get("fp_type", "PERIPHERY"))
                groups[key].append(v)

            print(f"\n  🔧 Proposing patterns for {len(groups)} FP groups ...")
            for (symbol, fp_type), group_fps in groups.items():
                if len(group_fps) < MIN_FPS_TO_PROPOSE:
                    continue
                company_name = company_names[symbol]
                change = judge.propose_pattern(group_fps, symbol, company_name, fp_type)
                if change and change.get("pattern"):
                    change["symbol"]  = symbol
                    change["fp_type"] = fp_type
                    change["n_fps"]   = len(group_fps)
                    proposed_changes.append(change)
                    risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                        change.get("risk", "medium"), "⚪"
                    )
                    print(f"    {risk_icon} [{symbol}/{fp_type}]  "
                          f"pattern: {change['pattern'][:60]}  "
                          f"→ {change.get('target_dict','?')}  "
                          f"risk={change.get('risk','?')}")

        # ── 4. Apply patches ───────────────────────────────────────────────
        applied_changes: list[dict] = []
        if proposed_changes and auto_apply_risk and not dry_run:
            for change in proposed_changes:
                if apply_patch(change, patches, auto_apply_risk):
                    applied_changes.append(change)
                    print(f"    ✅ Applied: [{change['symbol']}] {change.get('pattern','')[:60]}")
            if applied_changes:
                patches["version"] = patches.get("version", 0) + 1
                save_patches(patches)

        # ── 5. Persist round results ───────────────────────────────────────
        save_round(
            db, round_n, symbols, metrics, all_verdicts,
            proposed_changes, applied_changes,
            judge_model=("claude" if judge.backend == "claude" else LOCAL_JUDGE_MODEL),
        )
        dump_round_json(round_n, metrics, proposed_changes)

        # ── 6. Convergence check ───────────────────────────────────────────
        if prev_precision is not None:
            delta = abs(metrics["precision"] - prev_precision)
            print(f"\n  Δ precision vs last round: {delta:+.1%}")
            if delta < CONVERGENCE_DELTA and round_n > 1:
                print(f"\n🏁 Converged after {round_n} rounds "
                      f"(Δ={delta:.3f} < threshold {CONVERGENCE_DELTA}). Done.")
                break

        prev_precision = metrics["precision"]

        if round_n < max_rounds:
            print(f"\n  ⏳ Next round in 3s ...")
            time.sleep(3)

    print(f"\n{'='*65}")
    print("  OPTIMIZATION COMPLETE")
    print(f"  Final precision: {prev_precision:.1%}")
    print(f"  Patches file: {PATCHES_FILE}")
    print(f"{'='*65}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="F.9 Rule Optimization Agent — iterative FP diagnosis & rule proposal"
    )
    parser.add_argument(
        "--symbols", nargs="+", required=True,
        help="Symbols to evaluate, e.g. --symbols MS ISRG AAPL GS"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help="Years to sample (default: 2016-2025). E.g. --years 2020 2021 2022"
    )
    parser.add_argument(
        "--n-per-year", type=int, default=DEFAULT_N_PER_YEAR,
        help=f"Articles sampled per (symbol, year) (default {DEFAULT_N_PER_YEAR})"
    )
    parser.add_argument(
        "--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
        help=f"Maximum optimization rounds (default {DEFAULT_MAX_ROUNDS})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate only; do not propose or apply rule changes"
    )
    parser.add_argument(
        "--judge", choices=["auto", "claude", "local"], default="auto",
        help="Judge model: auto=Claude if API key set else local SLM (default: auto)"
    )
    parser.add_argument(
        "--auto-apply", choices=["low", "medium", "high"], default=None,
        metavar="RISK",
        help="Auto-apply proposed patches with risk ≤ RISK. "
             "Omit to only write proposals without applying."
    )
    parser.add_argument(
        "--show-patches", action="store_true",
        help="Print current patches file and exit"
    )
    args = parser.parse_args()

    if args.show_patches:
        patches = load_patches()
        print(json.dumps(patches, indent=2, ensure_ascii=False))
        return

    symbols = [s.upper() for s in args.symbols]
    years   = args.years or DEFAULT_YEARS

    run(
        symbols=symbols,
        years=years,
        n_per_year=args.n_per_year,
        max_rounds=args.max_rounds,
        dry_run=args.dry_run,
        judge_model=args.judge,
        auto_apply_risk=args.auto_apply,
    )


if __name__ == "__main__":
    main()
