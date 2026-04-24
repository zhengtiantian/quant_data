#!/usr/bin/env python3
"""
Compare Qwen3.5-2B-4bit vs Qwen3.5-4B on the company relevance task.

Uses production SLMFilter + CompanyMatchSkill prompt.
Ground truth: existing decisions in news_articles_company_matched_v2
  (matched=True → YES, articles that were rejected are sampled from news_articles).

Usage:
  # Step 1: load 4B model in LM Studio, run:
  SLM_API_URL=http://192.168.31.226:1234/v1 \
  .venv/bin/python3 research/benchmark_2b_vs_4b.py --model qwen3.5-4b --samples 200

  # Step 2: load 2B model in LM Studio, run:
  SLM_API_URL=http://192.168.31.226:1234/v1 \
  .venv/bin/python3 research/benchmark_2b_vs_4b.py --model qwen3.5-2b-4bit --samples 200

  # Step 3: compare:
  .venv/bin/python3 research/benchmark_2b_vs_4b.py \
    --compare benchmark_results/qwen3.5-4b.json benchmark_results/qwen3.5-2b-4bit.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from pymongo import MongoClient

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from news_collectors.gdelt.special_rules.slm_skills import CompanyMatchSkill, SkillContext

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"
API_URL = os.getenv("SLM_API_URL", "http://192.168.31.226:1234/v1")
RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"

_skill = CompanyMatchSkill()

COMPANY_UNIVERSE: dict[str, tuple[str, list[str]]] = {
    "AAPL":  ("Apple Inc.",            ["Apple", "iPhone", "iPad", "MacBook", "Tim Cook"]),
    "GOOGL": ("Alphabet Inc.",         ["Google", "Alphabet", "YouTube", "Android", "DeepMind"]),
    "MSFT":  ("Microsoft Corporation", ["Microsoft", "Windows", "Azure", "Xbox", "Satya Nadella"]),
    "TSLA":  ("Tesla, Inc.",           ["Tesla", "Elon Musk", "Model 3", "Model S", "Model Y", "Cybertruck"]),
    "AMZN":  ("Amazon.com Inc.",       ["Amazon", "AWS", "Jeff Bezos", "Andy Jassy"]),
    "NVDA":  ("NVIDIA Corporation",    ["NVIDIA", "GeForce", "Jensen Huang", "CUDA", "H100", "A100"]),
    "META":  ("Meta Platforms",        ["Meta Platforms", "Facebook", "Instagram", "WhatsApp", "Zuckerberg"]),
    "INTC":  ("Intel Corporation",     ["Intel Corporation", "Intel Corp", "Xeon processor", "Intel chip"]),
    "QCOM":  ("Qualcomm Inc.",         ["Qualcomm", "Snapdragon"]),
    "AMD":   ("Advanced Micro Devices",["Advanced Micro Devices", "Ryzen", "EPYC", "Radeon GPU"]),
    "ARM":   ("ARM Holdings",          ["ARM Holdings", "Arm Ltd", "Arm architecture"]),
    "AVGO":  ("Broadcom Inc.",         ["Broadcom"]),
    "MU":    ("Micron Technology",     ["Micron Technology", "Micron Memory"]),
    "DDOG":  ("Datadog Inc.",          ["Datadog"]),
    "TSM":   ("Taiwan Semiconductor",  ["TSMC", "Taiwan Semiconductor", "Morris Chang"]),
    "ASML":  ("ASML Holding",          ["ASML", "EUV lithography"]),
    "AMAT":  ("Applied Materials",     ["Applied Materials"]),
    "LRCX":  ("Lam Research",          ["Lam Research"]),
    "KLAC":  ("KLA Corporation",       ["KLA Corporation", "KLA Corp"]),
    "TXN":   ("Texas Instruments",     ["Texas Instruments"]),
    "ADI":   ("Analog Devices",        ["Analog Devices"]),
    "MCHP":  ("Microchip Technology",  ["Microchip Technology"]),
    "CRM":   ("Salesforce",            ["Salesforce", "Marc Benioff"]),
    "NOW":   ("ServiceNow",            ["ServiceNow"]),
    "ADBE":  ("Adobe Inc.",            ["Adobe", "Photoshop", "Creative Cloud"]),
    "ORCL":  ("Oracle Corporation",    ["Oracle Corporation", "Larry Ellison"]),
    "PLTR":  ("Palantir Technologies", ["Palantir"]),
    "SNOW":  ("Snowflake Inc.",        ["Snowflake Inc", "Snowflake data"]),
    "MDB":   ("MongoDB Inc.",          ["MongoDB"]),
    "PANW":  ("Palo Alto Networks",    ["Palo Alto Networks"]),
    "FTNT":  ("Fortinet",              ["Fortinet"]),
    "CRWD":  ("CrowdStrike",           ["CrowdStrike"]),
    "NFLX":  ("Netflix",               ["Netflix", "Reed Hastings"]),
    "UBER":  ("Uber Technologies",     ["Uber"]),
    "ABNB":  ("Airbnb Inc.",           ["Airbnb", "Brian Chesky"]),
    "CSCO":  ("Cisco Systems",         ["Cisco Systems", "Cisco Corp"]),
    "IBM":   ("IBM",                   ["IBM", "International Business Machines"]),
    "DELL":  ("Dell Technologies",     ["Dell Technologies", "Dell Computer"]),
    "SMCI":  ("Super Micro Computer",  ["Supermicro", "Super Micro Computer"]),
    "INTU":  ("Intuit Inc.",           ["Intuit", "TurboTax", "QuickBooks"]),
}


def load_raw_articles(n: int) -> list[dict]:
    """
    Load n random articles from news_articles (no ground truth labels).
    Covers all 40 symbols proportionally.
    """
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    src_col = db["news_articles"]

    all_syms = list(COMPANY_UNIVERSE.keys())
    per_sym = max(1, n // len(all_syms))

    docs = []
    for sym in all_syms:
        batch = list(
            src_col.find(
                {"symbol": sym, "title": {"$exists": True, "$ne": ""},
                 "content": {"$exists": True, "$ne": ""}},
                {"_id": 1, "symbol": 1, "name": 1, "title": 1, "content": 1},
            ).limit(per_sym * 3)
        )
        random.shuffle(batch)
        docs.extend(batch[:per_sym])

    random.shuffle(docs)
    docs = docs[:n]
    print(f"Loaded {len(docs)} articles across {len(set(d['symbol'] for d in docs))} symbols")
    return docs


def call_model(model: str, symbol: str, company_name: str,
               title: str, content: str, api_url: str) -> tuple[str, float]:
    """Call model with production prompt. Returns (YES|NO|ERROR, elapsed_s)."""
    ctx = SkillContext(
        symbol=symbol,
        company_name=company_name,
        title=title,
        content=content,
    )
    prompt = _skill.build_prompt(ctx)

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{api_url}/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 6,
                "temperature": 0,
            },
            timeout=60,
        )
        elapsed = time.perf_counter() - t0
        if resp.status_code != 200:
            return "ERROR", elapsed
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return "ERROR", elapsed
        raw = (choices[0].get("text") or "").strip().upper()
        if "YES" in raw:
            return "YES", elapsed
        if "NO" in raw:
            return "NO", elapsed
        return "ERROR", elapsed
    except Exception:
        return "ERROR", time.perf_counter() - t0


def run_benchmark(model: str, articles: list[dict], api_url: str) -> dict:
    """Run model on articles, return per-article answers + latency stats."""
    latencies: list[float] = []
    answers: list[str] = []

    print(f"\n{'='*60}")
    print(f"Model: {model}  ({len(articles)} articles)")
    print("="*60)

    for i, doc in enumerate(articles, 1):
        symbol = (doc.get("symbol") or "").strip().upper()
        company_name = COMPANY_UNIVERSE.get(symbol, (symbol, []))[0]
        title = doc.get("title") or ""
        content = (doc.get("content") or "")[:1500]

        ans, elapsed = call_model(model, symbol, company_name, title, content, api_url)
        latencies.append(elapsed)
        answers.append(ans)

        if i % 50 == 0 or i == len(articles):
            yes_so_far = answers.count("YES")
            print(
                f"  [{i:>3}/{len(articles)}]  "
                f"last={elapsed:.2f}s  mean={statistics.mean(latencies):.2f}s  "
                f"yes={yes_so_far}  no={answers.count('NO')}  err={answers.count('ERROR')}"
            )

    n_error = answers.count("ERROR")
    n_valid = len(answers) - n_error
    yes_rate = answers.count("YES") / n_valid if n_valid else 0.0
    sorted_lat = sorted(latencies)
    return {
        "model": model,
        "n_total": len(articles),
        "n_valid": n_valid,
        "n_error": n_error,
        "answers": answers,
        "mean_latency_s": statistics.mean(latencies),
        "p50_latency_s": sorted_lat[int(len(sorted_lat) * 0.50)],
        "p95_latency_s": sorted_lat[int(len(sorted_lat) * 0.95)],
        "total_time_s": sum(latencies),
        "yes_rate": yes_rate,
    }


def run_both_parallel(
    model_a: str, model_b: str, articles: list[dict], api_url: str
) -> tuple[dict, dict]:
    """Run both models concurrently on the SAME article list."""
    import threading
    results: dict[str, dict] = {}

    def _run(model: str) -> None:
        results[model] = run_benchmark(model, articles, api_url)

    t_a = threading.Thread(target=_run, args=(model_a,), daemon=True)
    t_b = threading.Thread(target=_run, args=(model_b,), daemon=True)
    t_a.start(); t_b.start()
    t_a.join(); t_b.join()
    return results[model_a], results[model_b]


def print_comparison(a: dict, b: dict, articles: list[dict] | None = None) -> None:
    print("\n" + "="*62)
    print(f"COMPARISON:  A={a['model']}  vs  B={b['model']}")
    print("="*62)

    ans_a = a.get("answers", [])
    ans_b = b.get("answers", [])

    # Agreement analysis (only where both gave valid answers)
    both_yes = both_no = a_yes_b_no = a_no_b_yes = 0
    disagreements = []
    for i, (aa, ab) in enumerate(zip(ans_a, ans_b)):
        if aa == "ERROR" or ab == "ERROR":
            continue
        if aa == "YES" and ab == "YES":
            both_yes += 1
        elif aa == "NO" and ab == "NO":
            both_no += 1
        elif aa == "YES" and ab == "NO":
            a_yes_b_no += 1
            if articles and len(disagreements) < 5:
                doc = articles[i]
                disagreements.append(("A=YES,B=NO", doc.get("symbol",""), doc.get("title","")[:70]))
        elif aa == "NO" and ab == "YES":
            a_no_b_yes += 1
            if articles and len(disagreements) < 5:
                doc = articles[i]
                disagreements.append(("A=NO,B=YES", doc.get("symbol",""), doc.get("title","")[:70]))

    n_compared = both_yes + both_no + a_yes_b_no + a_no_b_yes
    agree_rate = (both_yes + both_no) / n_compared if n_compared else 0.0

    def tl(va: str, vb: str) -> tuple[str, str]:
        try:
            fa, fb = float(va.split()[0]), float(vb.split()[0])
            if fa < fb: return va + " ✓", vb
            if fb < fa: return va, vb + " ✓"
        except ValueError: pass
        return va, vb

    w = 16
    rows = [
        ("articles tested",  f"{a['n_total']}",              f"{b['n_total']}"),
        ("errors",           f"{a['n_error']}",              f"{b['n_error']}"),
        ("mean latency",     f"{a['mean_latency_s']:.3f}s",  f"{b['mean_latency_s']:.3f}s"),
        ("p50 latency",      f"{a['p50_latency_s']:.3f}s",   f"{b['p50_latency_s']:.3f}s"),
        ("p95 latency",      f"{a['p95_latency_s']:.3f}s",   f"{b['p95_latency_s']:.3f}s"),
        ("YES rate",         f"{a['yes_rate']:.1%}",         f"{b['yes_rate']:.1%}"),
    ]

    print(f"{'metric':<22} {'Model A':>{w}} {'Model B':>{w}}")
    print("-" * (22 + w*2 + 2))
    for label, va, vb in rows:
        if "latency" in label or "errors" in label:
            va, vb = tl(va, vb)
        print(f"{label:<22} {va:>{w}} {vb:>{w}}")

    print(f"\n--- Agreement on {n_compared} articles ---")
    print(f"  Both YES   : {both_yes:>4}  ({100*both_yes/n_compared:.1f}%)")
    print(f"  Both NO    : {both_no:>4}  ({100*both_no/n_compared:.1f}%)")
    print(f"  A=YES B=NO : {a_yes_b_no:>4}  ({100*a_yes_b_no/n_compared:.1f}%)  ← A more permissive")
    print(f"  A=NO  B=YES: {a_no_b_yes:>4}  ({100*a_no_b_yes/n_compared:.1f}%)  ← B more permissive")
    print(f"  Agreement  : {agree_rate:.1%}")

    if disagreements:
        print("\n--- Sample disagreements ---")
        for label, sym, title in disagreements:
            print(f"  [{label}] {sym}: {title}")

    print()
    speed_ratio = b["mean_latency_s"] / a["mean_latency_s"] if a["mean_latency_s"] > 0 else 1
    faster = "A" if speed_ratio > 1 else "B"
    ratio = max(speed_ratio, 1 / speed_ratio)
    print(f"Speed: Model {faster} ({(a if faster=='A' else b)['model']}) is {ratio:.1f}x faster")
    yes_delta = a["yes_rate"] - b["yes_rate"]
    if abs(yes_delta) < 0.03:
        print("YES rate: essentially the same")
    else:
        more_perm = "A" if yes_delta > 0 else "B"
        print(f"YES rate: Model {more_perm} is more permissive (+{abs(yes_delta):.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-a", default=None, help="First model name (e.g. qwen3.5-4b)")
    parser.add_argument("--model-b", default=None, help="Second model name (e.g. qwen3.5-2b-4bit)")
    parser.add_argument("--model", default=None, help="Single-model mode: benchmark one model")
    parser.add_argument("--samples", type=int, default=200, help="Total articles 50%% YES + 50%% NO")
    parser.add_argument("--api-url", default=API_URL, help=f"LM Studio base URL (default: {API_URL})")
    parser.add_argument("--compare", nargs=2, metavar="JSON", help="Compare two saved result JSON files")
    args = parser.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        print_comparison(a, b)
        return

    # Check API reachable
    try:
        resp = requests.get(f"{args.api_url}/models", timeout=5)
        available = [m["id"] for m in resp.json().get("data", [])]
        print(f"LM Studio connected — models loaded: {available}")
    except Exception as e:
        print(f"Cannot reach LM Studio at {args.api_url}: {e}")
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    def _save(result: dict) -> Path:
        safe = result["model"].replace("/", "_").replace(" ", "_").replace(":", "-")
        out = RESULTS_DIR / f"{safe}.json"
        out.write_text(json.dumps(result, indent=2))
        return out

    articles = load_raw_articles(args.samples)

    if args.model_a and args.model_b:
        print(f"\nParallel benchmark: {args.model_a}  vs  {args.model_b}")
        res_a, res_b = run_both_parallel(args.model_a, args.model_b, articles, args.api_url)
        _save(res_a)
        _save(res_b)
        print_comparison(res_a, res_b, articles)

    elif args.model or args.model_a:
        model = args.model or args.model_a
        result = run_benchmark(model, articles, args.api_url)
        out = _save(result)
        print(f"\nResult saved → {out}")
        print(f"  mean latency : {result['mean_latency_s']:.3f}s")
        print(f"  YES rate     : {result['yes_rate']:.1%}")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Both models loaded simultaneously:")
        print("  SLM_API_URL=http://192.168.31.226:1234/v1 \\")
        print("    .venv/bin/python3 research/benchmark_2b_vs_4b.py \\")
        print("    --model-a qwen3.5-4b --model-b qwen3.5-2b-4bit --samples 200")
        sys.exit(0)


if __name__ == "__main__":
    main()
