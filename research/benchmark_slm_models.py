#!/usr/bin/env python3
"""
Benchmark two LM Studio models on the company-relevance YES/NO task.

Usage:
    python research/benchmark_slm_models.py \
        --model-a qwen3.5-4b \
        --model-b gemma-4-e2b \
        --samples 200

What this measures:
    - mean latency per article (seconds)
    - p50 / p95 latency
    - agreement rate vs existing Qwen scores (treated as ground truth)
    - YES / NO distribution
    - tokens-per-second estimate

Ground truth:
    Articles that already have slm_relevance_v1 scores from the current
    production Qwen model are used as reference. Both new models are
    compared against those scores to measure agreement rate.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil
import requests
from dotenv import load_dotenv
from pymongo import MongoClient

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GLOBAL_ENV = ROOT / ".env"
load_dotenv(GLOBAL_ENV, override=False)

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = os.getenv("FEATURE_DB_NAME", "quant_data")
NEWS_COLLECTION = os.getenv("FEATURE_NEWS_COLLECTION", "news_articles")
RESCORE_FIELD = os.getenv("SLM_RESCORE_FIELD", "slm_relevance_v1")
API_URL = os.getenv("SLM_API_URL", os.getenv("LMSTUDIO_API", "http://127.0.0.1:1234/v1"))

def _find_lmstudio_pids() -> list[int]:
    """Find LM Studio and its inference backend processes."""
    targets = {"lm studio", "lmstudio", "llamafile", "llama-server", "llama_server", "llamacpp"}
    pids = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info["name"] or "").lower()
            cmdline = " ".join(proc.info["cmdline"] or []).lower()
            if any(t in name or t in cmdline for t in targets):
                pids.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return pids


class MemoryMonitor:
    """
    Sample RSS of LM Studio process(es) in a background thread.
    Reports peak and mean unified memory usage in MB.
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            pids = _find_lmstudio_pids()
            total_mb = 0.0
            for pid in pids:
                try:
                    p = psutil.Process(pid)
                    total_mb += p.memory_info().rss / (1024 ** 2)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if total_mb > 0:
                self._samples.append(total_mb)
            time.sleep(self.interval)

    @property
    def peak_mb(self) -> float | None:
        return max(self._samples) if self._samples else None

    @property
    def mean_mb(self) -> float | None:
        return statistics.mean(self._samples) if self._samples else None

    @property
    def n_samples(self) -> int:
        return len(self._samples)


PROMPT_TEMPLATE = """\
Does the following news article actually discuss {company_name} ({symbol}) as a primary subject?

Title: {title}
Content: {content}

Reply with only YES or NO.
"""


def build_prompt(symbol: str, company_name: str, title: str, content: str) -> str:
    return PROMPT_TEMPLATE.format(
        symbol=symbol,
        company_name=company_name,
        title=title,
        content=content[:800].strip(),
    )


def call_model(api_url: str, model: str, prompt: str, use_no_think: bool) -> tuple[str, float]:
    """
    Call the model and return (answer, elapsed_seconds).
    answer is one of: 'YES', 'NO', 'ERROR'
    """
    content = f"/no_think\n{prompt}" if use_no_think else prompt
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8,
        "temperature": 0,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{api_url}/chat/completions",
            json=payload,
            timeout=60,
        )
        elapsed = time.perf_counter() - t0
        if resp.status_code != 200:
            return "ERROR", elapsed
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return "ERROR", elapsed
        raw = (choices[0].get("message", {}).get("content") or "").strip().upper()
        # Extract YES or NO from response
        if raw.startswith("YES"):
            return "YES", elapsed
        if raw.startswith("NO"):
            return "NO", elapsed
        # Try to find YES/NO anywhere in short response
        if "YES" in raw:
            return "YES", elapsed
        if "NO" in raw:
            return "NO", elapsed
        return "ERROR", elapsed
    except Exception:
        elapsed = time.perf_counter() - t0
        return "ERROR", elapsed


def load_test_articles(n: int) -> list[dict]:
    """
    Load articles that already have slm_relevance_v1 scores.
    Mix YES and NO roughly 50/50.
    """
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    col = client[DB_NAME][NEWS_COLLECTION]

    half = n // 2
    yes_docs = list(
        col.find(
            {
                "symbol": {"$exists": True, "$ne": None},
                "title": {"$exists": True, "$ne": ""},
                f"{RESCORE_FIELD}.relevant": True,
            },
            {"_id": 1, "symbol": 1, "name": 1, "title": 1, "content": 1, RESCORE_FIELD: 1},
        ).limit(half)
    )
    no_docs = list(
        col.find(
            {
                "symbol": {"$exists": True, "$ne": None},
                "title": {"$exists": True, "$ne": ""},
                f"{RESCORE_FIELD}.relevant": False,
            },
            {"_id": 1, "symbol": 1, "name": 1, "title": 1, "content": 1, RESCORE_FIELD: 1},
        ).limit(half)
    )

    docs = yes_docs + no_docs
    if not docs:
        raise RuntimeError(
            f"No articles with {RESCORE_FIELD} found. "
            "Run slm_relevance_rescore.py first to create ground-truth labels."
        )

    print(f"Loaded {len(yes_docs)} YES + {len(no_docs)} NO articles as ground truth")
    return docs


def benchmark_model(
    model: str,
    articles: list[dict],
    api_url: str,
    use_no_think: bool,
) -> dict:
    latencies: list[float] = []
    results: list[str] = []
    ground_truth: list[bool] = []
    mem_monitor = MemoryMonitor(interval=0.3)

    print(f"\n{'='*60}")
    print(f"Model: {model}  (no_think={'ON' if use_no_think else 'OFF'})")
    print(f"{'='*60}")

    lm_pids = _find_lmstudio_pids()
    if lm_pids:
        print(f"LM Studio process(es) found: PIDs {lm_pids}")
    else:
        print("⚠️  LM Studio process not found — memory stats will be unavailable")

    mem_monitor.start()

    for i, doc in enumerate(articles, 1):
        symbol = doc.get("symbol", "")
        name = doc.get("name") or symbol
        title = doc.get("title") or ""
        content = doc.get("content") or ""
        gt_relevant = bool(doc.get(RESCORE_FIELD, {}).get("relevant", False))

        prompt = build_prompt(symbol, name, title, content)
        answer, elapsed = call_model(api_url, model, prompt, use_no_think)

        latencies.append(elapsed)
        results.append(answer)
        ground_truth.append(gt_relevant)

        if i % 20 == 0 or i == len(articles):
            mem_peak = mem_monitor.peak_mb
            mem_str = f"  mem_peak={mem_peak:.0f}MB" if mem_peak else ""
            print(
                f"  [{i:>3}/{len(articles)}]  "
                f"last={elapsed:.2f}s  "
                f"mean={statistics.mean(latencies):.2f}s  "
                f"errors={results.count('ERROR')}"
                f"{mem_str}"
            )

    mem_monitor.stop()

    # Compute metrics
    valid = [(r, g) for r, g in zip(results, ground_truth) if r != "ERROR"]
    n_valid = len(valid)
    n_error = results.count("ERROR")

    agreement = sum(1 for r, g in valid if (r == "YES") == g) / n_valid if n_valid else 0.0
    yes_rate = sum(1 for r, _ in valid if r == "YES") / n_valid if n_valid else 0.0

    sorted_lat = sorted(latencies)
    p50 = sorted_lat[int(len(sorted_lat) * 0.50)]
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)]

    return {
        "model": model,
        "n_total": len(articles),
        "n_valid": n_valid,
        "n_error": n_error,
        "mean_latency_s": statistics.mean(latencies),
        "p50_latency_s": p50,
        "p95_latency_s": p95,
        "total_time_s": sum(latencies),
        "agreement_rate": agreement,
        "yes_rate": yes_rate,
        "mem_peak_mb": mem_monitor.peak_mb,
        "mem_mean_mb": mem_monitor.mean_mb,
        "mem_samples": mem_monitor.n_samples,
    }


def print_comparison(a: dict, b: dict) -> None:
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    def _mem(val: float | None) -> str:
        return f"{val:.0f} MB" if val is not None else "n/a"

    def _tag_lower(va: str, vb: str) -> tuple[str, str]:
        """Mark the lower (better) value for latency/memory rows."""
        try:
            fa = float(va.split()[0])
            fb = float(vb.split()[0])
            if fa < fb:
                return va + " ←", vb
            if fb < fa:
                return va, vb + " ←"
        except ValueError:
            pass
        return va, vb

    def _tag_higher(va: str, vb: str) -> tuple[str, str]:
        """Mark the higher (better) value for accuracy rows."""
        try:
            fa = float(va.rstrip("%"))
            fb = float(vb.rstrip("%"))
            if fa > fb:
                return va + " ←", vb
            if fb > fa:
                return va, vb + " ←"
        except ValueError:
            pass
        return va, vb

    rows: list[tuple[str, str, str]] = [
        ("articles tested",          f"{a['n_total']}",                     f"{b['n_total']}"),
        ("errors",                   f"{a['n_error']}",                     f"{b['n_error']}"),
        ("mean latency",             f"{a['mean_latency_s']:.3f}s",         f"{b['mean_latency_s']:.3f}s"),
        ("p50 latency",              f"{a['p50_latency_s']:.3f}s",          f"{b['p50_latency_s']:.3f}s"),
        ("p95 latency",              f"{a['p95_latency_s']:.3f}s",          f"{b['p95_latency_s']:.3f}s"),
        ("total time",               f"{a['total_time_s']:.1f}s",           f"{b['total_time_s']:.1f}s"),
        ("peak memory (RSS)",        _mem(a["mem_peak_mb"]),                _mem(b["mem_peak_mb"])),
        ("mean memory (RSS)",        _mem(a["mem_mean_mb"]),                _mem(b["mem_mean_mb"])),
        ("agreement vs ground truth",f"{a['agreement_rate']:.1%}",         f"{b['agreement_rate']:.1%}"),
        ("YES rate",                 f"{a['yes_rate']:.1%}",                f"{b['yes_rate']:.1%}"),
    ]

    col_w = 30
    print(f"{'metric':<{col_w}} {'Model A':>18} {'Model B':>18}")
    print("-" * (col_w + 38))
    for label, va, vb in rows:
        if label in ("mean latency", "p50 latency", "p95 latency", "total time",
                     "peak memory (RSS)", "mean memory (RSS)"):
            va, vb = _tag_lower(va, vb)
        elif label == "agreement vs ground truth":
            va, vb = _tag_higher(va, vb)
        print(f"{label:<{col_w}} {va:>18} {vb:>18}")

    print()
    print(f"Model A: {a['model']}")
    print(f"Model B: {b['model']}")

    # Speed-up ratio
    if b["mean_latency_s"] > 0:
        ratio = b["mean_latency_s"] / a["mean_latency_s"]
        faster = "A" if ratio > 1 else "B"
        ratio = max(ratio, 1 / ratio)
        print(f"\nSpeed: Model {faster} is {ratio:.1f}x faster per article")

    # Agreement delta
    delta = abs(a["agreement_rate"] - b["agreement_rate"])
    if delta < 0.02:
        print(f"Accuracy: models agree within {delta:.1%} — essentially equal")
    else:
        better = "A" if a["agreement_rate"] > b["agreement_rate"] else "B"
        print(f"Accuracy: Model {better} agrees {delta:.1%} more with ground truth")

    print()
    print("Recommendation:")
    speed_winner = "A" if a["mean_latency_s"] < b["mean_latency_s"] else "B"
    acc_winner = "A" if a["agreement_rate"] > b["agreement_rate"] else "B"
    acc_delta = abs(a["agreement_rate"] - b["agreement_rate"])

    if speed_winner == acc_winner:
        print(f"  Model {speed_winner} ({a['model'] if speed_winner == 'A' else b['model']})")
        print(f"  is both faster AND more accurate. Switch to it.")
    elif acc_delta < 0.03:
        print(f"  Accuracy difference is small ({acc_delta:.1%}).")
        print(f"  Model {speed_winner} ({a['model'] if speed_winner == 'A' else b['model']})")
        print(f"  is faster. Switch if resource pressure is the main concern.")
    else:
        print(f"  Accuracy difference is significant ({acc_delta:.1%}).")
        print(f"  Stick with the more accurate model unless you are severely")
        print(f"  resource-constrained.")


RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"


def save_result(result: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    safe_name = result["model"].replace("/", "_").replace(" ", "_")
    out = RESULTS_DIR / f"{safe_name}.json"
    import json
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    return out


def load_result(path: Path) -> dict:
    import json
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    # Single-model mode (run one at a time, save result)
    parser.add_argument("--model", default=None,
                        help="Run a single model and save result to benchmark_results/")
    parser.add_argument("--no-think", action="store_true", default=False,
                        help="Use /no_think prefix (set this for Qwen3 models)")

    # Compare mode (load two saved results and print comparison)
    parser.add_argument("--compare", nargs=2, metavar="RESULT_FILE",
                        help="Compare two saved result JSON files")

    # Shared options
    parser.add_argument("--samples", type=int, default=100,
                        help="Total articles to test, split 50/50 YES/NO (default: 100)")
    parser.add_argument("--api-url", default=API_URL,
                        help=f"LM Studio API base URL (default: {API_URL})")

    args = parser.parse_args()

    # ── Compare mode ──────────────────────────────────────────────
    if args.compare:
        a = load_result(Path(args.compare[0]))
        b = load_result(Path(args.compare[1]))
        print_comparison(a, b)
        return

    # ── Single-model mode ─────────────────────────────────────────
    if not args.model:
        parser.print_help()
        print("\nExamples:")
        print("  # Step 1 — load qwen3.5-4b in LM Studio, then run:")
        print("  python research/benchmark_slm_models.py --model qwen3.5-4b --no-think --samples 100")
        print()
        print("  # Step 2 — switch to gemma-4-e2b in LM Studio, then run:")
        print("  python research/benchmark_slm_models.py --model gemma-4-e2b --samples 100")
        print()
        print("  # Step 3 — compare the two saved results:")
        print("  python research/benchmark_slm_models.py \\")
        print("    --compare benchmark_results/qwen3.5-4b.json benchmark_results/gemma-4-e2b.json")
        sys.exit(0)

    # Verify API is reachable
    try:
        resp = requests.get(f"{args.api_url}/models", timeout=5)
        available = [m["id"] for m in resp.json().get("data", [])]
        print(f"LM Studio connected — models loaded: {available}")
    except Exception as e:
        print(f"Cannot reach LM Studio at {args.api_url}: {e}")
        print("→ Open LM Studio → Developer tab → Start Server")
        sys.exit(1)

    print(f"\nBenchmarking: {args.model}")
    print(f"Samples: {args.samples}   no_think: {args.no_think}")

    articles = load_test_articles(args.samples)

    result = benchmark_model(
        model=args.model,
        articles=articles,
        api_url=args.api_url,
        use_no_think=args.no_think,
    )

    out_path = save_result(result)
    print(f"\nResult saved → {out_path}")
    print(f"  mean latency : {result['mean_latency_s']:.3f}s")
    print(f"  p95 latency  : {result['p95_latency_s']:.3f}s")
    print(f"  agreement    : {result['agreement_rate']:.1%}")
    if result["mem_peak_mb"]:
        print(f"  peak memory  : {result['mem_peak_mb']:.0f} MB")
    print(f"\nRun the other model, then compare with --compare")


if __name__ == "__main__":
    main()
