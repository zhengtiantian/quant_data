"""M.12 step 1 — measure what news syndication does to the feature block.

Read-only. Writes nothing, changes no collection, touches no pipeline. The point is
to find out whether the distortion is large enough to justify changing anything.

The distortion: GDELT ingests each site of a syndication network as a separate
article. One McDonald's story on 2025-11-03 appears 402 times, byte-identical body,
identical sentiment, 402 distinct iheart.com subdomains -- so `unique_url_count`,
the field that exists to catch exactly this, counts 402 unique sources.

Both feature aggregators are row-weighted, so a story carried by 402 stations gets
402 votes:
  * aggregate_news_features   -> article_count is a raw row count
  * aggregate_llm_sentiment_features -> _weighted_mean sums over rows

Method: load the real frames, run the real aggregators twice -- once on raw input,
once on input deduplicated by (symbol, date, title) -- and diff. Reusing the
production functions rather than reimplementing them is the whole point; a
reimplementation would measure my copy, not the pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.features.daily_symbol_features import (  # noqa: E402
    DB_NAME,
    LLM_COLLECTION,
    NEWS_COLLECTION,
    aggregate_llm_sentiment_features,
    aggregate_news_features,
    create_client,
)

DEDUPE_KEY = ["symbol", "date", "title"]


def load_news_with_title() -> pd.DataFrame:
    """Same shape as the pipeline's load_news_frame(), plus `title`.

    `title` is the dedupe key and the production loader does not project it, which is
    part of why this was invisible: the field that identifies a duplicate never
    reached the dataframe.
    """
    col = create_client()[DB_NAME][NEWS_COLLECTION]
    cursor = col.find(
        {"symbol": {"$exists": True, "$ne": None}},
        {"_id": 0, "symbol": 1, "name": 1, "date": 1, "title": 1, "data_quality": 1,
         "content_length": 1, "content": 1, "url": 1, "note": 1, "source": 1},
    )
    rows = []
    for doc in cursor:
        raw = str(doc.get("date") or "")[:8]
        if len(raw) != 8 or not raw.isdigit():
            continue
        source = doc.get("source") or {}
        if isinstance(source, str):
            platform = name = source
        else:
            platform = source.get("platform") or "unknown"
            name = source.get("name") or platform
        note = doc.get("note") or ""
        rows.append({
            "symbol": doc.get("symbol"),
            "name": doc.get("name"),
            "date": pd.Timestamp(raw),
            "title": doc.get("title") or "",
            "data_quality": doc.get("data_quality") or "unknown",
            "content_length": int(doc.get("content_length") or len(doc.get("content") or "")),
            "url": doc.get("url"),
            "source_platform": platform,
            "source_name": name,
            "note": note,
            "is_extraction_failed": int("failed" in note.lower() or "unavailable" in note.lower()),
            "is_timeout_fallback": int("fetch_timeout" in note.lower()),
        })
    return pd.DataFrame(rows)


def load_llm_with_title() -> pd.DataFrame:
    col = create_client()[DB_NAME][LLM_COLLECTION]
    cursor = col.find(
        {"llm_sentiment_final": {"$exists": True}},
        {"_id": 0, "symbol": 1, "date": 1, "title": 1, "llm_sentiment_final": 1,
         "llm_signal_strength_a": 1, "llm_event_type_a": 1, "llm_disagreement": 1},
    )
    rows = []
    for doc in cursor:
        raw = str(doc.get("date") or "")[:8]
        if len(raw) != 8 or not raw.isdigit():
            continue
        rows.append({
            "symbol": doc.get("symbol"),
            "date": pd.Timestamp(raw),
            "title": doc.get("title") or "",
            "sentiment": float(doc.get("llm_sentiment_final", 0.0)),
            "strength": doc.get("llm_signal_strength_a", "low"),
            "event_type": doc.get("llm_event_type_a", "other"),
            "disagreement": float(doc.get("llm_disagreement", 0.0)),
        })
    return pd.DataFrame(rows)


def compare(raw: pd.DataFrame, ded: pd.DataFrame, cols: list[str], label: str) -> None:
    print(f"\n=== {label} ===")
    m = raw.merge(ded, on=["symbol", "date"], suffixes=("_raw", "_ded"), how="inner")
    print(f"symbol-days compared: {len(m)}")
    print(f"{'feature':<22}{'mean raw':>12}{'mean ded':>12}{'mean |diff|':>13}"
          f"{'% rows differ':>14}{'max |diff|':>12}")
    for c in cols:
        a, b = f"{c}_raw", f"{c}_ded"
        if a not in m or b not in m:
            continue
        d = (m[a] - m[b]).abs()
        differ = float((d > 1e-9).mean() * 100)
        print(f"  {c:<20}{m[a].mean():12.4f}{m[b].mean():12.4f}{d.mean():13.4f}"
              f"{differ:13.1f}%{d.max():12.4f}")


def main() -> None:
    print("loading news ...", flush=True)
    news = load_news_with_title()
    print(f"  {len(news)} rows")
    news_ded = news.drop_duplicates(subset=DEDUPE_KEY, keep="first")
    print(f"  {len(news_ded)} after dedupe "
          f"({100 * (1 - len(news_ded) / len(news)):.1f}% removed)")

    compare(
        aggregate_news_features(news),
        aggregate_news_features(news_ded),
        ["article_count", "unique_url_count", "unique_source_count",
         "news_count_5d", "news_count_20d", "news_burst_20d", "full_ratio"],
        "news volume features",
    )

    print("\nloading llm sentiment ...", flush=True)
    llm = load_llm_with_title()
    print(f"  {len(llm)} rows")
    llm_ded = llm.drop_duplicates(subset=DEDUPE_KEY, keep="first")
    print(f"  {len(llm_ded)} after dedupe "
          f"({100 * (1 - len(llm_ded) / len(llm)):.1f}% removed)")

    compare(
        aggregate_llm_sentiment_features(llm),
        aggregate_llm_sentiment_features(llm_ded),
        ["sent_wavg", "sent_std", "high_signal_n", "negative_n",
         "disagreement_mean", "sent_wavg_5d", "sent_wavg_20d"],
        "llm sentiment features",
    )

    # Per-symbol, because the corpus-average duplicate rate hides the real problem:
    # the distortion is uneven across symbols, so it biases any cross-sectional rank.
    print("\n=== per-symbol duplicate share (worst 15 with >2000 articles) ===")
    g = news.groupby("symbol").agg(total=("title", "size"))
    u = news_ded.groupby("symbol").agg(uniq=("title", "size"))
    s = g.join(u)
    s["dup_pct"] = 100 * (1 - s["uniq"] / s["total"])
    s = s[s["total"] > 2000].sort_values("dup_pct", ascending=False)
    print(s.head(15).to_string(float_format=lambda v: f"{v:.1f}"))
    print(f"\ncorpus-wide: {100 * (1 - len(news_ded) / len(news)):.1f}%  |  "
          f"spread across symbols: {s['dup_pct'].min():.1f}% to {s['dup_pct'].max():.1f}%")


if __name__ == "__main__":
    main()
