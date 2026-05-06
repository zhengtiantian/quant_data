#!/usr/bin/env python3
"""Daily signal scoring — computes composite scores for each symbol and writes
top-N signals to MongoDB `daily_signals` collection for Kafka publishing.

Reads from:   quant_data.daily_symbol_features  (latest trade_date)
Writes to:    quant_data.daily_signals           (upsert by date+symbol)

Usage:
  LOCAL_MONGO_URI="mongodb://root:root@127.0.0.1:37018/" \
  .venv311/bin/python research/score_daily_signals.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"
FEATURE_COLLECTION = os.getenv("FEATURE_OUTPUT_COLLECTION", "daily_symbol_features")
SIGNAL_COLLECTION = "daily_signals"
TOP_N = int(os.getenv("SIGNAL_TOP_N", "10"))
LOOKBACK_DAYS = int(os.getenv("SIGNAL_LOOKBACK_DAYS", "1"))

_WEIGHTS = {
    "quality_score":        1.0,
    "news_burst_20d":       0.8,
    "full_ratio":           0.5,
    "avg_sentiment_5d":     1.5,
    "sentiment_shift_5d":   0.8,
    "earnings_beat_signal": 1.2,
    "high_signal_count_3d": 0.6,
    "past_ret_20d":        -0.3,  # mild reversal
}


def load_latest_features(col) -> pd.DataFrame:
    latest_doc = col.find_one(sort=[("trade_date", -1)])
    if not latest_doc:
        return pd.DataFrame()
    latest_date = latest_doc["trade_date"]
    docs = list(col.find({"trade_date": latest_date}))
    return pd.DataFrame(docs)


def compute_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score = pd.Series(0.0, index=df.index)
    for feat, w in _WEIGHTS.items():
        if feat not in df.columns:
            continue
        col = pd.to_numeric(df[feat], errors="coerce").fillna(0.0)
        # cross-sectional rank normalised to [-0.5, 0.5]
        ranked = col.rank(pct=True) - 0.5
        score += w * ranked
    df["composite_score"] = score
    df["signal_rank"] = score.rank(ascending=False).astype(int)
    return df


def build_signal_docs(df: pd.DataFrame, top_n: int) -> list[dict]:
    df_sorted = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    docs = []
    for i, row in df_sorted.iterrows():
        rank = int(row["signal_rank"])
        signal_type = "LONG" if rank <= top_n else "NEUTRAL"
        doc = {
            "trade_date": row["trade_date"],
            "symbol": row["symbol"],
            "composite_score": round(float(row["composite_score"]), 6),
            "signal_rank": rank,
            "signal_type": signal_type,
            "top_n": top_n,
            # key features for context
            "avg_sentiment_5d": _safe_float(row.get("avg_sentiment_5d")),
            "sentiment_shift_5d": _safe_float(row.get("sentiment_shift_5d")),
            "earnings_beat_signal": _safe_int(row.get("earnings_beat_signal")),
            "earnings_miss_signal": _safe_int(row.get("earnings_miss_signal")),
            "news_burst_20d": _safe_float(row.get("news_burst_20d")),
            "quality_score": _safe_float(row.get("quality_score")),
            "published_at": datetime.now(timezone.utc),
        }
        docs.append(doc)
    return docs


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> int:
    try:
        f = float(v)
        return 0 if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return 0


def upsert_signals(col, docs: list[dict]) -> None:
    ops = [
        UpdateOne(
            {"trade_date": d["trade_date"], "symbol": d["symbol"]},
            {"$set": d},
            upsert=True,
        )
        for d in docs
    ]
    if ops:
        result = col.bulk_write(ops)
        print(f"Upserted {result.upserted_count} / modified {result.modified_count} signals")


def main() -> None:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    feature_col = db[FEATURE_COLLECTION]
    signal_col = db[SIGNAL_COLLECTION]

    signal_col.create_index([("trade_date", 1), ("symbol", 1)], unique=True, background=True)
    signal_col.create_index([("trade_date", -1), ("signal_rank", 1)], background=True)

    print(f"Loading latest features from {FEATURE_COLLECTION}...")
    df = load_latest_features(feature_col)
    if df.empty:
        print("No features found.")
        return

    trade_date = df["trade_date"].iloc[0]
    trade_date_str = trade_date if isinstance(trade_date, str) else str(trade_date)[:10]
    print(f"trade_date={trade_date_str}  symbols={len(df)}")

    df = compute_score(df)
    docs = build_signal_docs(df, TOP_N)
    upsert_signals(signal_col, docs)

    top = [d for d in docs if d["signal_type"] == "LONG"]
    print(f"\nTop {TOP_N} signals for {trade_date_str}:")
    for d in top:
        print(f"  #{d['signal_rank']:>2} {d['symbol']:<6}  score={d['composite_score']:+.4f}"
              f"  sent={d['avg_sentiment_5d']}  beat={d['earnings_beat_signal']}")


if __name__ == "__main__":
    main()
