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

# ---------------------------------------------------------------------------
# H.2 Dynamic regime weights
# Each regime gets its own weight dict; classify_regime() picks which to use.
# Features are still rank-normalised to [-0.5, +0.5] before weighting.
# ---------------------------------------------------------------------------

# RISK_ON: VIX calm (pctile < 0.3), SPY above 200MA → amplify short-term signals
_WEIGHTS_RISK_ON = {
    "quality_score":            1.0,
    "news_burst_20d":           0.9,
    "full_ratio":               0.5,
    "avg_sentiment_5d":         2.0,   # sentiment reliable in calm markets
    "sentiment_shift_5d":       1.0,
    "earnings_beat_signal":     1.5,   # earnings surprises rewarded more
    "high_signal_count_3d":     0.8,
    "past_ret_20d":            -0.2,
    "ah_gap":                   1.6,   # pre/after-hours gap CS IC5d=+0.23
    "analyst_buy_ratio_chg_1m": 1.0,
    "analyst_buy_ratio":        0.8,
    "inst_holding_pct_chg":     0.5,
    "retail_sent_score":        0.4,
}

# NEUTRAL: VIX moderate, market trending — current default calibration
_WEIGHTS_NEUTRAL = {
    "quality_score":            1.0,
    "news_burst_20d":           0.8,
    "full_ratio":               0.5,
    "avg_sentiment_5d":         1.5,
    "sentiment_shift_5d":       0.8,
    "earnings_beat_signal":     1.2,
    "high_signal_count_3d":     0.6,
    "past_ret_20d":            -0.3,
    "ah_gap":                   1.2,
    "analyst_buy_ratio_chg_1m": 0.9,
    "analyst_buy_ratio":        0.7,
    "inst_holding_pct_chg":     0.6,
    "retail_sent_score":        0.3,
}

# STRESSED: VIX elevated (pctile 0.5-0.8) or risk_on=0 → shift toward slow/smart signals
_WEIGHTS_STRESSED = {
    "quality_score":            0.8,
    "news_burst_20d":           0.5,
    "full_ratio":               0.4,
    "avg_sentiment_5d":         0.8,   # LLM sentiment noisier under stress
    "sentiment_shift_5d":       0.4,
    "earnings_beat_signal":     1.0,
    "high_signal_count_3d":     0.3,
    "past_ret_20d":             0.0,   # no momentum in choppy market
    "ah_gap":                   0.6,
    "analyst_buy_ratio_chg_1m": 0.7,
    "analyst_buy_ratio":        0.6,
    "inst_holding_pct_chg":     1.0,   # smart-money signal more reliable
    "retail_sent_score":        0.1,
}

# RISK_OFF: VIX extreme (pctile >= 0.8) / crisis — reduce all, follow institutions
_WEIGHTS_RISK_OFF = {
    "quality_score":            0.4,
    "news_burst_20d":           0.2,
    "full_ratio":               0.2,
    "avg_sentiment_5d":         0.3,   # news sentiment is noise in crisis
    "sentiment_shift_5d":       0.2,
    "earnings_beat_signal":     0.7,
    "high_signal_count_3d":     0.1,
    "past_ret_20d":             0.2,   # slight mean-reversion in bear market
    "ah_gap":                   0.3,
    "analyst_buy_ratio_chg_1m": 0.5,
    "analyst_buy_ratio":        0.4,
    "inst_holding_pct_chg":     1.2,   # follow smart money; highest weight
    "retail_sent_score":       -0.2,   # retail is contrarian signal in crisis
}

_WEIGHTS_BY_REGIME: dict[str, dict[str, float]] = {
    "RISK_ON":  _WEIGHTS_RISK_ON,
    "NEUTRAL":  _WEIGHTS_NEUTRAL,
    "STRESSED": _WEIGHTS_STRESSED,
    "RISK_OFF": _WEIGHTS_RISK_OFF,
}

# Overall conviction scalar per regime (scales composite score magnitude;
# does NOT change cross-sectional ranking — only affects downstream sizing)
_REGIME_CONVICTION: dict[str, float] = {
    "RISK_ON":  1.20,
    "NEUTRAL":  1.00,
    "STRESSED": 0.75,
    "RISK_OFF": 0.50,
}


def load_latest_features(col) -> pd.DataFrame:
    latest_doc = col.find_one(sort=[("trade_date", -1)])
    if not latest_doc:
        return pd.DataFrame()
    latest_date = latest_doc["trade_date"]
    docs = list(col.find({"trade_date": latest_date}))
    return pd.DataFrame(docs)


def classify_regime(df: pd.DataFrame) -> str:
    """Classify today's market regime from macro features (same value for all symbols)."""
    row = df.iloc[0]

    def _get(field):
        v = row.get(field)
        try:
            f = float(v)
            return None if np.isnan(f) else f
        except (TypeError, ValueError):
            return None

    vix_pct = _get("macro_vix_pctile_252d")
    risk_on = _get("macro_risk_on")

    if vix_pct is None:
        return "NEUTRAL"
    if vix_pct >= 0.8:
        return "RISK_OFF"
    if vix_pct >= 0.5 or risk_on == 0:
        return "STRESSED"
    if risk_on == 1 and vix_pct < 0.3:
        return "RISK_ON"
    return "NEUTRAL"


def compute_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    regime = classify_regime(df)
    weights = _WEIGHTS_BY_REGIME[regime]
    conviction = _REGIME_CONVICTION[regime]

    score = pd.Series(0.0, index=df.index)
    for feat, w in weights.items():
        if feat not in df.columns:
            continue
        col = pd.to_numeric(df[feat], errors="coerce").fillna(0.0)
        ranked = col.rank(pct=True) - 0.5   # cross-sectional, [-0.5, +0.5]
        score += w * ranked

    score *= conviction

    df["composite_score"] = score
    df["signal_rank"] = score.rank(ascending=False).astype(int)
    df["regime_label"] = regime
    df["regime_mult"] = conviction
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
            "regime_label": str(row.get("regime_label", "NEUTRAL")),
            "regime_mult": _safe_float(row.get("regime_mult")),
            # existing context fields
            "avg_sentiment_5d": _safe_float(row.get("avg_sentiment_5d")),
            "sentiment_shift_5d": _safe_float(row.get("sentiment_shift_5d")),
            "earnings_beat_signal": _safe_int(row.get("earnings_beat_signal")),
            "earnings_miss_signal": _safe_int(row.get("earnings_miss_signal")),
            "news_burst_20d": _safe_float(row.get("news_burst_20d")),
            "quality_score": _safe_float(row.get("quality_score")),
            # D-series context fields
            "ah_gap": _safe_float(row.get("ah_gap")),
            "analyst_buy_ratio": _safe_float(row.get("analyst_buy_ratio")),
            "analyst_buy_ratio_chg_1m": _safe_float(row.get("analyst_buy_ratio_chg_1m")),
            "inst_holding_pct_chg": _safe_float(row.get("inst_holding_pct_chg")),
            "retail_sent_score": _safe_float(row.get("retail_sent_score")),
            "macro_risk_on": _safe_int(row.get("macro_risk_on")),
            "macro_vix": _safe_float(row.get("macro_vix")),
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
    regime_label = docs[0].get("regime_label", "NEUTRAL") if docs else "NEUTRAL"
    regime_mult = docs[0].get("regime_mult", 1.0) if docs else 1.0
    print(f"\nTop {TOP_N} signals for {trade_date_str}  "
          f"[regime={regime_label}  conviction={regime_mult:.2f}]:")
    print(f"  {'#':>2} {'sym':<6}  {'score':>7}  {'sent5d':>6}  {'ah_gap':>6}  "
          f"{'ana_chg':>7}  {'inst_chg':>8}  {'beat':>4}")
    for d in top:
        print(f"  #{d['signal_rank']:>2} {d['symbol']:<6}  {d['composite_score']:>+7.4f}"
              f"  {str(d.get('avg_sentiment_5d') or ''):>6}"
              f"  {str(d.get('ah_gap') or ''):>6}"
              f"  {str(d.get('analyst_buy_ratio_chg_1m') or ''):>7}"
              f"  {str(d.get('inst_holding_pct_chg') or ''):>8}"
              f"  {d.get('earnings_beat_signal', 0):>4}")


if __name__ == "__main__":
    main()
