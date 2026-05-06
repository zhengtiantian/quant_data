#!/usr/bin/env python3
"""Event-driven backtest framework.

Simulates dynamic holding periods where positions are exited based on
signal conditions rather than a fixed number of days.

Entry: composite score > entry_threshold on a given day
Hold check (daily):
  - score drops below exit_threshold  → exit immediately
  - held >= min_hold days AND score <= hold_threshold → exit
  - held >= max_hold days → forced exit
Exit: whichever condition is met first

Compares against a fixed-horizon baseline using the same entry signal.

Usage:
  python research/backtest_event_driven.py --collection daily_symbol_features_company_matched_v2
  python research/backtest_event_driven.py --max-hold 30 --min-hold 5 --top-n 5
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from scipy.stats import spearmanr

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
load_dotenv(ROOT / ".env", override=False)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Missing MONGO_URI")
LOCAL_MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")

FORWARD_HORIZONS = [5, 10, 15, 20, 30, 45, 60]


def _fallback_mongo_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.hostname in {"mongo6", "mongodb", "mongo"}:
        fallback = urlparse(LOCAL_MONGO_URI)
        return urlunparse((
            parsed.scheme or fallback.scheme,
            fallback.netloc,
            parsed.path or fallback.path,
            parsed.params, parsed.query, parsed.fragment,
        ))
    return uri


def create_client() -> MongoClient:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except Exception:
        fallback = _fallback_mongo_uri(MONGO_URI)
        client = MongoClient(fallback, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client


def load_feature_frame(collection: str, db: str = "quant_data", min_date: str = "2015-12-04") -> pd.DataFrame:
    client = create_client()
    coll = client[db][collection]
    projection = {
        "_id": 0, "symbol": 1, "sector": 1, "trade_date": 1, "date": 1,
        "full_ratio": 1, "quality_score": 1, "article_count": 1,
        "news_burst_20d": 1, "unique_source_count": 1, "avg_content_length": 1,
        "past_ret_20d": 1, "past_ret_60d": 1, "volatility_20d": 1,
        "surprise_pct_last": 1, "is_positive_surprise": 1, "is_negative_surprise": 1,
        "surprise_bucket": 1, "earnings_recency_weight": 1,
        "full_ratio_x_post_positive": 1, "quality_score_x_post_negative": 1,
        "days_to_earnings": 1, "days_since_earnings": 1,
        # LLM sentiment features (Stage 3.5.2)
        "avg_sentiment_3d": 1, "avg_sentiment_5d": 1, "sentiment_shift_5d": 1,
        "high_signal_count_3d": 1, "has_regulatory_risk_5d": 1,
        "earnings_beat_signal": 1, "earnings_miss_signal": 1,
        "disagreement_avg_5d": 1, "negative_event_count_5d": 1,
    }
    for h in FORWARD_HORIZONS:
        projection[f"future_ret_{h}d"] = 1
        projection[f"excess_ret_{h}d"] = 1

    rows = list(coll.find({"trade_date": {"$gte": min_date}}, projection))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank-based composite score per trade_date."""
    df = df.copy()
    components = {
        "full_ratio": 1.0,
        "quality_score": 1.0,
        "news_burst_20d": 0.5,
        "earnings_recency_weight": 0.5,
        # LLM sentiment
        "avg_sentiment_5d": 1.5,
        "sentiment_shift_5d": 0.8,
        # LLM event quality
        "earnings_beat_signal": 1.2,
        "high_signal_count_3d": 0.6,
    }
    df["composite_score"] = 0.0
    df["composite_count"] = 0

    for dt, idx in df.groupby("trade_date").groups.items():
        group = df.loc[idx]
        score = pd.Series(0.0, index=group.index)
        count = pd.Series(0, index=group.index)
        for col, weight in components.items():
            if col not in group.columns:
                continue
            vals = pd.to_numeric(group[col], errors="coerce")
            valid = vals.notna()
            if valid.sum() < 2:
                continue
            ranked = vals[valid].rank(pct=True) - 0.5
            score.loc[valid.index] += ranked * weight
            count.loc[valid.index] += 1
        valid_count = count > 0
        score.loc[valid_count] /= count.loc[valid_count]
        score.loc[~valid_count] = pd.NA
        df.loc[idx, "composite_score"] = score
        df.loc[idx, "composite_count"] = count

    return df


@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_score: float
    days_held: int = 0
    exit_date: pd.Timestamp | None = None
    exit_reason: str = ""
    actual_return: float | None = None
    excess_return: float | None = None


@dataclass
class EventDrivenBacktestResult:
    total_trades: int
    avg_hold_days: float
    mean_excess_ret: float
    hit_rate: float
    exit_reasons: dict[str, int]
    yearly: dict[int, dict]


def run_event_driven_backtest(
    df: pd.DataFrame,
    entry_threshold: float = 0.15,
    exit_threshold: float = -0.10,
    hold_threshold: float = 0.0,
    min_hold: int = 5,
    max_hold: int = 60,
    top_n: int = 5,
    sentiment_exit_threshold: float = -0.20,
    sentiment_shift_exit: float = -0.20,
    sentiment_entry_min: float = 0.0,
    block_miss_entry: bool = False,
) -> EventDrivenBacktestResult:
    """
    Simulate event-driven positions.

    entry_threshold:          composite_score percentile (0-1) to enter.
    exit_threshold:           if score drops below this percentile → forced exit.
    hold_threshold:           after min_hold days, exit if score <= this percentile.
    min_hold:                 minimum days before signal-based exit is allowed.
    max_hold:                 maximum days to hold regardless of signal.
    top_n:                    max simultaneous positions.
    sentiment_exit_threshold: avg_sentiment_5d below this + regulatory risk → immediate exit.
    sentiment_shift_exit:     sentiment_shift_5d below this after min_hold → exit.
    """
    dates = sorted(df["trade_date"].unique())
    date_to_df: dict[pd.Timestamp, pd.DataFrame] = {
        dt: df[df["trade_date"] == dt].drop_duplicates("symbol").set_index("symbol")
        for dt in dates
    }

    open_positions: list[Position] = []
    closed_positions: list[Position] = []

    for dt in dates:
        day_df = date_to_df[dt]

        # --- Update open positions ---
        still_open = []
        for pos in open_positions:
            pos.days_held += 1
            row_now = day_df.loc[pos.symbol] if pos.symbol in day_df.index else None
            score_now = row_now["composite_score"] if row_now is not None else pd.NA

            # LLM event signals
            has_reg_risk   = bool(row_now["has_regulatory_risk_5d"]) if row_now is not None and pd.notna(row_now.get("has_regulatory_risk_5d")) else False
            sent_5d        = float(row_now["avg_sentiment_5d"])       if row_now is not None and pd.notna(row_now.get("avg_sentiment_5d")) else None
            sent_shift     = float(row_now["sentiment_shift_5d"])     if row_now is not None and pd.notna(row_now.get("sentiment_shift_5d")) else None
            earnings_miss  = bool(row_now["earnings_miss_signal"])    if row_now is not None and pd.notna(row_now.get("earnings_miss_signal")) else False
            earnings_beat  = bool(row_now["earnings_beat_signal"])    if row_now is not None and pd.notna(row_now.get("earnings_beat_signal")) else False

            # earnings_beat 锁仓：持有期内有正向收益信号，屏蔽情感退出，强持到 min_hold
            beat_lock = earnings_beat and pos.days_held < min_hold

            exit_reason = None
            if pos.days_held >= max_hold:
                exit_reason = f"max_hold_{max_hold}d"
            elif pd.notna(score_now) and score_now < exit_threshold:
                exit_reason = "score_below_exit"
            # LLM: regulatory risk + negative sentiment → immediate exit (beat_lock 不保护)
            elif not earnings_beat and has_reg_risk and sent_5d is not None and sent_5d < sentiment_exit_threshold:
                exit_reason = "regulatory_risk_negative_sent"
            # LLM: earnings miss → immediate exit (beat_lock 不保护)
            elif not earnings_beat and earnings_miss and sent_5d is not None and sent_5d < sentiment_exit_threshold:
                exit_reason = "earnings_miss_negative_sent"
            # LLM: 情感动量逆转 — beat_lock 期间屏蔽
            elif not beat_lock and pos.days_held >= min_hold and sent_shift is not None and sent_shift < sentiment_shift_exit:
                exit_reason = "sentiment_reversal"
            elif not beat_lock and pos.days_held >= min_hold and pd.notna(score_now) and score_now <= hold_threshold:
                exit_reason = "score_weak_after_min_hold"

            if exit_reason:
                pos.exit_date = dt
                pos.exit_reason = exit_reason
                # Look up actual return at days_held horizon (closest available)
                available = [h for h in FORWARD_HORIZONS if h <= pos.days_held]
                horizon = max(available) if available else min(FORWARD_HORIZONS)
                entry_day_df = date_to_df.get(pos.entry_date)
                if entry_day_df is not None and pos.symbol in entry_day_df.index:
                    row = entry_day_df.loc[pos.symbol]
                    pos.actual_return = row.get(f"future_ret_{horizon}d")
                    pos.excess_return = row.get(f"excess_ret_{horizon}d")
                closed_positions.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # --- Open new positions ---
        if len(open_positions) < top_n:
            already_held = {p.symbol for p in open_positions}
            eligible = day_df[~day_df.index.isin(already_held)].copy()
            eligible = eligible[pd.to_numeric(eligible["composite_score"], errors="coerce").notna()]
            eligible = eligible.sort_values("composite_score", ascending=False)

            slots = top_n - len(open_positions)
            for symbol, row in eligible.head(slots * 3).iterrows():
                score = row["composite_score"]
                if pd.isna(score) or score < entry_threshold:
                    continue
                # sentiment_entry_min 门控：要求 avg_sentiment_5d 高于最低阈值
                if sentiment_entry_min > 0:
                    sent = row.get("avg_sentiment_5d")
                    if pd.isna(sent) or float(sent) < sentiment_entry_min:
                        continue
                # 屏蔽 earnings_miss：近期有负向收益信号时不开新仓
                if block_miss_entry and row.get("earnings_miss_signal", 0):
                    continue
                open_positions.append(Position(
                    symbol=symbol,
                    entry_date=dt,
                    entry_score=float(score),
                ))
                if len(open_positions) >= top_n:
                    break

    # Force-close remaining open positions at max_hold
    for pos in open_positions:
        pos.exit_date = dates[-1]
        pos.exit_reason = "end_of_data"
        available = [h for h in FORWARD_HORIZONS if h <= max(pos.days_held, 1)]
        horizon = max(available) if available else min(FORWARD_HORIZONS)
        entry_day_df = date_to_df.get(pos.entry_date)
        if entry_day_df is not None and pos.symbol in entry_day_df.index:
            row = entry_day_df.loc[pos.symbol]
            pos.actual_return = row.get(f"future_ret_{horizon}d")
            pos.excess_return = row.get(f"excess_ret_{horizon}d")
        closed_positions.append(pos)

    if not closed_positions:
        return EventDrivenBacktestResult(0, 0, 0, 0, {}, {})

    excess_rets = [p.excess_return for p in closed_positions if p.excess_return is not None and pd.notna(p.excess_return)]
    hold_days = [p.days_held for p in closed_positions]
    exit_reasons: dict[str, int] = {}
    for p in closed_positions:
        exit_reasons[p.exit_reason] = exit_reasons.get(p.exit_reason, 0) + 1

    # Yearly breakdown
    yearly: dict[int, dict] = {}
    for p in closed_positions:
        yr = p.entry_date.year
        if yr not in yearly:
            yearly[yr] = {"trades": 0, "excess_rets": [], "hold_days": []}
        yearly[yr]["trades"] += 1
        if p.excess_return is not None and pd.notna(p.excess_return):
            yearly[yr]["excess_rets"].append(p.excess_return)
        yearly[yr]["hold_days"].append(p.days_held)

    return EventDrivenBacktestResult(
        total_trades=len(closed_positions),
        avg_hold_days=float(np.mean(hold_days)),
        mean_excess_ret=float(np.mean(excess_rets)) if excess_rets else 0.0,
        hit_rate=float(np.mean([r > 0 for r in excess_rets])) if excess_rets else 0.0,
        exit_reasons=exit_reasons,
        yearly=yearly,
    )


def run_fixed_baseline(df: pd.DataFrame, horizon: int, top_n: int = 5) -> dict:
    """Fixed-horizon baseline using composite_score for comparison."""
    dates = sorted(df["trade_date"].unique())
    excess_col = f"excess_ret_{horizon}d"
    if excess_col not in df.columns:
        return {}

    daily_rets = []
    for dt in dates:
        day = df[df["trade_date"] == dt].copy()
        day = day[pd.to_numeric(day["composite_score"], errors="coerce").notna()]
        day = day[day[excess_col].notna()]
        if len(day) < top_n:
            continue
        top = day.nlargest(top_n, "composite_score")
        daily_rets.append(top[excess_col].mean())

    if not daily_rets:
        return {}
    return {
        "mean_excess_ret": float(np.mean(daily_rets)),
        "hit_rate": float(np.mean([r > 0 for r in daily_rets])),
        "observations": len(daily_rets),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", default="daily_symbol_features")
    p.add_argument("--db", default="quant_data")
    p.add_argument("--min-date", default="2018-01-01")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--min-hold", type=int, default=5)
    p.add_argument("--max-hold", type=int, default=60)
    p.add_argument("--entry-threshold", type=float, default=0.15,
                   help="Min composite_score to open a position")
    p.add_argument("--exit-threshold", type=float, default=-0.10,
                   help="Score below this triggers immediate exit")
    p.add_argument("--hold-threshold", type=float, default=0.0,
                   help="After min_hold days, exit if score <= this")
    p.add_argument("--sentiment-exit", type=float, default=-0.20,
                   help="avg_sentiment_5d threshold for event-driven exit")
    p.add_argument("--sentiment-shift-exit", type=float, default=-0.20,
                   help="sentiment_shift_5d threshold for momentum reversal exit")
    p.add_argument("--sentiment-entry-min", type=float, default=0.0,
                   help="Require avg_sentiment_5d >= this to open a position (0=disabled)")
    p.add_argument("--block-miss-entry", action="store_true",
                   help="Block new entries when earnings_miss_signal=1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.collection} ...")
    df = load_feature_frame(args.collection, args.db, args.min_date)
    if df.empty:
        print("No data loaded.")
        return
    print(f"Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols, "
          f"{df['trade_date'].min().date()} → {df['trade_date'].max().date()}")

    print("Computing composite score ...")
    df = compute_composite_score(df)

    print("\n=== Event-Driven Backtest ===")
    print(f"entry_threshold={args.entry_threshold}  exit_threshold={args.exit_threshold}  "
          f"hold_threshold={args.hold_threshold}  min_hold={args.min_hold}  "
          f"max_hold={args.max_hold}  top_n={args.top_n}")

    result = run_event_driven_backtest(
        df,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        hold_threshold=args.hold_threshold,
        min_hold=args.min_hold,
        max_hold=args.max_hold,
        top_n=args.top_n,
        sentiment_exit_threshold=args.sentiment_exit,
        sentiment_shift_exit=args.sentiment_shift_exit,
        sentiment_entry_min=args.sentiment_entry_min,
        block_miss_entry=args.block_miss_entry,
    )

    print(f"\nTotal trades:      {result.total_trades:,}")
    print(f"Avg hold days:     {result.avg_hold_days:.1f}")
    print(f"Mean excess ret:   {result.mean_excess_ret:.2%}")
    print(f"Hit rate:          {result.hit_rate:.1%}")
    print(f"\nExit reasons:")
    for reason, count in sorted(result.exit_reasons.items(), key=lambda x: -x[1]):
        pct = count / result.total_trades * 100
        print(f"  {reason:<35} {count:>6} ({pct:.1f}%)")

    print(f"\n{'Year':<6} {'Trades':>7} {'Avg Hold':>9} {'Mean ExRet':>11} {'Hit Rate':>9}")
    print("-" * 47)
    for yr in sorted(result.yearly):
        yr_data = result.yearly[yr]
        er = yr_data["excess_rets"]
        hd = yr_data["hold_days"]
        mean_er = np.mean(er) if er else float("nan")
        hit = np.mean([r > 0 for r in er]) if er else float("nan")
        avg_hold = np.mean(hd) if hd else float("nan")
        print(f"{yr:<6} {yr_data['trades']:>7} {avg_hold:>9.1f} {mean_er:>10.2%} {hit:>8.1%}")

    print("\n=== Fixed-Horizon Baseline (same entry signal) ===")
    print(f"{'Horizon':<10} {'Mean ExRet':>11} {'Hit Rate':>9} {'Obs':>7}")
    print("-" * 40)
    for h in FORWARD_HORIZONS:
        baseline = run_fixed_baseline(df, h, args.top_n)
        if not baseline:
            continue
        print(f"{h}d{'':<8} {baseline['mean_excess_ret']:>10.2%} "
              f"{baseline['hit_rate']:>8.1%} {baseline['observations']:>7,}")


if __name__ == "__main__":
    main()
