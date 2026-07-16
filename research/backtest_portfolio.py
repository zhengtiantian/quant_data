#!/usr/bin/env python3
"""Portfolio backtest (C.6 / C.2 visualised).

Walks the full feature history, scores symbols each rebalance date with the
same composite score used live, holds the top-N for a fixed horizon
(non-overlapping), and builds an equity curve vs SPY plus headline stats
(annualised return, Sharpe, max drawdown, win rate).

Uses precomputed forward-return labels (future_ret_Nd / spy_ret_Nd) in
daily_symbol_features — no price re-fetch.

Reads:   quant_data.daily_symbol_features
Writes:  quant_data.portfolio_performance   (single latest doc)

Run:
  LOCAL_MONGO_URI="mongodb://root:root@127.0.0.1:37018/" \
  .venv311/bin/python research/backtest_portfolio.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import MongoClient

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_daily_signals import compute_score, _WEIGHTS_BY_REGIME  # reuse live scoring
# All regimes share the same feature keys; use NEUTRAL as the canonical key set.
_WEIGHTS = _WEIGHTS_BY_REGIME["NEUTRAL"]

ROOT = Path(__file__).resolve().parents[1]
MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"
FEATURE_COLLECTION = os.getenv("FEATURE_OUTPUT_COLLECTION", "daily_symbol_features")
PERF_COLLECTION = "portfolio_performance"

HOLD_DAYS = int(os.getenv("BACKTEST_HOLD_DAYS", "20"))   # uses future_ret_{HOLD}d
TOP_N = int(os.getenv("BACKTEST_TOP_N", "5"))
MIN_SYMBOLS = int(os.getenv("BACKTEST_MIN_SYMBOLS", "10"))
RF_ANNUAL = float(os.getenv("BACKTEST_RF", "0.04"))
PERIODS_PER_YEAR = 252.0 / HOLD_DAYS

# H.1.1 — transaction cost model (round-trip: entry + exit each incur cost)
COMMISSION_BPS = float(os.getenv("BACKTEST_COMMISSION_BPS", "5"))      # 0.05% one-way
SLIPPAGE_LARGE_BPS = float(os.getenv("BACKTEST_SLIPPAGE_LARGE_BPS", "10"))  # 0.10% one-way, liquid names
SLIPPAGE_SMALL_BPS = float(os.getenv("BACKTEST_SLIPPAGE_SMALL_BPS", "30"))  # 0.30% one-way, thin names
LARGE_DOLLAR_VOL = float(os.getenv("BACKTEST_LARGE_DOLLAR_VOL", "50000000"))  # $50M/day = "liquid"

# H.1.2 — liquidity filter: exclude symbols whose 20d avg dollar volume is too thin to execute
MIN_DOLLAR_VOL = float(os.getenv("BACKTEST_MIN_DOLLAR_VOL", "5000000"))  # $5M/day


def load_features() -> pd.DataFrame:
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][FEATURE_COLLECTION]
    fut = f"future_ret_{HOLD_DAYS}d"
    bench = f"spy_ret_{HOLD_DAYS}d"
    proj = {"_id": 0, "trade_date": 1, "symbol": 1, fut: 1, bench: 1,
            "volume_20d_avg": 1, "close": 1}
    for f in _WEIGHTS:
        proj[f] = 1
    df = pd.DataFrame(list(col.find({}, proj)))
    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    df["dollar_vol_20d"] = pd.to_numeric(df.get("volume_20d_avg"), errors="coerce") * \
        pd.to_numeric(df.get("close"), errors="coerce")
    return df, fut, bench


def _round_trip_cost_bps(dollar_vol: float | None) -> float:
    """Commission + slippage for one entry + one exit, in return units (not bps)."""
    if dollar_vol is None or pd.isna(dollar_vol):
        slippage = SLIPPAGE_SMALL_BPS  # unknown liquidity -> assume thin
    else:
        slippage = SLIPPAGE_LARGE_BPS if dollar_vol >= LARGE_DOLLAR_VOL else SLIPPAGE_SMALL_BPS
    one_way = (COMMISSION_BPS + slippage) / 10_000.0
    return one_way * 2  # entry + exit


def run_backtest(df: pd.DataFrame, fut: str, bench: str):
    dates = sorted(df["trade_date"].unique())
    by_date = {d: g for d, g in df.groupby("trade_date")}

    rebal_dates, strat_rets, bench_rets, net_strat_rets, cost_drags = [], [], [], [], []
    excluded_thin = 0
    i = 0
    while i < len(dates):
        d = dates[i]
        g = by_date[d]
        g = g[pd.to_numeric(g[fut], errors="coerce").notna()]

        # H.1.2 liquidity filter: drop symbols too thin to execute at size
        liquid = g[pd.to_numeric(g["dollar_vol_20d"], errors="coerce").fillna(0) >= MIN_DOLLAR_VOL]
        excluded_thin += len(g) - len(liquid)
        g = liquid

        if len(g) >= MIN_SYMBOLS:
            scored = compute_score(g)
            top = scored[scored["signal_rank"] <= TOP_N]
            sr = pd.to_numeric(top[fut], errors="coerce").mean()
            br = pd.to_numeric(g[bench], errors="coerce").dropna()
            br = float(br.iloc[0]) if len(br) else 0.0
            if pd.notna(sr):
                # H.1.1 transaction cost model: per-position round-trip cost, averaged across top-N
                costs = top["dollar_vol_20d"].apply(_round_trip_cost_bps)
                avg_cost = float(costs.mean()) if len(costs) else _round_trip_cost_bps(None)
                rebal_dates.append(d)
                strat_rets.append(float(sr))
                net_strat_rets.append(float(sr) - avg_cost)
                cost_drags.append(avg_cost)
                bench_rets.append(br)
        i += HOLD_DAYS  # non-overlapping

    if excluded_thin:
        print(f"Liquidity filter: excluded {excluded_thin} thin symbol-dates "
              f"(dollar_vol_20d < ${MIN_DOLLAR_VOL:,.0f})")

    return rebal_dates, np.array(strat_rets), np.array(bench_rets), np.array(net_strat_rets), np.array(cost_drags)


def equity(rets: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + rets)


def stats(rets: np.ndarray, eq: np.ndarray) -> dict:
    n = len(rets)
    if n == 0:
        return {}
    total = float(eq[-1] - 1.0)
    ann_ret = float(eq[-1] ** (PERIODS_PER_YEAR / n) - 1.0)
    vol = float(rets.std(ddof=1)) if n > 1 else 0.0
    ann_vol = vol * np.sqrt(PERIODS_PER_YEAR)
    sharpe = float((rets.mean() * PERIODS_PER_YEAR - RF_ANNUAL) / ann_vol) if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min())
    return {
        "total_return": round(total, 4),
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(mdd, 4),
        "win_rate": round(float((rets > 0).mean()), 4),
        "avg_period_return": round(float(rets.mean()), 4),
        "num_periods": int(n),
    }


def main():
    df, fut, bench = load_features()
    print(f"Loaded {len(df):,} feature rows; backtesting hold={HOLD_DAYS}d top_n={TOP_N}")

    dates, sret, bret, nret, cost_drags = run_backtest(df, fut, bench)
    if len(dates) == 0:
        print("No rebalance periods produced.")
        return

    seq, beq, neq = equity(sret), equity(bret), equity(nret)
    curve = [
        {"date": d, "strategy": round(float(s), 4), "benchmark": round(float(b), 4),
         "strategy_net": round(float(n), 4)}
        for d, s, b, n in zip(dates, seq, beq, neq)
    ]
    doc = {
        "_id": "latest",
        "generated_at": datetime.now(timezone.utc),
        "hold_days": HOLD_DAYS,
        "top_n": TOP_N,
        "rebalances": len(dates),
        "start_date": dates[0],
        "end_date": dates[-1],
        "equity_curve": curve,
        "avg_round_trip_cost_bps": round(float(cost_drags.mean()) * 10_000, 1) if len(cost_drags) else None,
        "stats": {
            "strategy_gross": stats(sret, seq),
            "strategy_net": stats(nret, neq),
            "benchmark": stats(bret, beq),
        },
    }

    client = MongoClient(MONGO_URI)
    client[DB_NAME][PERF_COLLECTION].replace_one({"_id": "latest"}, doc, upsert=True)

    st = doc["stats"]["strategy_gross"]
    nt = doc["stats"]["strategy_net"]
    bt = doc["stats"]["benchmark"]
    avg_cost = doc["avg_round_trip_cost_bps"]
    print(f"\nBacktest {dates[0]} -> {dates[-1]}  ({len(dates)} rebalances, {HOLD_DAYS}d hold)")
    print(f"Avg round-trip cost: {avg_cost} bps/position "
          f"(commission {COMMISSION_BPS}bps + liquidity-tiered slippage {SLIPPAGE_LARGE_BPS}/{SLIPPAGE_SMALL_BPS}bps, x2 for entry+exit)")
    print(f"{'':12}{'Gross':>12}{'Net (after cost)':>18}{'SPY':>12}")
    for k in ["total_return", "ann_return", "sharpe", "max_drawdown", "win_rate"]:
        print(f"{k:12}{st[k]:>12}{nt[k]:>18}{bt.get(k,'-'):>12}")
    print(f"\nFinal equity: gross {seq[-1]:.2f}x  net {neq[-1]:.2f}x  vs  SPY {beq[-1]:.2f}x")


if __name__ == "__main__":
    main()
