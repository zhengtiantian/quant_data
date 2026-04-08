#!/usr/bin/env python3
"""Run a simple cross-sectional factor backtest on daily symbol features."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse, urlunparse

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
GLOBAL_ENV = ROOT / ".env"
load_dotenv(GLOBAL_ENV, override=False)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Missing MONGO_URI")
LOCAL_MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")


def _fallback_mongo_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.hostname in {"mongo6", "mongodb", "mongo"}:
        fallback = urlparse(LOCAL_MONGO_URI)
        return urlunparse(
            (
                parsed.scheme or fallback.scheme,
                fallback.netloc,
                parsed.path or fallback.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
    return uri


def create_client() -> MongoClient:
    primary_uri = MONGO_URI
    try:
        client = MongoClient(primary_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except Exception:
        fallback_uri = _fallback_mongo_uri(primary_uri)
        if fallback_uri == primary_uri:
            raise
        client = MongoClient(fallback_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client


@dataclass
class BacktestResult:
    factor: str
    horizon: int
    observations: int
    mean_portfolio_return: float
    mean_universe_return: float
    mean_excess_return: float
    annualized_portfolio_return: float | None
    annualized_excess_return: float | None
    hit_rate: float | None
    avg_names_selected: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default="daily_symbol_features_company_matched_v1",
        help="Mongo collection containing daily symbol features",
    )
    parser.add_argument("--db", default="quant_data", help="Mongo database name")
    parser.add_argument(
        "--factors",
        nargs="+",
        default=["article_count", "news_burst_20d", "full_ratio"],
        help="Feature columns to rank on",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[5, 20, 60],
        choices=[5, 20, 60],
        help="Forward return horizons to evaluate",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of names to hold each rebalance")
    parser.add_argument(
        "--min-trade-date",
        default="2015-12-04",
        help="Safety cutoff for trade_date; rows before this are excluded",
    )
    parser.add_argument(
        "--max-trade-date",
        default=None,
        help="Optional inclusive end date in YYYY-MM-DD",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional symbol subset",
    )
    return parser.parse_args()


def load_feature_frame(
    db_name: str,
    collection_name: str,
    min_trade_date: str,
    max_trade_date: str | None,
    symbols: Iterable[str] | None,
) -> pd.DataFrame:
    client = create_client()
    coll = client[db_name][collection_name]

    query: dict = {"trade_date": {"$gte": min_trade_date}}
    if max_trade_date:
        query["trade_date"]["$lte"] = max_trade_date
    if symbols:
        query["symbol"] = {"$in": list(symbols)}

    projection = {
        "_id": 0,
        "symbol": 1,
        "name": 1,
        "date": 1,
        "trade_date": 1,
        "article_count": 1,
        "news_burst_20d": 1,
        "full_ratio": 1,
        "future_ret_5d": 1,
        "future_ret_20d": 1,
        "future_ret_60d": 1,
    }
    rows = list(coll.find(query, projection))
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _annualize(mean_period_return: float, horizon: int) -> float | None:
    if pd.isna(mean_period_return):
        return None
    base = 1.0 + mean_period_return
    if base <= 0:
        return None
    return float(base ** (252.0 / horizon) - 1.0)


def run_factor_backtest(frame: pd.DataFrame, factor: str, horizon: int, top_n: int) -> BacktestResult:
    target_col = f"future_ret_{horizon}d"
    subset = frame[["trade_date", "symbol", factor, target_col]].copy()
    subset = subset.dropna(subset=[factor, target_col])
    subset = subset[subset[factor].apply(lambda x: isinstance(x, (int, float)) and not math.isnan(float(x)))]
    if subset.empty:
        return BacktestResult(factor, horizon, 0, math.nan, math.nan, math.nan, None, None, None, 0.0)

    selected_returns = []
    universe_returns = []
    selection_sizes = []

    for trade_date, group in subset.groupby("trade_date"):
        ordered = group.sort_values([factor, "symbol"], ascending=[False, True])
        selection = ordered.head(top_n)
        if selection.empty:
            continue

        selected_returns.append(selection[target_col].mean())
        universe_returns.append(group[target_col].mean())
        selection_sizes.append(len(selection))

    if not selected_returns:
        return BacktestResult(factor, horizon, 0, math.nan, math.nan, math.nan, None, None, None, 0.0)

    portfolio = pd.Series(selected_returns, dtype="float64")
    universe = pd.Series(universe_returns, dtype="float64")
    excess = portfolio - universe

    mean_portfolio = float(portfolio.mean())
    mean_universe = float(universe.mean())
    mean_excess = float(excess.mean())
    hit_rate = float((excess > 0).mean()) if len(excess) else None

    return BacktestResult(
        factor=factor,
        horizon=horizon,
        observations=len(portfolio),
        mean_portfolio_return=mean_portfolio,
        mean_universe_return=mean_universe,
        mean_excess_return=mean_excess,
        annualized_portfolio_return=_annualize(mean_portfolio, horizon),
        annualized_excess_return=_annualize(mean_excess, horizon),
        hit_rate=hit_rate,
        avg_names_selected=float(pd.Series(selection_sizes).mean()),
    )


def print_summary(results: list[BacktestResult], frame: pd.DataFrame, collection: str, min_trade_date: str) -> None:
    print("=== News Factor Backtest ===")
    print(f"collection={collection}")
    print(f"rows={len(frame):,}")
    print(f"trade_date_range={frame['trade_date'].min().date()} -> {frame['trade_date'].max().date()}")
    print(f"symbols={frame['symbol'].nunique()}")
    print(f"min_trade_date={min_trade_date}")
    print("")
    header = (
        f"{'factor':<18} {'horizon':>7} {'obs':>7} {'mean_ret':>11} "
        f"{'mean_excess':>12} {'ann_ret':>11} {'ann_excess':>12} {'hit_rate':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        ann_ret = f"{item.annualized_portfolio_return:.2%}" if item.annualized_portfolio_return is not None else "n/a"
        ann_ex = f"{item.annualized_excess_return:.2%}" if item.annualized_excess_return is not None else "n/a"
        hit_rate = f"{item.hit_rate:.2%}" if item.hit_rate is not None else "n/a"
        print(
            f"{item.factor:<18} {item.horizon:>7} {item.observations:>7} "
            f"{item.mean_portfolio_return:>10.2%} {item.mean_excess_return:>11.2%} "
            f"{ann_ret:>11} {ann_ex:>12} {hit_rate:>10}"
        )


def main() -> None:
    args = parse_args()
    frame = load_feature_frame(
        db_name=args.db,
        collection_name=args.collection,
        min_trade_date=args.min_trade_date,
        max_trade_date=args.max_trade_date,
        symbols=args.symbols,
    )
    if frame.empty:
        print("No feature rows available for backtest.")
        return

    results = []
    for factor in args.factors:
        if factor not in frame.columns:
            print(f"Skipping missing factor: {factor}")
            continue
        for horizon in args.horizons:
            results.append(run_factor_backtest(frame, factor, horizon, args.top_n))

    print_summary(results, frame, args.collection, args.min_trade_date)


if __name__ == "__main__":
    main()
