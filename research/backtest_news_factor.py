#!/usr/bin/env python3
"""Run simple cross-sectional factor backtests on daily symbol features."""

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
    strategy: str
    bucket_size: int
    observations: int
    mean_return: float
    mean_universe_return: float | None
    mean_excess_return: float | None
    annualized_return: float | None
    annualized_excess_return: float | None
    hit_rate: float | None
    avg_names_selected: float
    sharpe: float | None = None
    information_ratio: float | None = None
    max_drawdown: float | None = None
    volatility: float | None = None


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
        default=["article_count", "news_burst_20d", "full_ratio", "quality_score", "surprise_pct_last"],
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
    parser.add_argument(
        "--top-ns",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        help="List of portfolio sizes to evaluate",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["top", "bottom", "long_short"],
        choices=["top", "bottom", "long_short"],
        help="Which simple portfolio constructions to evaluate",
    )
    parser.add_argument(
        "--min-trade-date",
        default="2015-12-04",
        help="Safety cutoff for trade_date; rows before this are excluded",
    )
    parser.add_argument(
        "--excess-mode",
        default="benchmark",
        choices=["benchmark", "universe"],
        help="How to compute excess return for long-only buckets",
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
    parser.add_argument(
        "--group-by-year",
        action="store_true",
        help="Also print a year-by-year breakdown using the same backtest settings",
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
        "sector": 1,
        "date": 1,
        "trade_date": 1,
        "article_count": 1,
        "news_burst_20d": 1,
        "full_ratio": 1,
        "quality_score": 1,
        "unique_source_count": 1,
        "avg_content_length": 1,
        "extraction_failed_count": 1,
        "future_ret_5d": 1,
        "future_ret_10d": 1,
        "future_ret_15d": 1,
        "future_ret_20d": 1,
        "future_ret_30d": 1,
        "future_ret_45d": 1,
        "future_ret_60d": 1,
        "benchmark_symbol": 1,
        "benchmark_ret_5d": 1,
        "benchmark_ret_10d": 1,
        "benchmark_ret_15d": 1,
        "benchmark_ret_20d": 1,
        "benchmark_ret_30d": 1,
        "benchmark_ret_45d": 1,
        "benchmark_ret_60d": 1,
        "excess_ret_5d": 1,
        "excess_ret_10d": 1,
        "excess_ret_15d": 1,
        "excess_ret_20d": 1,
        "excess_ret_30d": 1,
        "excess_ret_45d": 1,
        "excess_ret_60d": 1,
        "past_ret_5d": 1,
        "past_ret_20d": 1,
        "past_ret_60d": 1,
        "volatility_20d": 1,
        "volatility_60d": 1,
        "volume_shock_20d": 1,
        "days_to_earnings": 1,
        "days_since_earnings": 1,
        "is_earnings_window_5d": 1,
        "is_post_earnings_window_20d": 1,
        "eps_estimate_last": 1,
        "reported_eps_last": 1,
        "surprise_pct_last": 1,
        "is_positive_surprise": 1,
        "is_negative_surprise": 1,
        "days_to_earnings_bucket": 1,
        "days_since_earnings_bucket": 1,
        "is_pre_earnings_10d": 1,
        "is_post_earnings_5d": 1,
        "is_post_earnings_10d": 1,
        "is_post_positive_surprise_20d": 1,
        "is_post_negative_surprise_20d": 1,
        "surprise_bucket": 1,
        "full_ratio_x_post_positive": 1,
        "quality_score_x_post_negative": 1,
        "earnings_recency_weight": 1,
        # LLM sentiment features (Stage 3.5.2)
        "avg_sentiment_3d": 1,
        "avg_sentiment_5d": 1,
        "sentiment_shift_5d": 1,
        "high_signal_count_3d": 1,
        "has_regulatory_risk_5d": 1,
        "earnings_beat_signal": 1,
        "earnings_miss_signal": 1,
        "disagreement_avg_5d": 1,
        "negative_event_count_5d": 1,
    }
    rows = list(coll.find(query, projection))
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return add_quality_score(frame)


def add_quality_score(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    quality_components = {
        "full_ratio": 1.0,
        "unique_source_count": 1.0,
        "avg_content_length": 1.0,
        "extraction_failed_count": -1.0,
        "news_burst_20d": 0.5,
    }

    result = frame.copy()
    result["quality_score"] = 0.0
    result["quality_component_count"] = 0

    for trade_date, idx in result.groupby("trade_date").groups.items():
        group = result.loc[idx].copy()
        score = pd.Series(0.0, index=group.index, dtype="float64")
        count = pd.Series(0, index=group.index, dtype="int64")

        for column, weight in quality_components.items():
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            valid = values.notna()
            if valid.sum() == 0:
                continue

            ranked = values[valid].rank(method="average", pct=True)
            centered = ranked - 0.5
            score.loc[valid.index] += centered * weight
            count.loc[valid.index] += 1

        valid_count = count > 0
        score.loc[valid_count] = score.loc[valid_count] / count.loc[valid_count]
        score.loc[~valid_count] = pd.NA

        result.loc[group.index, "quality_score"] = score
        result.loc[group.index, "quality_component_count"] = count

    return result


def _annualize(mean_period_return: float, horizon: int) -> float | None:
    if pd.isna(mean_period_return):
        return None
    base = 1.0 + mean_period_return
    if base <= 0:
        return None
    return float(base ** (252.0 / horizon) - 1.0)


def _sharpe(returns: pd.Series, horizon: int) -> float | None:
    """Annualized Sharpe ratio (risk-free = 0)."""
    if len(returns) < 2:
        return None
    std = returns.std()
    if std == 0 or pd.isna(std):
        return None
    return float(returns.mean() / std * math.sqrt(252.0 / horizon))


def _information_ratio(excess: pd.Series, horizon: int) -> float | None:
    """Annualized Information Ratio = annualized mean excess / tracking error."""
    if len(excess) < 2:
        return None
    std = excess.std()
    if std == 0 or pd.isna(std):
        return None
    return float(excess.mean() / std * math.sqrt(252.0 / horizon))


def _max_drawdown(returns: pd.Series) -> float | None:
    """Maximum drawdown from period return series."""
    if len(returns) < 2:
        return None
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    return float(drawdowns.min())


def _clean_subset(frame: pd.DataFrame, factor: str, horizon: int) -> pd.DataFrame:
    target_col = f"future_ret_{horizon}d"
    subset = frame[["trade_date", "symbol", factor, target_col]].copy()
    subset = subset.dropna(subset=[factor, target_col])
    subset = subset[subset[factor].apply(lambda x: isinstance(x, (int, float)) and not math.isnan(float(x)))]
    return subset


def run_factor_backtest(
    frame: pd.DataFrame,
    factor: str,
    horizon: int,
    bucket_size: int,
    strategy: str,
    excess_mode: str,
) -> BacktestResult:
    target_col = f"future_ret_{horizon}d"
    benchmark_col = f"benchmark_ret_{horizon}d"
    subset = _clean_subset(frame, factor, horizon)
    if subset.empty:
        return BacktestResult(factor, horizon, strategy, bucket_size, 0, math.nan, math.nan, math.nan, None, None, None, 0.0)

    strategy_returns = []
    universe_returns = []
    selection_sizes = []

    for _, group in subset.groupby("trade_date"):
        ordered = group.sort_values([factor, "symbol"], ascending=[False, True])
        group_benchmark = pd.to_numeric(group.get(benchmark_col), errors="coerce") if benchmark_col in group.columns else None

        if strategy == "top":
            selected = ordered.head(bucket_size)
            if selected.empty:
                continue
            strategy_ret = selected[target_col].mean()
            selection_sizes.append(len(selected))
        elif strategy == "bottom":
            selected = ordered.tail(bucket_size)
            if selected.empty:
                continue
            strategy_ret = selected[target_col].mean()
            selection_sizes.append(len(selected))
        else:
            top_bucket = ordered.head(bucket_size)
            bottom_bucket = ordered.tail(bucket_size)
            if top_bucket.empty or bottom_bucket.empty:
                continue
            strategy_ret = top_bucket[target_col].mean() - bottom_bucket[target_col].mean()
            selection_sizes.append(min(len(top_bucket), len(bottom_bucket)) * 2)

        strategy_returns.append(strategy_ret)
        if excess_mode == "benchmark" and group_benchmark is not None and group_benchmark.notna().any():
            universe_returns.append(group_benchmark.dropna().mean())
        else:
            universe_returns.append(group[target_col].mean())

    if not strategy_returns:
        return BacktestResult(factor, horizon, strategy, bucket_size, 0, math.nan, math.nan, math.nan, None, None, None, 0.0)

    series = pd.Series(strategy_returns, dtype="float64")
    universe = pd.Series(universe_returns, dtype="float64")

    if strategy == "long_short":
        excess = series
        mean_universe = None
    else:
        excess = series - universe
        mean_universe = float(universe.mean())

    mean_return = float(series.mean())
    mean_excess = float(excess.mean())
    hit_rate = float((excess > 0).mean()) if len(excess) else None

    return BacktestResult(
        factor=factor,
        horizon=horizon,
        strategy=strategy,
        bucket_size=bucket_size,
        observations=len(series),
        mean_return=mean_return,
        mean_universe_return=mean_universe,
        mean_excess_return=mean_excess,
        annualized_return=_annualize(mean_return, horizon),
        annualized_excess_return=_annualize(mean_excess, horizon),
        hit_rate=hit_rate,
        avg_names_selected=float(pd.Series(selection_sizes).mean()),
        sharpe=_sharpe(series, horizon),
        information_ratio=_information_ratio(excess, horizon),
        max_drawdown=_max_drawdown(series),
        volatility=float(series.std() * math.sqrt(252.0 / horizon)) if len(series) >= 2 else None,
    )


def print_summary(
    results: list[BacktestResult],
    frame: pd.DataFrame,
    collection: str,
    min_trade_date: str,
    excess_mode: str,
) -> None:
    print("=== News Factor Backtest ===")
    print(f"collection={collection}")
    print(f"rows={len(frame):,}")
    print(f"trade_date_range={frame['trade_date'].min().date()} -> {frame['trade_date'].max().date()}")
    print(f"symbols={frame['symbol'].nunique()}")
    print(f"min_trade_date={min_trade_date}")
    print(f"excess_mode={excess_mode}")
    print("")
    header = (
        f"{'factor':<18} {'horizon':>7} {'strategy':<10} {'N':>4} {'obs':>7} "
        f"{'mean_excess':>12} {'ann_excess':>12} {'hit_rate':>10} "
        f"{'sharpe':>8} {'IR':>8} {'max_dd':>9} {'vol':>8}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        ann_ex  = f"{item.annualized_excess_return:.2%}" if item.annualized_excess_return is not None else "n/a"
        hit_rate = f"{item.hit_rate:.2%}" if item.hit_rate is not None else "n/a"
        sharpe   = f"{item.sharpe:.2f}"  if item.sharpe  is not None else "n/a"
        ir       = f"{item.information_ratio:.2f}" if item.information_ratio is not None else "n/a"
        mdd      = f"{item.max_drawdown:.2%}" if item.max_drawdown is not None else "n/a"
        vol      = f"{item.volatility:.2%}" if item.volatility is not None else "n/a"
        print(
            f"{item.factor:<18} {item.horizon:>7} {item.strategy:<10} {item.bucket_size:>4} {item.observations:>7} "
            f"{item.mean_excess_return:>11.2%} {ann_ex:>12} {hit_rate:>10} "
            f"{sharpe:>8} {ir:>8} {mdd:>9} {vol:>8}"
        )


def print_yearly_summary(
    frame: pd.DataFrame,
    factors: list[str],
    horizons: list[int],
    top_ns: list[int],
    strategies: list[str],
    excess_mode: str,
) -> None:
    if frame.empty:
        return

    working = frame.copy()
    working["year"] = working["trade_date"].dt.year

    print("")
    print("=== Yearly Stability Breakdown ===")
    header = (
        f"{'year':>6} {'factor':<18} {'horizon':>7} {'strategy':<10} {'N':>4} {'obs':>7} "
        f"{'ann_excess':>12} {'hit_rate':>10} {'sharpe':>8} {'IR':>8} {'max_dd':>9}"
    )
    print(header)
    print("-" * len(header))

    for year in sorted(working["year"].dropna().unique()):
        year_frame = working[working["year"] == year].copy()
        if year_frame.empty:
            continue
        for factor in factors:
            if factor not in year_frame.columns:
                continue
            for horizon in horizons:
                for bucket_size in top_ns:
                    for strategy in strategies:
                        result = run_factor_backtest(
                            year_frame,
                            factor,
                            horizon,
                            bucket_size,
                            strategy,
                            excess_mode,
                        )
                        if result.observations == 0:
                            continue
                        ann_ex   = f"{result.annualized_excess_return:.2%}" if result.annualized_excess_return is not None else "n/a"
                        hit_rate = f"{result.hit_rate:.2%}" if result.hit_rate is not None else "n/a"
                        sharpe   = f"{result.sharpe:.2f}" if result.sharpe is not None else "n/a"
                        ir       = f"{result.information_ratio:.2f}" if result.information_ratio is not None else "n/a"
                        mdd      = f"{result.max_drawdown:.2%}" if result.max_drawdown is not None else "n/a"
                        print(
                            f"{int(year):>6} {result.factor:<18} {horizon:>7} {strategy:<10} {bucket_size:>4} {result.observations:>7} "
                            f"{ann_ex:>12} {hit_rate:>10} {sharpe:>8} {ir:>8} {mdd:>9}"
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
            for bucket_size in args.top_ns:
                for strategy in args.strategies:
                    results.append(
                        run_factor_backtest(
                            frame,
                            factor,
                            horizon,
                            bucket_size,
                            strategy,
                            args.excess_mode,
                        )
                    )

    print_summary(results, frame, args.collection, args.min_trade_date, args.excess_mode)
    if args.group_by_year:
        print_yearly_summary(
            frame=frame,
            factors=args.factors,
            horizons=args.horizons,
            top_ns=args.top_ns,
            strategies=args.strategies,
            excess_mode=args.excess_mode,
        )


if __name__ == "__main__":
    main()
