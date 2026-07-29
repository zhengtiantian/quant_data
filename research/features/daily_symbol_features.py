#!/usr/bin/env python3
"""Build daily symbol-level research features from news and price history."""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[2]
GLOBAL_ENV = ROOT / ".env"
load_dotenv(GLOBAL_ENV, override=False)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Missing MONGO_URI")
LOCAL_MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")

DB_NAME = os.getenv("FEATURE_DB_NAME", "quant_data")
NEWS_COLLECTION = os.getenv("FEATURE_NEWS_COLLECTION", "news_articles")
PRICE_COLLECTION = os.getenv("FEATURE_PRICE_COLLECTION", "stock_prices_history")
FEATURE_COLLECTION = os.getenv("FEATURE_OUTPUT_COLLECTION", "daily_symbol_features")
UNIVERSE_COLLECTION = os.getenv("FEATURE_UNIVERSE_COLLECTION", "stock_universe")
EARNINGS_COLLECTION = os.getenv("FEATURE_EARNINGS_COLLECTION", "earnings_events")
LLM_COLLECTION = os.getenv("FEATURE_LLM_COLLECTION", "news_articles_company_matched_v2")
BENCHMARK_SYMBOLS = [
    symbol.strip().upper()
    for symbol in os.getenv("FEATURE_BENCHMARK_SYMBOLS", "SPY,QQQ").split(",")
    if symbol.strip()
]
PRIMARY_BENCHMARK = os.getenv("FEATURE_PRIMARY_BENCHMARK", "QQQ").strip().upper()

FORWARD_HORIZONS = [5, 10, 15, 20, 30, 45, 60]

FEATURE_REBUILD_ALL = os.getenv("FEATURE_REBUILD_ALL", "false").lower() == "true"
FEATURE_LOOKBACK_DAYS = int(os.getenv("FEATURE_LOOKBACK_DAYS", "180"))
FEATURE_START_DATE = os.getenv("FEATURE_START_DATE")
FEATURE_END_DATE = os.getenv("FEATURE_END_DATE")
MIN_VALID_NEWS_DATE = pd.Timestamp(os.getenv("FEATURE_MIN_VALID_DATE", "1990-01-01"))
MAX_VALID_NEWS_DATE = pd.Timestamp(datetime.now(UTC).date()) + pd.Timedelta(days=1)


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


def _parse_iso_date(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    value = raw_value.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return None


def _is_reasonable_feature_date(value: str | None) -> bool:
    if not value:
        return False
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return False
    return MIN_VALID_NEWS_DATE <= ts <= MAX_VALID_NEWS_DATE


def _extract_news_date(doc: dict) -> str | None:
    raw_date = doc.get("date")
    if raw_date:
        digits = "".join(ch for ch in str(raw_date) if ch.isdigit())
        if len(digits) >= 8:
            parsed = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
            if _is_reasonable_feature_date(parsed):
                return parsed
    for key in ("publishedAt", "timestamp", "collectedAt"):
        parsed = _parse_iso_date(doc.get(key))
        if parsed and _is_reasonable_feature_date(parsed):
            return parsed
    return None


def _extract_price_date(doc: dict) -> str | None:
    for key in ("timestamp", "date", "publishedAt", "collectedAt"):
        value = doc.get(key)
        if key == "date" and value:
            digits = "".join(ch for ch in str(value) if ch.isdigit())
            if len(digits) >= 8:
                parsed = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
                if _is_reasonable_feature_date(parsed):
                    return parsed
        parsed = _parse_iso_date(value)
        if parsed and _is_reasonable_feature_date(parsed):
            return parsed
    return None


def _resolve_range() -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    if FEATURE_START_DATE:
        base_start = pd.Timestamp(FEATURE_START_DATE)
    elif FEATURE_REBUILD_ALL:
        base_start = None
    else:
        base_start = pd.Timestamp(datetime.now(UTC).date()) - pd.Timedelta(days=FEATURE_LOOKBACK_DAYS)

    end_date = pd.Timestamp(FEATURE_END_DATE) if FEATURE_END_DATE else None
    effective_start = None if base_start is None else base_start - pd.Timedelta(days=20)
    return base_start, end_date, effective_start


def load_news_frame() -> pd.DataFrame:
    _, end_date, effective_start = _resolve_range()
    client = create_client()
    col = client[DB_NAME][NEWS_COLLECTION]

    cursor = col.find(
        {"symbol": {"$exists": True, "$ne": None}},
        {
            "_id": 0,
            "symbol": 1,
            "name": 1,
            "date": 1,
            "publishedAt": 1,
            "timestamp": 1,
            "collectedAt": 1,
            "data_quality": 1,
            "content_length": 1,
            "content": 1,
            "url": 1,
            "note": 1,
            "source": 1,
        },
    )

    rows = []
    for doc in cursor:
        news_date = _extract_news_date(doc)
        if not news_date:
            continue
        date_ts = pd.Timestamp(news_date)
        if effective_start is not None and date_ts < effective_start:
            continue
        if end_date is not None and date_ts > end_date:
            continue
        source = doc.get("source") or {}
        if isinstance(source, str):
            source_platform = source
            source_name = source
        else:
            source_platform = source.get("platform") or "unknown"
            source_name = source.get("name") or source_platform
        note = doc.get("note") or ""
        content_length = int(doc.get("content_length") or len(doc.get("content") or ""))
        rows.append(
            {
                "symbol": doc.get("symbol"),
                "name": doc.get("name"),
                "date": date_ts,
                "data_quality": doc.get("data_quality") or "unknown",
                "content_length": content_length,
                "url": doc.get("url"),
                "source_platform": source_platform,
                "source_name": source_name,
                "note": note,
                "is_extraction_failed": int("failed" in note.lower() or "unavailable" in note.lower()),
                "is_timeout_fallback": int("fetch_timeout" in note.lower()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def aggregate_news_features(news_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        news_df.groupby(["symbol", "date"], as_index=False)
        .agg(
            name=("name", "first"),
            article_count=("url", "count"),
            full_count=("data_quality", lambda s: int((s == "full").sum())),
            title_only_count=("data_quality", lambda s: int((s == "title_only").sum())),
            url_only_count=("data_quality", lambda s: int((s == "url_only").sum())),
            avg_content_length=("content_length", "mean"),
            max_content_length=("content_length", "max"),
            unique_url_count=("url", pd.Series.nunique),
            unique_source_count=("source_name", pd.Series.nunique),
            unique_platform_count=("source_platform", pd.Series.nunique),
            extraction_failed_count=("is_extraction_failed", "sum"),
            timeout_fallback_count=("is_timeout_fallback", "sum"),
        )
        .sort_values(["symbol", "date"])
    )

    grouped["full_ratio"] = grouped["full_count"] / grouped["article_count"]
    grouped["title_only_ratio"] = grouped["title_only_count"] / grouped["article_count"]
    grouped["avg_content_length"] = grouped["avg_content_length"].fillna(0.0)

    def _add_rollups(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values("date").copy()
        frame["news_count_3d"] = frame["article_count"].rolling(3, min_periods=1).sum()
        frame["news_count_5d"] = frame["article_count"].rolling(5, min_periods=1).sum()
        frame["news_count_20d"] = frame["article_count"].rolling(20, min_periods=1).sum()
        frame["full_count_5d"] = frame["full_count"].rolling(5, min_periods=1).sum()
        frame["avg_full_ratio_5d"] = frame["full_ratio"].rolling(5, min_periods=1).mean()
        prior_mean = frame["article_count"].shift(1).rolling(20, min_periods=5).mean()
        frame["news_burst_20d"] = frame["article_count"] / prior_mean
        return frame

    # Iterate groups instead of .apply(): pandas 3.0 drops the grouping column
    # ('symbol') inside .apply(), which broke downstream groupby("symbol").
    # Iterating keeps 'symbol' in each group regardless of pandas version.
    parts = [_add_rollups(grp) for _, grp in grouped.groupby("symbol", group_keys=False)]
    grouped = pd.concat(parts, ignore_index=True) if parts else grouped
    return grouped.reset_index(drop=True)


def load_price_frame() -> pd.DataFrame:
    _, end_date, effective_start = _resolve_range()
    client = create_client()
    col = client[DB_NAME][PRICE_COLLECTION]

    cursor = col.find(
        {"symbol": {"$exists": True, "$ne": None}},
        {"_id": 0, "symbol": 1, "timestamp": 1, "date": 1, "close": 1, "volume": 1},
    )

    rows = []
    for doc in cursor:
        price_date = _extract_price_date(doc)
        if not price_date or doc.get("close") in (None, 0):
            continue
        date_ts = pd.Timestamp(price_date)
        if effective_start is not None and date_ts < effective_start:
            continue
        if end_date is not None and date_ts > end_date + pd.Timedelta(days=90):
            continue
        rows.append({
            "symbol": doc["symbol"], 
            "trade_date": date_ts, 
            "close": float(doc["close"]),
            "volume": float(doc.get("volume", 0))
        })

    if not rows:
        return pd.DataFrame()

    price_df = pd.DataFrame(rows).sort_values(["symbol", "trade_date"]).drop_duplicates(
        subset=["symbol", "trade_date"], keep="last"
    )
    return price_df


def load_earnings_frame() -> pd.DataFrame:
    _, end_date, effective_start = _resolve_range()
    client = create_client()
    col = client[DB_NAME][EARNINGS_COLLECTION]

    cursor = col.find(
        {"symbol": {"$exists": True, "$ne": None}},
        {
            "_id": 0,
            "symbol": 1,
            "event_date": 1,
            "earnings_date": 1,
            "eps_estimate": 1,
            "reported_eps": 1,
            "surprise_pct": 1,
        },
    )

    rows = []
    for doc in cursor:
        raw_event_date = doc.get("event_date") or doc.get("earnings_date")
        if not raw_event_date:
            continue
        try:
            event_date = pd.Timestamp(raw_event_date).normalize()
        except Exception:
            continue
        if effective_start is not None and event_date < effective_start - pd.Timedelta(days=90):
            continue
        if end_date is not None and event_date > end_date + pd.Timedelta(days=120):
            continue
        rows.append({
            "symbol": doc["symbol"],
            "event_date": event_date,
            "eps_estimate": doc.get("eps_estimate"),
            "reported_eps": doc.get("reported_eps"),
            "surprise_pct": doc.get("surprise_pct"),
        })

    if not rows:
        return pd.DataFrame(columns=["symbol", "event_date", "eps_estimate", "reported_eps", "surprise_pct"])

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["symbol", "event_date"])
        .sort_values(["symbol", "event_date"])
        .reset_index(drop=True)
    )


def build_benchmark_return_maps(price_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    benchmark_maps: dict[str, pd.DataFrame] = {}
    if price_df.empty:
        return benchmark_maps

    for symbol in BENCHMARK_SYMBOLS:
        bench = price_df[price_df["symbol"] == symbol].copy()
        if bench.empty:
            continue

        bench = bench.sort_values("trade_date").reset_index(drop=True)
        closes = bench["close"].to_numpy()

        def _forward_return(pos: int, offset: int):
            if pos + offset >= len(bench):
                return pd.NA
            return float(closes[pos + offset] / closes[pos] - 1.0)

        for h in FORWARD_HORIZONS:
            bench[f"{symbol.lower()}_ret_{h}d"] = [_forward_return(i, h) for i in range(len(bench))]

        bench["trade_date_key"] = bench["trade_date"].dt.strftime("%Y-%m-%d")
        benchmark_maps[symbol] = bench.set_index("trade_date_key")[
            [f"{symbol.lower()}_ret_{h}d" for h in FORWARD_HORIZONS]
        ]

    return benchmark_maps


def attach_price_labels(feature_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    benchmark_maps = build_benchmark_return_maps(price_df)
    benchmark_columns = []
    for symbol in BENCHMARK_SYMBOLS:
        benchmark_columns.extend([f"{symbol.lower()}_ret_{h}d" for h in FORWARD_HORIZONS])
    excess_columns = ["benchmark_symbol"] + \
        [f"benchmark_ret_{h}d" for h in FORWARD_HORIZONS] + \
        [f"excess_ret_{h}d" for h in FORWARD_HORIZONS]

    if feature_df.empty or price_df.empty:
        feature_df["trade_date"] = pd.NaT
        feature_df["close"] = pd.NA
        for h in FORWARD_HORIZONS:
            feature_df[f"future_ret_{h}d"] = pd.NA
        for column in benchmark_columns + excess_columns:
            feature_df[column] = pd.NA
        return feature_df

    enriched_frames = []
    for symbol, frame in feature_df.groupby("symbol"):
        prices = price_df[price_df["symbol"] == symbol].copy().sort_values("trade_date")
        prices["ret_1d"] = prices["close"].pct_change()
        prices["past_ret_5d"] = prices["close"] / prices["close"].shift(5) - 1.0
        prices["past_ret_20d"] = prices["close"] / prices["close"].shift(20) - 1.0
        prices["past_ret_60d"] = prices["close"] / prices["close"].shift(60) - 1.0
        prices["volatility_20d"] = prices["ret_1d"].rolling(20).std()
        prices["volatility_60d"] = prices["ret_1d"].rolling(60).std()
        prices["volume_20d_avg"] = prices["volume"].rolling(20, min_periods=5).mean()
        prices["volume_shock_20d"] = prices["volume"] / prices["volume_20d_avg"]
        
        frame = frame.sort_values("date").copy()
        if prices.empty:
            frame["trade_date"] = pd.NaT
            frame["close"] = pd.NA
            for h in FORWARD_HORIZONS:
                frame[f"future_ret_{h}d"] = pd.NA
            frame["past_ret_5d"] = pd.NA
            frame["past_ret_20d"] = pd.NA
            frame["past_ret_60d"] = pd.NA
            frame["volatility_20d"] = pd.NA
            frame["volatility_60d"] = pd.NA
            frame["volume_20d_avg"] = pd.NA
            frame["volume_shock_20d"] = pd.NA
            for column in benchmark_columns + excess_columns:
                frame[column] = pd.NA
            enriched_frames.append(frame)
            continue

        trade_dates = prices["trade_date"].to_numpy()
        closes = prices["close"].to_numpy()
        idx = trade_dates.searchsorted(frame["date"].to_numpy(), side="left")

        mapped_trade_dates = []
        mapped_closes = []
        ret_by_horizon: dict[int, list] = {h: [] for h in FORWARD_HORIZONS}
        past_ret_5d = []
        past_ret_20d = []
        past_ret_60d = []
        vol_20d = []
        vol_60d = []
        volume_20d_avg = []
        volume_shock_20d = []

        past_ret_5_arr = prices["past_ret_5d"].to_numpy()
        past_ret_20_arr = prices["past_ret_20d"].to_numpy()
        past_ret_60_arr = prices["past_ret_60d"].to_numpy()
        vol_20_arr = prices["volatility_20d"].to_numpy()
        vol_60_arr = prices["volatility_60d"].to_numpy()
        vol_20d_avg_arr = prices["volume_20d_avg"].to_numpy()
        vol_shock_20d_arr = prices["volume_shock_20d"].to_numpy()

        for pos in idx:
            if pos >= len(prices):
                mapped_trade_dates.append(pd.NaT)
                mapped_closes.append(pd.NA)
                for h in FORWARD_HORIZONS:
                    ret_by_horizon[h].append(pd.NA)
                past_ret_5d.append(pd.NA)
                past_ret_20d.append(pd.NA)
                past_ret_60d.append(pd.NA)
                vol_20d.append(pd.NA)
                vol_60d.append(pd.NA)
                volume_20d_avg.append(pd.NA)
                volume_shock_20d.append(pd.NA)
                continue

            close_now = closes[pos]
            mapped_trade_dates.append(trade_dates[pos])
            mapped_closes.append(close_now)

            for h in FORWARD_HORIZONS:
                if pos + h >= len(prices):
                    ret_by_horizon[h].append(pd.NA)
                else:
                    ret_by_horizon[h].append(float(closes[pos + h] / close_now - 1.0))

            past_ret_5d.append(past_ret_5_arr[pos])
            past_ret_20d.append(past_ret_20_arr[pos])
            past_ret_60d.append(past_ret_60_arr[pos])
            vol_20d.append(vol_20_arr[pos])
            vol_60d.append(vol_60_arr[pos])
            volume_20d_avg.append(vol_20d_avg_arr[pos])
            volume_shock_20d.append(vol_shock_20d_arr[pos])

        frame["trade_date"] = mapped_trade_dates
        frame["close"] = mapped_closes
        for h in FORWARD_HORIZONS:
            frame[f"future_ret_{h}d"] = ret_by_horizon[h]
        frame["past_ret_5d"] = past_ret_5d
        frame["past_ret_20d"] = past_ret_20d
        frame["past_ret_60d"] = past_ret_60d
        frame["volatility_20d"] = vol_20d
        frame["volatility_60d"] = vol_60d
        frame["volume_20d_avg"] = volume_20d_avg
        frame["volume_shock_20d"] = volume_shock_20d

        for bench_symbol, bench_frame in benchmark_maps.items():
            prefix = bench_symbol.lower()
            mapped_keys = [ts.strftime("%Y-%m-%d") if pd.notna(ts) else None for ts in frame["trade_date"]]
            mapped = bench_frame.reindex(pd.Index(mapped_keys))
            for h in FORWARD_HORIZONS:
                frame[f"{prefix}_ret_{h}d"] = mapped[f"{prefix}_ret_{h}d"].to_list()

        benchmark_symbol = PRIMARY_BENCHMARK if PRIMARY_BENCHMARK in benchmark_maps else None
        if benchmark_symbol:
            prefix = benchmark_symbol.lower()
            frame["benchmark_symbol"] = benchmark_symbol
            for h in FORWARD_HORIZONS:
                frame[f"benchmark_ret_{h}d"] = frame[f"{prefix}_ret_{h}d"]
                frame[f"excess_ret_{h}d"] = frame[f"future_ret_{h}d"] - frame[f"benchmark_ret_{h}d"]
        else:
            frame["benchmark_symbol"] = pd.NA
            for h in FORWARD_HORIZONS:
                frame[f"benchmark_ret_{h}d"] = pd.NA
                frame[f"excess_ret_{h}d"] = pd.NA

        enriched_frames.append(frame)

    return pd.concat(enriched_frames, ignore_index=True)


def add_quality_score(feature_df: pd.DataFrame) -> pd.DataFrame:
    quality_components = {
        "full_ratio": 1.0,
        "unique_source_count": 1.0,
        "avg_content_length": 1.0,
        "extraction_failed_count": -1.0,
        "news_burst_20d": 0.5,
    }
    result = feature_df.copy()
    result["quality_score"] = 0.0
    result["quality_component_count"] = 0

    group_col = "trade_date" if "trade_date" in result.columns else "date"
    for _, idx in result.groupby(group_col).groups.items():
        group = result.loc[idx].copy()
        score = pd.Series(0.0, index=group.index, dtype="float64")
        count = pd.Series(0, index=group.index, dtype="int64")
        for col, weight in quality_components.items():
            if col not in group.columns:
                continue
            values = pd.to_numeric(group[col], errors="coerce")
            valid = values.notna()
            if valid.sum() == 0:
                continue
            ranked = values[valid].rank(method="average", pct=True)
            score.loc[valid.index] += (ranked - 0.5) * weight
            count.loc[valid.index] += 1
        valid_count = count > 0
        score.loc[valid_count] = score.loc[valid_count] / count.loc[valid_count]
        score.loc[~valid_count] = pd.NA
        result.loc[idx, "quality_score"] = score
        result.loc[idx, "quality_component_count"] = count
    return result


def _bucket_days(days: int | None) -> int | None:
    """Map a trading-day gap to an ordinal bucket (0=0-5, 1=6-15, 2=16-30, 3=31+)."""
    if days is None or (isinstance(days, float) and pd.isna(days)):
        return None
    if days <= 5:
        return 0
    if days <= 15:
        return 1
    if days <= 30:
        return 2
    return 3


def attach_earnings_event_features(
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        # existing
        "days_to_earnings",
        "days_since_earnings",
        "is_earnings_window_5d",
        "is_post_earnings_window_20d",
        # new: surprise
        "eps_estimate_last",
        "reported_eps_last",
        "surprise_pct_last",
        "is_positive_surprise",
        "is_negative_surprise",
        # new: timing buckets
        "days_to_earnings_bucket",
        "days_since_earnings_bucket",
        # new: tighter windows
        "is_pre_earnings_10d",
        "is_post_earnings_5d",
        "is_post_earnings_10d",
        # new: drift + surprise
        "is_post_positive_surprise_20d",
        "is_post_negative_surprise_20d",
        # layer 3: surprise magnitude buckets
        "surprise_bucket",
        # layer 3: news quality × earnings interaction
        "full_ratio_x_post_positive",
        "quality_score_x_post_negative",
        # layer 3: earnings recency decay
        "earnings_recency_weight",
    ]
    if feature_df.empty:
        for column in columns:
            feature_df[column] = pd.NA
        return feature_df

    if earnings_df.empty or price_df.empty:
        for column in columns:
            feature_df[column] = pd.NA
        return feature_df

    enriched_frames = []
    for symbol, frame in feature_df.groupby("symbol"):
        symbol_prices = (
            price_df[price_df["symbol"] == symbol]
            .sort_values("trade_date")
            .drop_duplicates(subset=["trade_date"], keep="last")
        )
        symbol_earnings = earnings_df[earnings_df["symbol"] == symbol].copy()

        frame = frame.sort_values("date").copy()
        if symbol_prices.empty or symbol_earnings.empty:
            for column in columns:
                frame[column] = pd.NA
            enriched_frames.append(frame)
            continue

        trade_dates = pd.DatetimeIndex(symbol_prices["trade_date"])

        # Build per-earnings-event lookup: trading-day position -> EPS/surprise data
        earnings_events_by_pos: dict[int, dict] = {}
        for _, earnings_row in symbol_earnings.iterrows():
            event_date = earnings_row["event_date"]
            pos = int(trade_dates.searchsorted(event_date, side="left"))
            if pos >= len(trade_dates):
                continue
            earnings_events_by_pos[pos] = {
                "eps_estimate": earnings_row.get("eps_estimate"),
                "reported_eps": earnings_row.get("reported_eps"),
                "surprise_pct": earnings_row.get("surprise_pct"),
            }

        aligned_positions = sorted(earnings_events_by_pos.keys())
        if not aligned_positions:
            for column in columns:
                frame[column] = pd.NA
            enriched_frames.append(frame)
            continue

        frame_trade_dates = pd.DatetimeIndex(frame["trade_date"])
        current_positions = trade_dates.searchsorted(frame_trade_dates, side="left")

        days_to: list = []
        days_since: list = []
        in_window: list = []
        post_window: list = []
        eps_estimate_last: list = []
        reported_eps_last: list = []
        surprise_pct_last: list = []
        is_positive_surprise: list = []
        is_negative_surprise: list = []
        days_to_bucket: list = []
        days_since_bucket: list = []
        pre_10d: list = []
        post_5d: list = []
        post_10d: list = []
        post_pos_surp: list = []
        post_neg_surp: list = []
        surprise_bucket: list = []
        full_ratio_x_post_positive: list = []
        quality_score_x_post_negative: list = []
        earnings_recency_weight: list = []

        for current_pos in current_positions:
            if current_pos >= len(trade_dates):
                for lst in (
                    days_to, days_since, in_window, post_window,
                    eps_estimate_last, reported_eps_last, surprise_pct_last,
                    is_positive_surprise, is_negative_surprise,
                    days_to_bucket, days_since_bucket,
                    pre_10d, post_5d, post_10d,
                    post_pos_surp, post_neg_surp,
                    surprise_bucket, full_ratio_x_post_positive,
                    quality_score_x_post_negative, earnings_recency_weight,
                ):
                    lst.append(pd.NA)
                continue

            next_pos = next((p for p in aligned_positions if p >= current_pos), None)
            prev_pos = next((p for p in reversed(aligned_positions) if p <= current_pos), None)

            next_gap = next_pos - current_pos if next_pos is not None else pd.NA
            prev_gap = current_pos - prev_pos if prev_pos is not None else pd.NA

            # Existing timing features
            days_to.append(int(next_gap) if pd.notna(next_gap) else pd.NA)
            days_since.append(int(prev_gap) if pd.notna(prev_gap) else pd.NA)
            in_window.append(int(pd.notna(next_gap) and next_gap <= 5))
            post_window.append(int(pd.notna(prev_gap) and 0 <= prev_gap <= 20))

            # Earnings surprise from most recent past event
            prev_event = earnings_events_by_pos.get(prev_pos, {}) if prev_pos is not None else {}
            eps_est = prev_event.get("eps_estimate")
            rep_eps = prev_event.get("reported_eps")
            surp_pct = prev_event.get("surprise_pct")

            eps_estimate_last.append(float(eps_est) if eps_est is not None and pd.notna(eps_est) else pd.NA)
            reported_eps_last.append(float(rep_eps) if rep_eps is not None and pd.notna(rep_eps) else pd.NA)
            surprise_pct_last.append(float(surp_pct) if surp_pct is not None and pd.notna(surp_pct) else pd.NA)

            surp_valid = surp_pct is not None and pd.notna(surp_pct)
            is_positive_surprise.append(int(surp_valid and float(surp_pct) > 0))
            is_negative_surprise.append(int(surp_valid and float(surp_pct) < 0))

            # Timing buckets
            days_to_bucket.append(_bucket_days(int(next_gap) if pd.notna(next_gap) else None))
            days_since_bucket.append(_bucket_days(int(prev_gap) if pd.notna(prev_gap) else None))

            # Tighter pre/post windows
            pre_10d.append(int(pd.notna(next_gap) and next_gap <= 10))
            post_5d.append(int(pd.notna(prev_gap) and 0 <= prev_gap <= 5))
            post_10d.append(int(pd.notna(prev_gap) and 0 <= prev_gap <= 10))

            # Post-earnings drift combined with surprise sign
            is_post_20d = pd.notna(prev_gap) and 0 <= prev_gap <= 20
            post_pos_surp.append(int(is_post_20d and surp_valid and float(surp_pct) > 0))
            post_neg_surp.append(int(is_post_20d and surp_valid and float(surp_pct) < 0))

            # Layer 3: surprise magnitude bucket
            # -2=large miss(<-5%), -1=small miss(-5%~0), 1=small beat(0~5%), 2=large beat(>5%)
            if surp_valid:
                sp = float(surp_pct)
                if sp < -5.0:
                    sbucket = -2
                elif sp < 0.0:
                    sbucket = -1
                elif sp <= 5.0:
                    sbucket = 1
                else:
                    sbucket = 2
            else:
                sbucket = pd.NA
            surprise_bucket.append(sbucket)

            # Layer 3: news quality × earnings interaction
            row_idx = len(surprise_bucket) - 1
            fr_col = frame["full_ratio"].to_numpy() if "full_ratio" in frame.columns else None
            qs_col = frame["quality_score"].to_numpy() if "quality_score" in frame.columns else None
            fr = fr_col[row_idx] if fr_col is not None and row_idx < len(fr_col) else pd.NA
            qs = qs_col[row_idx] if qs_col is not None and row_idx < len(qs_col) else pd.NA
            is_pp20 = int(is_post_20d and surp_valid and float(surp_pct) > 0)
            is_pn20 = int(is_post_20d and surp_valid and float(surp_pct) < 0)
            full_ratio_x_post_positive.append(float(fr) * is_pp20 if pd.notna(fr) else pd.NA)
            quality_score_x_post_negative.append(float(qs) * is_pn20 if pd.notna(qs) else pd.NA)

            # Layer 3: earnings recency decay — exp(-days_since / 20)
            if pd.notna(prev_gap):
                earnings_recency_weight.append(float(math.exp(-int(prev_gap) / 20.0)))
            else:
                earnings_recency_weight.append(pd.NA)

        frame["days_to_earnings"] = days_to
        frame["days_since_earnings"] = days_since
        frame["is_earnings_window_5d"] = in_window
        frame["is_post_earnings_window_20d"] = post_window
        frame["eps_estimate_last"] = eps_estimate_last
        frame["reported_eps_last"] = reported_eps_last
        frame["surprise_pct_last"] = surprise_pct_last
        frame["is_positive_surprise"] = is_positive_surprise
        frame["is_negative_surprise"] = is_negative_surprise
        frame["days_to_earnings_bucket"] = days_to_bucket
        frame["days_since_earnings_bucket"] = days_since_bucket
        frame["is_pre_earnings_10d"] = pre_10d
        frame["is_post_earnings_5d"] = post_5d
        frame["is_post_earnings_10d"] = post_10d
        frame["is_post_positive_surprise_20d"] = post_pos_surp
        frame["is_post_negative_surprise_20d"] = post_neg_surp
        frame["surprise_bucket"] = surprise_bucket
        frame["full_ratio_x_post_positive"] = full_ratio_x_post_positive
        frame["quality_score_x_post_negative"] = quality_score_x_post_negative
        frame["earnings_recency_weight"] = earnings_recency_weight
        enriched_frames.append(frame)

    return pd.concat(enriched_frames, ignore_index=True)


_STRENGTH_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}
_EVENT_TYPES = ["earnings", "product", "MA", "regulation", "macro", "competition", "other"]


def load_llm_sentiment_frame() -> pd.DataFrame:
    _, end_date, effective_start = _resolve_range()
    client = create_client()
    col = client[DB_NAME][LLM_COLLECTION]

    cursor = col.find(
        {"llm_sentiment_final": {"$exists": True}},
        {
            "_id": 0,
            "symbol": 1,
            "date": 1,
            "llm_sentiment_final": 1,
            "llm_signal_strength_a": 1,
            "llm_event_type_a": 1,
            "llm_disagreement": 1,
        },
    )

    rows = []
    for doc in cursor:
        news_date = _extract_news_date(doc)
        if not news_date:
            continue
        date_ts = pd.Timestamp(news_date)
        if effective_start is not None and date_ts < effective_start:
            continue
        if end_date is not None and date_ts > end_date:
            continue
        rows.append({
            "symbol": doc.get("symbol"),
            "date": date_ts,
            "sentiment": float(doc.get("llm_sentiment_final", 0.0)),
            "strength": doc.get("llm_signal_strength_a", "low"),
            "event_type": doc.get("llm_event_type_a", "other"),
            "disagreement": float(doc.get("llm_disagreement", 0.0)),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["symbol", "date", "sentiment", "strength", "event_type", "disagreement"]
    )


def aggregate_llm_sentiment_features(llm_df: pd.DataFrame) -> pd.DataFrame:
    if llm_df.empty:
        return pd.DataFrame(columns=["symbol", "date"])

    llm_df = llm_df.copy()
    llm_df["weight"] = llm_df["strength"].map(_STRENGTH_WEIGHT).fillna(0.3)

    def _weighted_mean(grp: pd.DataFrame) -> float:
        w = grp["weight"]
        s = grp["sentiment"]
        return float((s * w).sum() / w.sum()) if w.sum() > 0 else float(s.mean())

    # ── daily aggregates ──────────────────────────────────────────────────────
    daily_rows = []
    for (symbol, date), grp in llm_df.groupby(["symbol", "date"]):
        sent = grp["sentiment"]
        ev = grp["event_type"]
        daily_rows.append({
            "symbol": symbol,
            "date": date,
            "sent_wavg":       _weighted_mean(grp),
            "sent_std":        float(sent.std()) if len(sent) > 1 else 0.0,
            "high_signal_n":   int((grp["strength"] == "high").sum()),
            "has_earnings":    int((ev == "earnings").any()),
            "has_regulation":  int((ev == "regulation").any()),
            "has_ma":          int((ev == "MA").any()),
            "disagreement_mean": float(grp["disagreement"].mean()),
            "earnings_beat_n": int(((ev == "earnings") & (sent > 0.2)).sum()),
            "earnings_miss_n": int(((ev == "earnings") & (sent < -0.2)).sum()),
            "negative_n":      int((sent < -0.1).sum()),
        })

    daily = pd.DataFrame(daily_rows).sort_values(["symbol", "date"])

    # ── rolling features per symbol ───────────────────────────────────────────
    def _rollups(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp.sort_values("date").copy()
        s = g["sent_wavg"]
        g["avg_sentiment_3d"]       = s.rolling(3, min_periods=1).mean()
        g["avg_sentiment_5d"]       = s.rolling(5, min_periods=1).mean()
        g["sentiment_shift_5d"]     = s - s.shift(5)
        g["high_signal_count_3d"]   = g["high_signal_n"].rolling(3, min_periods=1).sum()
        g["has_regulatory_risk_5d"] = g["has_regulation"].rolling(5, min_periods=1).max()
        g["earnings_beat_signal"]   = (g["earnings_beat_n"] > 0).astype(int).rolling(5, min_periods=1).max()
        g["earnings_miss_signal"]   = (g["earnings_miss_n"] > 0).astype(int).rolling(5, min_periods=1).max()
        g["disagreement_avg_5d"]    = g["disagreement_mean"].rolling(5, min_periods=1).mean()
        g["negative_event_count_5d"]= g["negative_n"].rolling(5, min_periods=1).sum()
        return g

    # Iterate groups (not .apply()) so 'symbol' survives under pandas 3.0.
    parts = [_rollups(grp) for _, grp in daily.groupby("symbol", group_keys=False)]
    result = pd.concat(parts, ignore_index=True) if parts else daily
    return result.reset_index(drop=True)


def attach_llm_sentiment_features(
    feature_df: pd.DataFrame, llm_features: pd.DataFrame
) -> pd.DataFrame:
    _LLM_COLS = [
        "avg_sentiment_3d", "avg_sentiment_5d", "sentiment_shift_5d",
        "high_signal_count_3d", "has_regulatory_risk_5d",
        "earnings_beat_signal", "earnings_miss_signal",
        "disagreement_avg_5d", "negative_event_count_5d",
    ]
    if llm_features.empty or feature_df.empty:
        for col in _LLM_COLS:
            feature_df[col] = pd.NA
        return feature_df

    llm_slim = llm_features[["symbol", "date"] + _LLM_COLS]
    merged = feature_df.merge(llm_slim, on=["symbol", "date"], how="left")
    return merged


def save_features(feature_df: pd.DataFrame) -> int:
    base_start, _, _ = _resolve_range()
    if base_start is not None:
        feature_df = feature_df[feature_df["date"] >= base_start].copy()

    if feature_df.empty:
        return 0

    client = create_client()
    col = client[DB_NAME][FEATURE_COLLECTION]
    col.create_index([("symbol", 1), ("date", 1)], unique=True)

    if base_start is not None:
        col.delete_many({"date": {"$gte": base_start.date().isoformat()}})
    else:
        col.delete_many({})

    ops = []
    built_at = datetime.now(UTC).isoformat()
    # Load sector mapping
    universe_col = client[DB_NAME][UNIVERSE_COLLECTION]
    sector_map = {doc["symbol"]: doc.get("sector", "Unknown") for doc in universe_col.find({}, {"symbol": 1, "sector": 1})}

    for row in feature_df.to_dict("records"):
        symbol = row["symbol"]
        record = {
            "symbol": symbol,
            "name": row["name"],
            "sector": sector_map.get(symbol, "Unknown"),
            "date": row["date"].date().isoformat(),
            "article_count": int(row["article_count"]),
            "full_count": int(row["full_count"]),
            "title_only_count": int(row["title_only_count"]),
            "url_only_count": int(row["url_only_count"]),
            "full_ratio": float(row["full_ratio"]) if pd.notna(row["full_ratio"]) else None,
            "quality_score": float(row["quality_score"]) if pd.notna(row.get("quality_score")) else None,
            "title_only_ratio": float(row["title_only_ratio"]) if pd.notna(row["title_only_ratio"]) else None,
            "avg_content_length": float(row["avg_content_length"]) if pd.notna(row["avg_content_length"]) else 0.0,
            "max_content_length": int(row["max_content_length"]) if pd.notna(row["max_content_length"]) else 0,
            "unique_url_count": int(row["unique_url_count"]),
            "unique_source_count": int(row["unique_source_count"]),
            "unique_platform_count": int(row["unique_platform_count"]),
            "extraction_failed_count": int(row["extraction_failed_count"]),
            "timeout_fallback_count": int(row["timeout_fallback_count"]),
            "news_count_3d": float(row["news_count_3d"]),
            "news_count_5d": float(row["news_count_5d"]),
            "news_count_20d": float(row["news_count_20d"]),
            "full_count_5d": float(row["full_count_5d"]),
            "avg_full_ratio_5d": float(row["avg_full_ratio_5d"]) if pd.notna(row["avg_full_ratio_5d"]) else None,
            "news_burst_20d": float(row["news_burst_20d"]) if pd.notna(row["news_burst_20d"]) else None,
            "trade_date": row["trade_date"].date().isoformat() if pd.notna(row["trade_date"]) else None,
            "close": float(row["close"]) if pd.notna(row["close"]) else None,
            "past_ret_5d": float(row["past_ret_5d"]) if pd.notna(row.get("past_ret_5d")) else None,
            "past_ret_20d": float(row["past_ret_20d"]) if pd.notna(row.get("past_ret_20d")) else None,
            "past_ret_60d": float(row["past_ret_60d"]) if pd.notna(row.get("past_ret_60d")) else None,
            "volatility_20d": float(row["volatility_20d"]) if pd.notna(row.get("volatility_20d")) else None,
            "volatility_60d": float(row["volatility_60d"]) if pd.notna(row.get("volatility_60d")) else None,
            "volume_20d_avg": float(row["volume_20d_avg"]) if pd.notna(row.get("volume_20d_avg")) else None,
            "volume_shock_20d": float(row["volume_shock_20d"]) if pd.notna(row.get("volume_shock_20d")) else None,
            "days_to_earnings": int(row["days_to_earnings"]) if pd.notna(row.get("days_to_earnings")) else None,
            "days_since_earnings": int(row["days_since_earnings"]) if pd.notna(row.get("days_since_earnings")) else None,
            "is_earnings_window_5d": int(row["is_earnings_window_5d"]) if pd.notna(row.get("is_earnings_window_5d")) else None,
            "is_post_earnings_window_20d": int(row["is_post_earnings_window_20d"]) if pd.notna(row.get("is_post_earnings_window_20d")) else None,
            "eps_estimate_last": float(row["eps_estimate_last"]) if pd.notna(row.get("eps_estimate_last")) else None,
            "reported_eps_last": float(row["reported_eps_last"]) if pd.notna(row.get("reported_eps_last")) else None,
            "surprise_pct_last": float(row["surprise_pct_last"]) if pd.notna(row.get("surprise_pct_last")) else None,
            "is_positive_surprise": int(row["is_positive_surprise"]) if pd.notna(row.get("is_positive_surprise")) else None,
            "is_negative_surprise": int(row["is_negative_surprise"]) if pd.notna(row.get("is_negative_surprise")) else None,
            "days_to_earnings_bucket": int(row["days_to_earnings_bucket"]) if pd.notna(row.get("days_to_earnings_bucket")) else None,
            "days_since_earnings_bucket": int(row["days_since_earnings_bucket"]) if pd.notna(row.get("days_since_earnings_bucket")) else None,
            "is_pre_earnings_10d": int(row["is_pre_earnings_10d"]) if pd.notna(row.get("is_pre_earnings_10d")) else None,
            "is_post_earnings_5d": int(row["is_post_earnings_5d"]) if pd.notna(row.get("is_post_earnings_5d")) else None,
            "is_post_earnings_10d": int(row["is_post_earnings_10d"]) if pd.notna(row.get("is_post_earnings_10d")) else None,
            "is_post_positive_surprise_20d": int(row["is_post_positive_surprise_20d"]) if pd.notna(row.get("is_post_positive_surprise_20d")) else None,
            "is_post_negative_surprise_20d": int(row["is_post_negative_surprise_20d"]) if pd.notna(row.get("is_post_negative_surprise_20d")) else None,
            "surprise_bucket": int(row["surprise_bucket"]) if pd.notna(row.get("surprise_bucket")) else None,
            "full_ratio_x_post_positive": float(row["full_ratio_x_post_positive"]) if pd.notna(row.get("full_ratio_x_post_positive")) else None,
            "quality_score_x_post_negative": float(row["quality_score_x_post_negative"]) if pd.notna(row.get("quality_score_x_post_negative")) else None,
            "earnings_recency_weight": float(row["earnings_recency_weight"]) if pd.notna(row.get("earnings_recency_weight")) else None,
            # LLM sentiment features (Stage 3.5.2)
            "avg_sentiment_3d":        float(row["avg_sentiment_3d"])        if pd.notna(row.get("avg_sentiment_3d"))        else None,
            "avg_sentiment_5d":        float(row["avg_sentiment_5d"])        if pd.notna(row.get("avg_sentiment_5d"))        else None,
            "sentiment_shift_5d":      float(row["sentiment_shift_5d"])      if pd.notna(row.get("sentiment_shift_5d"))      else None,
            "high_signal_count_3d":    float(row["high_signal_count_3d"])    if pd.notna(row.get("high_signal_count_3d"))    else None,
            "has_regulatory_risk_5d":  int(row["has_regulatory_risk_5d"])    if pd.notna(row.get("has_regulatory_risk_5d"))  else None,
            "earnings_beat_signal":    int(row["earnings_beat_signal"])       if pd.notna(row.get("earnings_beat_signal"))    else None,
            "earnings_miss_signal":    int(row["earnings_miss_signal"])       if pd.notna(row.get("earnings_miss_signal"))    else None,
            "disagreement_avg_5d":     float(row["disagreement_avg_5d"])     if pd.notna(row.get("disagreement_avg_5d"))     else None,
            "negative_event_count_5d": float(row["negative_event_count_5d"]) if pd.notna(row.get("negative_event_count_5d")) else None,
            "benchmark_symbol": row.get("benchmark_symbol") if pd.notna(row.get("benchmark_symbol")) else None,
            "builtAt": built_at,
        }
        for h in FORWARD_HORIZONS:
            record[f"future_ret_{h}d"] = float(row[f"future_ret_{h}d"]) if pd.notna(row.get(f"future_ret_{h}d")) else None
            record[f"benchmark_ret_{h}d"] = float(row[f"benchmark_ret_{h}d"]) if pd.notna(row.get(f"benchmark_ret_{h}d")) else None
            record[f"excess_ret_{h}d"] = float(row[f"excess_ret_{h}d"]) if pd.notna(row.get(f"excess_ret_{h}d")) else None
        for symbol in BENCHMARK_SYMBOLS:
            prefix = symbol.lower()
            for h in FORWARD_HORIZONS:
                key = f"{prefix}_ret_{h}d"
                record[key] = float(row[key]) if pd.notna(row.get(key)) else None
        for mc in MACRO_FEATURE_COLS:
            v = row.get(mc)
            record[mc] = float(v) if pd.notna(v) else None
        for pc in PREMARKET_FEATURE_COLS:
            v = row.get(pc)
            record[pc] = float(v) if pd.notna(v) else None
        for ac in ANALYST_FEATURE_COLS:
            v = row.get(ac)
            record[ac] = float(v) if pd.notna(v) else None
        for ic in INST13F_FEATURE_COLS:
            v = row.get(ic)
            record[ic] = float(v) if pd.notna(v) else None
        for rc in RETAIL_FEATURE_COLS:
            v = row.get(rc)
            record[rc] = float(v) if pd.notna(v) else None
        ops.append(
            UpdateOne(
                {"symbol": record["symbol"], "date": record["date"]},
                {"$set": record},
                upsert=True,
            )
        )

    if not ops:
        return 0

    result = col.bulk_write(ops, ordered=False)
    return result.upserted_count + result.modified_count


MACRO_COLLECTION = os.getenv("MACRO_COLLECTION", "macro_indicators")
MACRO_FEATURE_COLS = [
    "macro_vix", "macro_vix_change_5d", "macro_vix_pctile_252d", "macro_is_high_vol",
    "macro_tnx", "macro_tnx_change_20d", "macro_dxy_change_20d",
    "macro_spy_above_200ma", "macro_spy_ret_20d", "macro_risk_on",
]


def load_macro_frame() -> pd.DataFrame:
    """Load the macro_indicators time series (vix/tnx/dxy/spy_close by date)."""
    client = create_client()
    col = client[DB_NAME][MACRO_COLLECTION]
    rows = list(col.find({}, {"_id": 0, "date": 1, "vix": 1, "tnx": 1, "dxy": 1, "spy_close": 1}))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = df["date"].astype(str).str[:10]
    return df.sort_values("date").reset_index(drop=True)


def compute_regime_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Market-level regime features (one row per trading date)."""
    if macro_df.empty:
        return pd.DataFrame(columns=["trade_date"])
    m = macro_df.sort_values("date").copy()
    for c in ["vix", "tnx", "dxy", "spy_close"]:
        m[c] = pd.to_numeric(m.get(c), errors="coerce")

    m["macro_vix"] = m["vix"]
    m["macro_vix_change_5d"] = m["vix"] - m["vix"].shift(5)
    m["macro_vix_pctile_252d"] = m["vix"].rolling(252, min_periods=30).apply(
        lambda s: float((s <= s[-1]).mean()), raw=True)
    m["macro_is_high_vol"] = (m["macro_vix_pctile_252d"] > 0.8).astype("Int64")

    m["macro_tnx"] = m["tnx"]
    m["macro_tnx_change_20d"] = m["tnx"] - m["tnx"].shift(20)
    m["macro_dxy_change_20d"] = (m["dxy"] / m["dxy"].shift(20)) - 1.0

    ma200 = m["spy_close"].rolling(200, min_periods=50).mean()
    m["macro_spy_above_200ma"] = (m["spy_close"] > ma200).astype("Int64")
    m["macro_spy_ret_20d"] = (m["spy_close"] / m["spy_close"].shift(20)) - 1.0
    # risk-on = uptrend AND not in a volatility spike
    m["macro_risk_on"] = ((m["macro_spy_above_200ma"] == 1) & (m["macro_is_high_vol"] != 1)).astype("Int64")

    cols = ["date", "macro_vix", "macro_vix_change_5d", "macro_vix_pctile_252d",
            "macro_is_high_vol", "macro_tnx", "macro_tnx_change_20d", "macro_dxy_change_20d",
            "macro_spy_above_200ma", "macro_spy_ret_20d", "macro_risk_on"]
    return m[cols].rename(columns={"date": "trade_date"})


def attach_macro_regime_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Broadcast market-level regime features onto every symbol-date row.

    Merges on a temporary string key so the original trade_date dtype
    (a Timestamp, relied on by save_features) is preserved.
    """
    regime = compute_regime_features(load_macro_frame())
    if regime.empty or "trade_date" not in feature_df.columns:
        return feature_df
    feature_df = feature_df.copy()
    feature_df["_macro_key"] = feature_df["trade_date"].astype(str).str[:10]
    regime = regime.rename(columns={"trade_date": "_macro_key"})
    regime["_macro_key"] = regime["_macro_key"].astype(str).str[:10]
    merged = feature_df.merge(regime, on="_macro_key", how="left")
    return merged.drop(columns=["_macro_key"])


PREMARKET_COLLECTION = os.getenv("PREMARKET_COLLECTION", "premarket_signals")
PREMARKET_FEATURE_COLS = [
    "pm_gap", "ah_gap", "ah_volume_ratio",
]


def load_premarket_frame() -> pd.DataFrame:
    """Load premarket_signals (last ~30 days, per symbol per date)."""
    client = create_client()
    col = client[DB_NAME][PREMARKET_COLLECTION]
    rows = list(col.find(
        {},
        {"_id": 0, "symbol": 1, "trade_date": 1,
         "pm_gap": 1, "ah_gap": 1, "ah_volume_ratio": 1},
    ))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    for c in PREMARKET_FEATURE_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df


def attach_premarket_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join per-symbol premarket/after-hours gap features onto feature_df.

    Uses a composite string key (symbol|date) to avoid Timestamp dtype issues,
    same pattern as attach_macro_regime_features.
    """
    pm_df = load_premarket_frame()
    if pm_df.empty or "trade_date" not in feature_df.columns:
        return feature_df
    feature_df = feature_df.copy()
    feature_df["_pm_key"] = (
        feature_df["symbol"].astype(str) + "|"
        + feature_df["trade_date"].astype(str).str[:10]
    )
    pm_df["_pm_key"] = (
        pm_df["symbol"].astype(str) + "|"
        + pm_df["trade_date"].astype(str).str[:10]
    )
    merged = feature_df.merge(
        pm_df[["_pm_key"] + PREMARKET_FEATURE_COLS], on="_pm_key", how="left"
    )
    return merged.drop(columns=["_pm_key"])


ANALYST_COLLECTION = os.getenv("ANALYST_COLLECTION", "analyst_consensus")
ANALYST_FEATURE_COLS = [
    "analyst_buy_ratio", "analyst_sell_ratio",
    "analyst_consensus_score", "analyst_buy_ratio_chg_1m",
]


def load_analyst_frame() -> pd.DataFrame:
    """Load analyst_consensus (monthly per symbol) from MongoDB."""
    client = create_client()
    col = client[DB_NAME][ANALYST_COLLECTION]
    rows = list(col.find(
        {},
        {"_id": 0, "symbol": 1, "period": 1,
         "analyst_buy_ratio": 1, "analyst_sell_ratio": 1,
         "analyst_consensus_score": 1, "analyst_buy_ratio_chg_1m": 1},
    ))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["period"] = df["period"].astype(str).str[:7]  # "YYYY-MM"
    for c in ANALYST_FEATURE_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df


def attach_analyst_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join monthly analyst consensus onto feature_df.

    Matches each trade_date to its calendar month (YYYY-MM) → avoids lookahead.
    Uses composite key (symbol|YYYY-MM) to preserve trade_date dtype.
    """
    analyst_df = load_analyst_frame()
    if analyst_df.empty or "trade_date" not in feature_df.columns:
        return feature_df
    feature_df = feature_df.copy()
    # Build join key: symbol + YYYY-MM of trade_date
    feature_df["_ana_key"] = (
        feature_df["symbol"].astype(str) + "|"
        + feature_df["trade_date"].astype(str).str[:7]
    )
    analyst_df["_ana_key"] = (
        analyst_df["symbol"].astype(str) + "|"
        + analyst_df["period"].astype(str).str[:7]
    )
    merged = feature_df.merge(
        analyst_df[["_ana_key"] + ANALYST_FEATURE_COLS], on="_ana_key", how="left"
    )
    return merged.drop(columns=["_ana_key"])


INST13F_COLLECTION = os.getenv("INST13F_COLLECTION", "inst_13f_holdings")
INST13F_FEATURE_COLS = ["inst_holding_pct", "inst_holding_pct_chg"]

# M.7 — disabled by default because this factor was reading the future.
#
# `inst_13f_holdings` holds one row per symbol with no date, quarter or filing date, and
# the join below broadcasts it to every trade_date. Measured on the built store: 79 of the
# 100 symbols carry a single value across all 8 years and the other 21 carry two (the
# second being null). So this was never a time series — it is a constant per symbol, which
# after rank-normalisation adds a fixed score offset to that symbol across all of history.
#
# The constant is "which symbols institutions were accumulating in 2026", applied to rows
# dated 2018. That is hindsight momentum with the answer written into the input, which is
# why it measured IC +0.198 — the strongest factor in the platform — while carrying up to
# 1.2 weight in the regime scorer.
#
# Re-enabling requires point-in-time 13F: filings joined as-of their FILING date, not their
# reporting period, since the 45-day statutory lag is exactly the information that must not
# leak. `inst_13f_raw` currently holds only 3 quarters (2025-09 … 2026-03) against a feature
# store spanning 2018-2026, so the history does not exist yet and must be backfilled from
# EDGAR first.
ENABLE_INST13F = os.getenv("ENABLE_INST13F_FEATURES", "false").lower() == "true"


def load_inst13f_frame() -> pd.DataFrame:
    """Load institutional 13F holding features (one row per symbol, latest quarter)."""
    client = create_client()
    col = client[DB_NAME][INST13F_COLLECTION]
    rows = list(col.find(
        {},
        {"_id": 0, "symbol": 1, "inst_holding_pct": 1, "inst_holding_pct_chg": 1},
    ))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in INST13F_FEATURE_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df


def attach_inst13f_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Join institutional holding features onto feature_df by symbol.

    Off unless ENABLE_INST13F_FEATURES=true — see the note on that flag. The join is by
    symbol alone, which broadcasts one value across every trade_date, so with only the
    latest quarter available it hands 2026 information to rows dated 2018.

    Left as a no-op rather than deleted: the shape is correct once `inst_13f_raw` carries
    filing-dated history and this becomes an as-of join, and keeping the seam makes that a
    smaller change than reintroducing the feature from scratch.
    """
    if not ENABLE_INST13F:
        return feature_df
    inst_df = load_inst13f_frame()
    if inst_df.empty or "symbol" not in feature_df.columns:
        return feature_df
    return feature_df.merge(inst_df[["symbol"] + INST13F_FEATURE_COLS], on="symbol", how="left")


RETAIL_COLLECTION = os.getenv("RETAIL_COLLECTION", "retail_sentiment")
RETAIL_FEATURE_COLS = [
    "retail_msg_count",
    "retail_bull_ratio",
    "retail_sent_score",
    "retail_sentiment_divergence",
]


def load_retail_frame() -> pd.DataFrame:
    """Load retail sentiment features (one row per symbol+date) from StockTwits."""
    client = create_client()
    col = client[DB_NAME][RETAIL_COLLECTION]
    rows = list(col.find(
        {},
        {"_id": 0, "symbol": 1, "date": 1,
         "retail_msg_count": 1, "retail_bull_ratio": 1, "retail_sent_score": 1},
    ))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ["retail_msg_count", "retail_bull_ratio", "retail_sent_score"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df


def attach_retail_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join retail sentiment features onto feature_df by symbol+date.

    Also derives retail_sentiment_divergence = retail_sent_score - avg_sentiment_3d
    (retail crowd vs LLM-scored institutional news, range roughly -2 to +2).
    """
    retail_df = load_retail_frame()
    if retail_df.empty or "symbol" not in feature_df.columns:
        for c in RETAIL_FEATURE_COLS:
            if c not in feature_df.columns:
                feature_df[c] = None
        return feature_df

    feature_df = feature_df.copy()
    feature_df["_ret_key"] = (
        feature_df["symbol"].astype(str)
        + "|"
        + feature_df["trade_date"].astype(str).str[:10]
    )
    retail_df["_ret_key"] = (
        retail_df["symbol"].astype(str)
        + "|"
        + retail_df["date"].astype(str).str[:10]
    )
    merged = feature_df.merge(
        retail_df[["_ret_key", "retail_msg_count", "retail_bull_ratio", "retail_sent_score"]],
        on="_ret_key",
        how="left",
    ).drop(columns=["_ret_key"])

    if "avg_sentiment_3d" in merged.columns:
        merged["retail_sentiment_divergence"] = (
            merged["retail_sent_score"] - merged["avg_sentiment_3d"]
        )
    else:
        merged["retail_sentiment_divergence"] = None

    return merged


def build_daily_symbol_features() -> int:
    print("=== Building daily symbol features ===")
    news_df = load_news_frame()
    if news_df.empty:
        print("No news records available for feature build.")
        return 0
    print(f"Loaded {len(news_df):,} news rows")

    feature_df = aggregate_news_features(news_df)
    print(f"Aggregated {len(feature_df):,} symbol-date rows")

    price_df = load_price_frame()
    print(f"Loaded {len(price_df):,} price rows")

    feature_df = attach_price_labels(feature_df, price_df)
    feature_df = add_quality_score(feature_df)
    print(f"Computed quality_score")
    earnings_df = load_earnings_frame()
    print(f"Loaded {len(earnings_df):,} earnings rows")
    feature_df = attach_earnings_event_features(feature_df, price_df, earnings_df)

    llm_df = load_llm_sentiment_frame()
    print(f"Loaded {len(llm_df):,} LLM-enriched article rows")
    llm_features = aggregate_llm_sentiment_features(llm_df)
    feature_df = attach_llm_sentiment_features(feature_df, llm_features)
    print(f"Attached LLM sentiment features")

    feature_df = attach_macro_regime_features(feature_df)
    print(f"Attached macro regime features")

    feature_df = attach_premarket_features(feature_df)
    print(f"Attached premarket/after-hours gap features")

    feature_df = attach_analyst_features(feature_df)
    print(f"Attached analyst consensus features")

    feature_df = attach_inst13f_features(feature_df)
    print(f"Attached institutional 13F holding features")

    feature_df = attach_retail_features(feature_df)
    print(f"Attached retail sentiment features (StockTwits)")

    saved = save_features(feature_df)
    print(f"Saved {saved:,} feature rows to {FEATURE_COLLECTION}")
    return saved


if __name__ == "__main__":
    build_daily_symbol_features()
