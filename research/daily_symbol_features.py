#!/usr/bin/env python3
"""Build daily symbol-level research features from news and price history."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
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
BENCHMARK_SYMBOLS = [
    symbol.strip().upper()
    for symbol in os.getenv("FEATURE_BENCHMARK_SYMBOLS", "SPY,QQQ").split(",")
    if symbol.strip()
]
PRIMARY_BENCHMARK = os.getenv("FEATURE_PRIMARY_BENCHMARK", "QQQ").strip().upper()

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
        rows.append(
            {
                "symbol": doc.get("symbol"),
                "name": doc.get("name"),
                "date": date_ts,
                "data_quality": doc.get("data_quality") or "unknown",
                "content_length": int(doc.get("content_length") or 0),
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

    grouped = grouped.groupby("symbol", group_keys=True).apply(_add_rollups)
    grouped = grouped.reset_index(level=0)
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

        bench[f"{symbol.lower()}_ret_5d"] = [_forward_return(i, 5) for i in range(len(bench))]
        bench[f"{symbol.lower()}_ret_20d"] = [_forward_return(i, 20) for i in range(len(bench))]
        bench[f"{symbol.lower()}_ret_60d"] = [_forward_return(i, 60) for i in range(len(bench))]

        bench["trade_date_key"] = bench["trade_date"].dt.strftime("%Y-%m-%d")
        benchmark_maps[symbol] = bench.set_index("trade_date_key")[
            [f"{symbol.lower()}_ret_5d", f"{symbol.lower()}_ret_20d", f"{symbol.lower()}_ret_60d"]
        ]

    return benchmark_maps


def attach_price_labels(feature_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    benchmark_maps = build_benchmark_return_maps(price_df)
    benchmark_columns = []
    for symbol in BENCHMARK_SYMBOLS:
        benchmark_columns.extend(
            [f"{symbol.lower()}_ret_5d", f"{symbol.lower()}_ret_20d", f"{symbol.lower()}_ret_60d"]
        )
    excess_columns = [
        "benchmark_symbol",
        "benchmark_ret_5d",
        "benchmark_ret_20d",
        "benchmark_ret_60d",
        "excess_ret_5d",
        "excess_ret_20d",
        "excess_ret_60d",
    ]

    if feature_df.empty or price_df.empty:
        feature_df["trade_date"] = pd.NaT
        feature_df["close"] = pd.NA
        feature_df["future_ret_5d"] = pd.NA
        feature_df["future_ret_20d"] = pd.NA
        feature_df["future_ret_60d"] = pd.NA
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
            frame["future_ret_5d"] = pd.NA
            frame["future_ret_20d"] = pd.NA
            frame["future_ret_60d"] = pd.NA
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
        ret_5d = []
        ret_20d = []
        ret_60d = []
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
                ret_5d.append(pd.NA)
                ret_20d.append(pd.NA)
                ret_60d.append(pd.NA)
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

            def _forward_return(offset: int):
                if pos + offset >= len(prices):
                    return pd.NA
                return float(closes[pos + offset] / close_now - 1.0)

            ret_5d.append(_forward_return(5))
            ret_20d.append(_forward_return(20))
            ret_60d.append(_forward_return(60))
            
            past_ret_5d.append(past_ret_5_arr[pos])
            past_ret_20d.append(past_ret_20_arr[pos])
            past_ret_60d.append(past_ret_60_arr[pos])
            vol_20d.append(vol_20_arr[pos])
            vol_60d.append(vol_60_arr[pos])
            volume_20d_avg.append(vol_20d_avg_arr[pos])
            volume_shock_20d.append(vol_shock_20d_arr[pos])

        frame["trade_date"] = mapped_trade_dates
        frame["close"] = mapped_closes
        frame["future_ret_5d"] = ret_5d
        frame["future_ret_20d"] = ret_20d
        frame["future_ret_60d"] = ret_60d
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
            frame[f"{prefix}_ret_5d"] = mapped[f"{prefix}_ret_5d"].to_list()
            frame[f"{prefix}_ret_20d"] = mapped[f"{prefix}_ret_20d"].to_list()
            frame[f"{prefix}_ret_60d"] = mapped[f"{prefix}_ret_60d"].to_list()

        benchmark_symbol = PRIMARY_BENCHMARK if PRIMARY_BENCHMARK in benchmark_maps else None
        if benchmark_symbol:
            prefix = benchmark_symbol.lower()
            frame["benchmark_symbol"] = benchmark_symbol
            frame["benchmark_ret_5d"] = frame[f"{prefix}_ret_5d"]
            frame["benchmark_ret_20d"] = frame[f"{prefix}_ret_20d"]
            frame["benchmark_ret_60d"] = frame[f"{prefix}_ret_60d"]
            frame["excess_ret_5d"] = frame["future_ret_5d"] - frame["benchmark_ret_5d"]
            frame["excess_ret_20d"] = frame["future_ret_20d"] - frame["benchmark_ret_20d"]
            frame["excess_ret_60d"] = frame["future_ret_60d"] - frame["benchmark_ret_60d"]
        else:
            frame["benchmark_symbol"] = pd.NA
            frame["benchmark_ret_5d"] = pd.NA
            frame["benchmark_ret_20d"] = pd.NA
            frame["benchmark_ret_60d"] = pd.NA
            frame["excess_ret_5d"] = pd.NA
            frame["excess_ret_20d"] = pd.NA
            frame["excess_ret_60d"] = pd.NA

        enriched_frames.append(frame)

    return pd.concat(enriched_frames, ignore_index=True)


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

        for current_pos in current_positions:
            if current_pos >= len(trade_dates):
                for lst in (
                    days_to, days_since, in_window, post_window,
                    eps_estimate_last, reported_eps_last, surprise_pct_last,
                    is_positive_surprise, is_negative_surprise,
                    days_to_bucket, days_since_bucket,
                    pre_10d, post_5d, post_10d,
                    post_pos_surp, post_neg_surp,
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
        enriched_frames.append(frame)

    return pd.concat(enriched_frames, ignore_index=True)


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
            "future_ret_5d": float(row["future_ret_5d"]) if pd.notna(row["future_ret_5d"]) else None,
            "future_ret_20d": float(row["future_ret_20d"]) if pd.notna(row["future_ret_20d"]) else None,
            "future_ret_60d": float(row["future_ret_60d"]) if pd.notna(row["future_ret_60d"]) else None,
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
            "benchmark_symbol": row.get("benchmark_symbol") if pd.notna(row.get("benchmark_symbol")) else None,
            "benchmark_ret_5d": float(row["benchmark_ret_5d"]) if pd.notna(row.get("benchmark_ret_5d")) else None,
            "benchmark_ret_20d": float(row["benchmark_ret_20d"]) if pd.notna(row.get("benchmark_ret_20d")) else None,
            "benchmark_ret_60d": float(row["benchmark_ret_60d"]) if pd.notna(row.get("benchmark_ret_60d")) else None,
            "excess_ret_5d": float(row["excess_ret_5d"]) if pd.notna(row.get("excess_ret_5d")) else None,
            "excess_ret_20d": float(row["excess_ret_20d"]) if pd.notna(row.get("excess_ret_20d")) else None,
            "excess_ret_60d": float(row["excess_ret_60d"]) if pd.notna(row.get("excess_ret_60d")) else None,
            "builtAt": built_at,
        }
        for symbol in BENCHMARK_SYMBOLS:
            prefix = symbol.lower()
            for horizon in (5, 20, 60):
                key = f"{prefix}_ret_{horizon}d"
                record[key] = float(row[key]) if pd.notna(row.get(key)) else None
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
    earnings_df = load_earnings_frame()
    print(f"Loaded {len(earnings_df):,} earnings rows")
    feature_df = attach_earnings_event_features(feature_df, price_df, earnings_df)
    saved = save_features(feature_df)
    print(f"Saved {saved:,} feature rows to {FEATURE_COLLECTION}")
    return saved


if __name__ == "__main__":
    build_daily_symbol_features()
