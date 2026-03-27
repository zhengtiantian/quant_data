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

FEATURE_REBUILD_ALL = os.getenv("FEATURE_REBUILD_ALL", "false").lower() == "true"
FEATURE_LOOKBACK_DAYS = int(os.getenv("FEATURE_LOOKBACK_DAYS", "180"))
FEATURE_START_DATE = os.getenv("FEATURE_START_DATE")
FEATURE_END_DATE = os.getenv("FEATURE_END_DATE")


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


def _extract_news_date(doc: dict) -> str | None:
    raw_date = doc.get("date")
    if raw_date:
        digits = "".join(ch for ch in str(raw_date) if ch.isdigit())
        if len(digits) >= 8:
            return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    for key in ("publishedAt", "timestamp", "collectedAt"):
        parsed = _parse_iso_date(doc.get(key))
        if parsed:
            return parsed
    return None


def _extract_price_date(doc: dict) -> str | None:
    for key in ("timestamp", "date", "publishedAt", "collectedAt"):
        value = doc.get(key)
        if key == "date" and value:
            digits = "".join(ch for ch in str(value) if ch.isdigit())
            if len(digits) >= 8:
                return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
        parsed = _parse_iso_date(value)
        if parsed:
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
        news_df.groupby(["symbol", "name", "date"], as_index=False)
        .agg(
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

    grouped = grouped.groupby("symbol", group_keys=False).apply(_add_rollups)
    return grouped


def load_price_frame() -> pd.DataFrame:
    _, end_date, effective_start = _resolve_range()
    client = create_client()
    col = client[DB_NAME][PRICE_COLLECTION]

    cursor = col.find(
        {"symbol": {"$exists": True, "$ne": None}},
        {"_id": 0, "symbol": 1, "timestamp": 1, "date": 1, "close": 1},
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
        rows.append({"symbol": doc["symbol"], "trade_date": date_ts, "close": float(doc["close"])})

    if not rows:
        return pd.DataFrame()

    price_df = pd.DataFrame(rows).sort_values(["symbol", "trade_date"]).drop_duplicates(
        subset=["symbol", "trade_date"], keep="last"
    )
    return price_df


def attach_price_labels(feature_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty or price_df.empty:
        feature_df["trade_date"] = pd.NaT
        feature_df["close"] = pd.NA
        feature_df["future_ret_5d"] = pd.NA
        feature_df["future_ret_20d"] = pd.NA
        feature_df["future_ret_60d"] = pd.NA
        return feature_df

    enriched_frames = []
    for symbol, frame in feature_df.groupby("symbol"):
        prices = price_df[price_df["symbol"] == symbol].copy()
        frame = frame.sort_values("date").copy()
        if prices.empty:
            frame["trade_date"] = pd.NaT
            frame["close"] = pd.NA
            frame["future_ret_5d"] = pd.NA
            frame["future_ret_20d"] = pd.NA
            frame["future_ret_60d"] = pd.NA
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

        for pos in idx:
            if pos >= len(prices):
                mapped_trade_dates.append(pd.NaT)
                mapped_closes.append(pd.NA)
                ret_5d.append(pd.NA)
                ret_20d.append(pd.NA)
                ret_60d.append(pd.NA)
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

        frame["trade_date"] = mapped_trade_dates
        frame["close"] = mapped_closes
        frame["future_ret_5d"] = ret_5d
        frame["future_ret_20d"] = ret_20d
        frame["future_ret_60d"] = ret_60d
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

    ops = []
    built_at = datetime.now(UTC).isoformat()
    for row in feature_df.to_dict("records"):
        record = {
            "symbol": row["symbol"],
            "name": row["name"],
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
            "builtAt": built_at,
        }
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
    saved = save_features(feature_df)
    print(f"Saved {saved:,} feature rows to {FEATURE_COLLECTION}")
    return saved


if __name__ == "__main__":
    build_daily_symbol_features()
