"""Unit tests for core feature-build helpers (C.8).

Pure-logic tests over fixed inputs — no MongoDB. Run from quant_data root:
  .venv311/bin/python -m pytest tests/ -q
"""

import numpy as np
import pandas as pd
import pytest

import daily_symbol_features as dsf


# ── date helpers ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("2026-06-12T00:00:00Z", "2026-06-12"),
    ("2026-06-12T10:30:00+00:00", "2026-06-12"),
    ("2026-06-12", "2026-06-12"),
    ("  2026-06-12  ", "2026-06-12"),
    (None, None),
    ("", None),
    ("not-a-date", None),
])
def test_parse_iso_date(raw, expected):
    assert dsf._parse_iso_date(raw) == expected


@pytest.mark.parametrize("value,expected", [
    ("2026-01-01", True),          # in range
    ("1990-01-01", True),          # lower bound
    ("2037-06-20", False),         # corrupt future date (the garbage we saw)
    ("1980-01-01", False),         # before lower bound
    (None, False),
    ("garbage", False),
])
def test_is_reasonable_feature_date(value, expected):
    assert dsf._is_reasonable_feature_date(value) is expected


@pytest.mark.parametrize("days,bucket", [
    (0, 0), (5, 0), (6, 1), (15, 1), (16, 2), (30, 2), (31, 3), (999, 3),
    (None, None),
])
def test_bucket_days(days, bucket):
    assert dsf._bucket_days(days) == bucket


def test_bucket_days_nan():
    assert dsf._bucket_days(float("nan")) is None


# ── news aggregation ──────────────────────────────────────────────────────────

def _news_df():
    rows = [
        # symbol, date, data_quality, content_length
        ("AAA", "2026-01-01", "full", 1000),
        ("AAA", "2026-01-01", "title_only", 0),
        ("AAA", "2026-01-02", "full", 800),
    ]
    df = pd.DataFrame(rows, columns=["symbol", "date", "data_quality", "content_length"])
    df["name"] = "Alpha Inc"
    df["url"] = [f"http://x/{i}" for i in range(len(df))]
    df["source_name"] = "Reuters"
    df["source_platform"] = "web"
    df["is_extraction_failed"] = 0
    df["is_timeout_fallback"] = 0
    return df


def test_aggregate_news_features_counts_and_ratio():
    out = dsf.aggregate_news_features(_news_df())
    d1 = out[(out["symbol"] == "AAA") & (out["date"] == "2026-01-01")].iloc[0]
    assert d1["article_count"] == 2
    assert d1["full_count"] == 1
    assert d1["title_only_count"] == 1
    assert d1["full_ratio"] == pytest.approx(0.5)
    assert d1["unique_url_count"] == 2


def test_aggregate_news_features_rolling():
    out = dsf.aggregate_news_features(_news_df()).sort_values("date")
    d2 = out[out["date"] == "2026-01-02"].iloc[0]
    # 3-day rolling sum of article_count over [2, 1]
    assert d2["news_count_3d"] == pytest.approx(3.0)


# ── LLM sentiment aggregation ─────────────────────────────────────────────────

def _llm_df():
    rows = [
        # symbol, date, sentiment, strength, event_type, disagreement
        ("AAA", "2026-01-01", 0.8, "high", "earnings", 0.1),
        ("AAA", "2026-01-01", 0.4, "low", "earnings", 0.2),
    ]
    return pd.DataFrame(rows, columns=["symbol", "date", "sentiment", "strength", "event_type", "disagreement"])


def test_llm_weighted_sentiment():
    out = dsf.aggregate_llm_sentiment_features(_llm_df())
    row = out.iloc[0]
    # weighted mean: (0.8*1.0 + 0.4*0.3) / (1.0 + 0.3)
    expected = (0.8 * 1.0 + 0.4 * 0.3) / (1.0 + 0.3)
    assert row["avg_sentiment_5d"] == pytest.approx(expected, rel=1e-6)


def test_llm_earnings_beat_signal():
    out = dsf.aggregate_llm_sentiment_features(_llm_df())
    row = out.iloc[0]
    # both rows are earnings with sentiment > 0.2 -> beat signal fires
    assert row["earnings_beat_signal"] == 1
    assert row["earnings_miss_signal"] == 0


def test_llm_empty_input():
    out = dsf.aggregate_llm_sentiment_features(pd.DataFrame(columns=["symbol", "date"]))
    assert out.empty


# ── cross-sectional quality score ─────────────────────────────────────────────

def test_quality_score_ranks_best_highest():
    # same date, three symbols; BEST dominates the positive components
    df = pd.DataFrame([
        {"symbol": "BEST", "trade_date": "2026-01-01", "full_ratio": 0.9,
         "unique_source_count": 5, "avg_content_length": 2000,
         "extraction_failed_count": 0, "news_burst_20d": 2.0},
        {"symbol": "MID", "trade_date": "2026-01-01", "full_ratio": 0.5,
         "unique_source_count": 3, "avg_content_length": 1000,
         "extraction_failed_count": 1, "news_burst_20d": 1.0},
        {"symbol": "WORST", "trade_date": "2026-01-01", "full_ratio": 0.1,
         "unique_source_count": 1, "avg_content_length": 100,
         "extraction_failed_count": 3, "news_burst_20d": 0.5},
    ])
    out = dsf.add_quality_score(df).set_index("symbol")
    assert out.loc["BEST", "quality_score"] > out.loc["MID", "quality_score"]
    assert out.loc["MID", "quality_score"] > out.loc["WORST", "quality_score"]
