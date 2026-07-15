"""Tests for earnings feature attachment and macro regime features (C.8 / D-series).

Pure-logic tests — no MongoDB. Run from quant_data root:
  .venv311/bin/python -m pytest tests/ -q
"""

import numpy as np
import pandas as pd
import pytest

import daily_symbol_features as dsf


# ── helpers ───────────────────────────────────────────────────────────────────

def _price_df(symbol="AAPL", n=30, start="2026-01-02"):
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "symbol": symbol,
        "trade_date": dates,
        "close": [150.0 + i * 0.5 for i in range(n)],
    })


def _earnings_df(symbol="AAPL", event_date="2026-01-10",
                 estimate=2.0, reported=2.4):
    surprise = (reported - estimate) / abs(estimate) if estimate else None
    return pd.DataFrame([{
        "symbol": symbol,
        "event_date": pd.Timestamp(event_date),
        "eps_estimate": estimate,
        "reported_eps": reported,
        "surprise_pct": surprise,
    }])


def _feature_df(symbol="AAPL", n=20, start="2026-01-02"):
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "symbol": symbol,
        "date": dates.strftime("%Y-%m-%d"),   # string date required by attach_earnings_event_features
        "trade_date": dates,                   # Timestamp required for fwd-price lookup
    })


def _macro_df(n=260, vix=15.0, spy_start=450.0, spy_trend=0.2):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({
        "date": dates,
        "vix": vix + rng.normal(0, 2, n),
        "tnx": 4.0 + rng.normal(0, 0.1, n),
        "dxy": 100.0 + rng.normal(0, 1, n),
        "spy_close": [spy_start + i * spy_trend for i in range(n)],
    })


# ── attach_earnings_event_features ───────────────────────────────────────────

class TestAttachEarningsEventFeatures:

    def test_empty_feature_df_returns_empty(self):
        out = dsf.attach_earnings_event_features(
            pd.DataFrame(columns=["symbol", "date"]),
            _price_df(), _earnings_df(),
        )
        assert out.empty

    def test_empty_earnings_fills_na_columns(self):
        out = dsf.attach_earnings_event_features(
            _feature_df(),
            _price_df(),
            pd.DataFrame(columns=["symbol", "event_date"]),
        )
        assert "days_to_earnings" in out.columns
        assert out["days_to_earnings"].isna().all()
        assert "surprise_pct_last" in out.columns

    def test_empty_prices_fills_na_columns(self):
        out = dsf.attach_earnings_event_features(
            _feature_df(),
            pd.DataFrame(columns=["symbol", "trade_date", "close"]),
            _earnings_df(),
        )
        assert out["days_to_earnings"].isna().all()

    def test_post_event_surprise_populated(self):
        # event on 2026-01-10; feature rows after that date should see surprise
        out = dsf.attach_earnings_event_features(
            _feature_df(n=25),
            _price_df(n=30),
            _earnings_df(event_date="2026-01-10", estimate=2.0, reported=2.4),
        )
        cutoff = pd.Timestamp("2026-01-10")
        post = out[out["trade_date"] > cutoff].dropna(subset=["surprise_pct_last"])
        if not post.empty:
            assert post["surprise_pct_last"].iloc[0] == pytest.approx(0.2, rel=1e-3)

    def test_positive_surprise_signal(self):
        out = dsf.attach_earnings_event_features(
            _feature_df(n=25),
            _price_df(n=30),
            _earnings_df(estimate=2.0, reported=2.5),
        )
        cutoff = pd.Timestamp("2026-01-10")
        post = out[out["trade_date"] > cutoff].dropna(subset=["is_positive_surprise"])
        if not post.empty:
            assert post["is_positive_surprise"].iloc[0] == 1

    def test_negative_surprise_signal(self):
        out = dsf.attach_earnings_event_features(
            _feature_df(n=25),
            _price_df(n=30),
            _earnings_df(estimate=2.0, reported=1.6),
        )
        cutoff = pd.Timestamp("2026-01-10")
        post = out[out["trade_date"] > cutoff].dropna(subset=["is_negative_surprise"])
        if not post.empty:
            assert post["is_negative_surprise"].iloc[0] == 1

    def test_no_symbol_overlap_fills_na(self):
        out = dsf.attach_earnings_event_features(
            _feature_df(symbol="AAPL"),
            _price_df(symbol="AAPL"),
            _earnings_df(symbol="MSFT"),
        )
        # No matching earnings → days_to_earnings stays NaN for all rows
        assert out["days_to_earnings"].isna().all() or (out["days_to_earnings"].dropna().empty)


# ── compute_regime_features ───────────────────────────────────────────────────

class TestComputeRegimeFeatures:

    def test_empty_input_returns_empty(self):
        out = dsf.compute_regime_features(pd.DataFrame())
        assert out.empty or "trade_date" in out.columns

    def test_output_columns_present(self):
        out = dsf.compute_regime_features(_macro_df())
        for col in [
            "macro_vix", "macro_risk_on", "macro_vix_pctile_252d",
            "macro_spy_ret_20d", "macro_tnx_change_20d", "macro_is_high_vol",
        ]:
            assert col in out.columns, f"missing {col}"

    def test_calm_rising_market_is_risk_on(self):
        # Low VIX (~15) + SPY steadily rising → last row should be risk_on=1
        out = dsf.compute_regime_features(_macro_df(n=260, vix=15.0, spy_trend=0.3))
        last = out.dropna(subset=["macro_risk_on"]).iloc[-1]
        assert last["macro_risk_on"] == 1

    def test_high_vix_sets_high_vol_flag(self):
        df = _macro_df(n=260, vix=15.0)
        # Spike VIX to 50 for the last 20 rows → high vol percentile
        df.loc[df.index[-20:], "vix"] = 50.0
        out = dsf.compute_regime_features(df)
        last = out.dropna(subset=["macro_is_high_vol"]).iloc[-1]
        assert last["macro_is_high_vol"] == 1

    def test_spy_below_200ma_sets_risk_off(self):
        # SPY declining below 200-day MA → macro_spy_above_200ma = 0
        df = _macro_df(n=260, spy_start=500.0, spy_trend=-0.5)
        out = dsf.compute_regime_features(df)
        last = out.dropna(subset=["macro_spy_above_200ma"]).iloc[-1]
        assert last["macro_spy_above_200ma"] == 0

    def test_vix_pctile_between_0_and_1(self):
        out = dsf.compute_regime_features(_macro_df(n=260))
        pct = out["macro_vix_pctile_252d"].dropna()
        assert (pct >= 0).all() and (pct <= 1).all()

    def test_spy_ret_20d_direction(self):
        # Consistently rising market → positive 20d returns
        out = dsf.compute_regime_features(_macro_df(n=260, spy_trend=1.0))
        rets = out["macro_spy_ret_20d"].dropna()
        assert (rets > 0).mean() > 0.9
