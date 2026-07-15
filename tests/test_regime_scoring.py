"""Tests for H.2 dynamic regime scoring — classify_regime, compute_score, _safe_float.

Pure-logic tests — no MongoDB. Run from quant_data root:
  .venv311/bin/python -m pytest tests/ -q
"""

import numpy as np
import pandas as pd
import pytest

import score_daily_signals as sds


# ── helpers ───────────────────────────────────────────────────────────────────

def _row(**kwargs) -> pd.DataFrame:
    """One-row DataFrame for classify_regime / compute_score."""
    return pd.DataFrame([kwargs])


def _feature_df(n: int = 20, sentiment_slope: float = 1.0) -> pd.DataFrame:
    """n-row cross-sectional slice with enough feature columns to exercise scoring."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "symbol": [f"SYM{i:03d}" for i in range(n)],
        "trade_date": "2026-01-10",
        "avg_sentiment_5d": rng.uniform(-1, 1, n) * sentiment_slope,
        "sentiment_shift_5d": rng.uniform(-0.5, 0.5, n),
        "earnings_beat_signal": rng.integers(0, 2, n),
        "earnings_miss_signal": rng.integers(0, 2, n),
        "news_burst_20d": rng.uniform(0, 3, n),
        "quality_score": rng.uniform(0, 1, n),
        "ah_gap": rng.uniform(-0.05, 0.05, n),
        "analyst_buy_ratio": rng.uniform(0.3, 0.9, n),
        "inst_holding_pct_chg": rng.uniform(-0.1, 0.1, n),
        "retail_sent_score": rng.uniform(-1, 1, n),
        "macro_vix_pctile_252d": 0.2,
        "macro_risk_on": 1,
    })
    return df


# ── classify_regime ───────────────────────────────────────────────────────────

class TestClassifyRegime:

    def test_missing_vix_pct_returns_neutral(self):
        assert sds.classify_regime(_row(macro_risk_on=1)) == "NEUTRAL"

    def test_nan_vix_pct_returns_neutral(self):
        assert sds.classify_regime(_row(macro_vix_pctile_252d=float("nan"), macro_risk_on=1)) == "NEUTRAL"

    def test_high_vix_pctile_returns_risk_off(self):
        assert sds.classify_regime(_row(macro_vix_pctile_252d=0.85, macro_risk_on=0)) == "RISK_OFF"

    def test_vix_at_exactly_0_8_returns_risk_off(self):
        assert sds.classify_regime(_row(macro_vix_pctile_252d=0.80, macro_risk_on=1)) == "RISK_OFF"

    def test_vix_moderate_returns_stressed(self):
        assert sds.classify_regime(_row(macro_vix_pctile_252d=0.6, macro_risk_on=1)) == "STRESSED"

    def test_risk_off_flag_returns_stressed(self):
        # low VIX but risk_on=0 → STRESSED
        assert sds.classify_regime(_row(macro_vix_pctile_252d=0.2, macro_risk_on=0)) == "STRESSED"

    def test_low_vix_high_risk_on_returns_risk_on(self):
        assert sds.classify_regime(_row(macro_vix_pctile_252d=0.15, macro_risk_on=1)) == "RISK_ON"

    def test_vix_between_0_3_and_0_5_risk_on_1_returns_neutral(self):
        # 0.3 <= vix_pct < 0.5 with risk_on=1 → doesn't hit STRESSED or RISK_ON → NEUTRAL
        assert sds.classify_regime(_row(macro_vix_pctile_252d=0.40, macro_risk_on=1)) == "NEUTRAL"


# ── compute_score ─────────────────────────────────────────────────────────────

class TestComputeScore:

    def test_output_columns_present(self):
        out = sds.compute_score(_feature_df())
        for col in ["composite_score", "signal_rank", "regime_label", "regime_mult"]:
            assert col in out.columns, f"missing column: {col}"

    def test_regime_label_propagated(self):
        df = _feature_df()
        df["macro_vix_pctile_252d"] = 0.15
        df["macro_risk_on"] = 1
        out = sds.compute_score(df)
        assert (out["regime_label"] == "RISK_ON").all()

    def test_conviction_multiplier_risk_off(self):
        df = _feature_df()
        df["macro_vix_pctile_252d"] = 0.9
        df["macro_risk_on"] = 0
        out = sds.compute_score(df)
        assert (out["regime_mult"] - 0.50).abs().max() < 1e-9

    def test_conviction_multiplier_risk_on(self):
        df = _feature_df()
        df["macro_vix_pctile_252d"] = 0.1
        df["macro_risk_on"] = 1
        out = sds.compute_score(df)
        assert (out["regime_mult"] - 1.20).abs().max() < 1e-9

    def test_signal_rank_is_integer_series(self):
        out = sds.compute_score(_feature_df())
        assert out["signal_rank"].dtype in (int, "int32", "int64")

    def test_signal_rank_range(self):
        n = 20
        out = sds.compute_score(_feature_df(n=n))
        assert out["signal_rank"].min() == 1
        assert out["signal_rank"].max() == n

    def test_missing_feature_column_skipped_gracefully(self):
        df = _feature_df()
        df = df.drop(columns=["avg_sentiment_5d", "earnings_beat_signal"])
        out = sds.compute_score(df)
        assert "composite_score" in out.columns
        assert not out["composite_score"].isna().any()

    def test_top_scorer_has_rank_1(self):
        df = _feature_df(n=10)
        # Force one symbol to dominate
        df.loc[5, "avg_sentiment_5d"] = 100.0
        df.loc[5, "news_burst_20d"] = 100.0
        df.loc[5, "quality_score"] = 1.0
        out = sds.compute_score(df)
        best = out.loc[out["composite_score"].idxmax()]
        assert best["signal_rank"] == 1


# ── _safe_float ───────────────────────────────────────────────────────────────

class TestSafeFloat:

    def test_none_returns_none(self):
        assert sds._safe_float(None) is None

    def test_nan_returns_none(self):
        assert sds._safe_float(float("nan")) is None

    def test_valid_float(self):
        assert sds._safe_float(3.14159) == pytest.approx(3.14159, rel=1e-4)

    def test_integer_input(self):
        assert sds._safe_float(5) == pytest.approx(5.0)

    def test_invalid_string_returns_none(self):
        assert sds._safe_float("abc") is None

    def test_numeric_string(self):
        result = sds._safe_float("2.5")
        assert result == pytest.approx(2.5)
