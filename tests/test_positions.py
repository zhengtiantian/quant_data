"""Tests for H.3 volatility-adaptive stop-loss and OOS IC monitoring.

Covers: _stop_pct, compute_daily_vols, _spearman_ic, build_positions stop-loss trigger.
Pure-logic tests — no MongoDB.
"""

import math

import pytest

import track_positions as tp


# ── _stop_pct ─────────────────────────────────────────────────────────────────

class TestStopPct:

    def test_none_vol_returns_floor(self):
        assert tp._stop_pct(None) == pytest.approx(0.04)

    def test_zero_vol_returns_floor(self):
        assert tp._stop_pct(0.0) == pytest.approx(0.04)

    def test_low_vol_clamped_to_floor(self):
        # 2.0 × 0.015 = 0.03 → clamped to 0.04
        assert tp._stop_pct(0.015) == pytest.approx(0.04)

    def test_typical_vol_passes_through(self):
        # 2.0 × 0.025 = 0.05 — between floor and cap
        assert tp._stop_pct(0.025) == pytest.approx(0.05)

    def test_high_vol_clamped_to_cap(self):
        # 2.0 × 0.10 = 0.20 → clamped to 0.12
        assert tp._stop_pct(0.10) == pytest.approx(0.12)

    def test_vol_at_exact_cap_boundary(self):
        # 2.0 × 0.06 = 0.12 → exactly cap
        assert tp._stop_pct(0.06) == pytest.approx(0.12)

    def test_negative_vol_returns_floor(self):
        assert tp._stop_pct(-0.01) == pytest.approx(0.04)


# ── compute_daily_vols ────────────────────────────────────────────────────────

def _make_prices(n: int, start: float = 100.0, daily_return: float = 0.001):
    """Generates n (date, close) pairs ascending."""
    dates = [f"2026-{(i // 30 + 1):02d}-{(i % 30 + 1):02d}" for i in range(n)]
    closes = [start * ((1 + daily_return) ** i) for i in range(n)]
    return sorted(zip(dates, closes))


class TestComputeDailyVols:

    def test_too_few_bars_returns_empty(self):
        prices = {"AAPL": _make_prices(21)}
        result = tp.compute_daily_vols(prices)
        assert result["AAPL"] == {}

    def test_exactly_22_bars_has_entries(self):
        prices = {"AAPL": _make_prices(22)}
        result = tp.compute_daily_vols(prices)
        assert len(result["AAPL"]) > 0

    def test_vols_are_non_negative(self):
        prices = {"AAPL": _make_prices(40)}
        result = tp.compute_daily_vols(prices)
        for v in result["AAPL"].values():
            assert v >= 0.0

    def test_constant_price_gives_zero_vol(self):
        prices = {"AAPL": _make_prices(30, daily_return=0.0)}
        result = tp.compute_daily_vols(prices)
        for v in result["AAPL"].values():
            assert v == pytest.approx(0.0, abs=1e-12)

    def test_high_vol_series_gives_larger_vol(self):
        low_vol  = {"AAPL": _make_prices(30, daily_return=0.001)}
        high_vol = {"AAPL": _make_prices(30, daily_return=0.05)}
        low_result  = tp.compute_daily_vols(low_vol)
        high_result = tp.compute_daily_vols(high_vol)
        avg_low  = sum(low_result["AAPL"].values())  / len(low_result["AAPL"])
        avg_high = sum(high_result["AAPL"].values()) / len(high_result["AAPL"])
        assert avg_high > avg_low

    def test_multiple_symbols_independent(self):
        prices = {
            "AAPL": _make_prices(30, daily_return=0.001),
            "MSFT": _make_prices(30, daily_return=0.003),
        }
        result = tp.compute_daily_vols(prices)
        assert "AAPL" in result and "MSFT" in result
        assert len(result["AAPL"]) > 0
        assert len(result["MSFT"]) > 0

    def test_missing_symbol_gets_empty_dict(self):
        prices = {"AAPL": _make_prices(10)}
        result = tp.compute_daily_vols(prices)
        assert result["AAPL"] == {}


# ── _spearman_ic ─────────────────────────────────────────────────────────────

class TestSpearmanIC:

    def test_less_than_5_returns_nan(self):
        assert math.isnan(tp._spearman_ic([1, 2, 3, 4], [1, 2, 3, 4]))

    def test_perfect_positive_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert tp._spearman_ic(xs, xs) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert tp._spearman_ic(xs, ys) == pytest.approx(-1.0, abs=1e-9)

    def test_uncorrelated_returns_near_zero(self):
        scores = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        rets   = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8]
        ic = tp._spearman_ic(scores, rets)
        assert not math.isnan(ic)
        assert abs(ic) < 0.6

    def test_returns_float(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert isinstance(tp._spearman_ic(xs, xs), float)

    def test_exactly_5_elements_does_not_return_nan(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 3.0, 1.0, 4.0, 2.0]
        result = tp._spearman_ic(xs, ys)
        assert not math.isnan(result)
        assert -1.0 <= result <= 1.0


# ── build_positions: stop-loss trigger ───────────────────────────────────────

class TestBuildPositionsStopLoss:

    def _minimal_fixture(self, entry_price=100.0, next_price=93.0,
                         vol=0.025) -> tuple:
        """Signal on 2026-01-02, price drops to next_price on 2026-01-03."""
        sig = {
            "symbol": "AAPL",
            "signal_type": "LONG",
            "signal_rank": 1,
            "composite_score": 1.0,
        }
        signals_by_date     = {"2026-01-02": [sig]}
        signals_by_sym_date = {("AAPL", "2026-01-02"): sig}
        prices = {
            "AAPL": [("2026-01-02", entry_price), ("2026-01-03", next_price)],
        }
        daily_vols = {"AAPL": {"2026-01-02": vol}}
        return signals_by_date, signals_by_sym_date, prices, daily_vols

    def test_stop_loss_triggered_on_large_drop(self):
        # vol=0.025 → stop_pct=0.05; price drops 7% → stop triggered
        args = self._minimal_fixture(entry_price=100.0, next_price=93.0, vol=0.025)
        positions, alerts = tp.build_positions(*args)
        closed = [p for p in positions if p.get("status") == "closed"]
        assert len(closed) == 1
        assert closed[0]["exit_trigger"] == "stop_loss"

    def test_stop_loss_not_triggered_on_small_drop(self):
        # vol=0.025 → stop_pct=0.05; price drops only 3% → no stop
        args = self._minimal_fixture(entry_price=100.0, next_price=97.0, vol=0.025)
        positions, alerts = tp.build_positions(*args)
        closed = [p for p in positions if p.get("status") == "closed"]
        assert len(closed) == 0

    def test_stop_loss_alert_generated(self):
        args = self._minimal_fixture(entry_price=100.0, next_price=93.0, vol=0.025)
        _, alerts = tp.build_positions(*args)
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "stop_loss"

    def test_entry_records_stop_pct(self):
        # After building, closed position should carry stop_pct from vol
        args = self._minimal_fixture(entry_price=100.0, next_price=93.0, vol=0.025)
        positions, _ = tp.build_positions(*args)
        closed = [p for p in positions if p.get("status") == "closed"]
        assert closed[0]["stop_pct"] == pytest.approx(0.05, abs=1e-4)

    def test_floor_stop_pct_used_when_no_vol(self):
        sig = {
            "symbol": "AAPL", "signal_type": "LONG",
            "signal_rank": 1, "composite_score": 1.0,
        }
        signals_by_date     = {"2026-01-02": [sig]}
        signals_by_sym_date = {("AAPL", "2026-01-02"): sig}
        prices = {"AAPL": [("2026-01-02", 100.0), ("2026-01-03", 95.0)]}
        daily_vols = {}  # no vol → floor=0.04

        positions, alerts = tp.build_positions(
            signals_by_date, signals_by_sym_date, prices, daily_vols
        )
        # 5% drop > 4% floor → stop should fire
        closed = [p for p in positions if p.get("status") == "closed"]
        assert len(closed) == 1
        assert closed[0]["stop_pct"] == pytest.approx(0.04, abs=1e-4)
