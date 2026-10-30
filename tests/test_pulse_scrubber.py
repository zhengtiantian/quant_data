"""Extended-hours SIP late-print scrubber and two-stage leg fill tests (earnings_pulse)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "models"))
import earnings_pulse as ep  # noqa: E402


def _make_m1(rows):
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="America/New_York") for t, *_ in rows])
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": v} for _, o, h, l, c, v in rows],
        index=idx,
    )


def test_scrub_removes_intc_after_hours_close16_phantom_high():
    # Exact pattern observed on INTC 2026-10-29: regular close 92.07, plunge to 75, then 16:15/16:16 h=92.07 TRF late print
    m = _make_m1([
        ("2026-10-29 15:59", 92.05, 92.10, 92.00, 92.07, 50000),
        ("2026-10-29 16:00", 92.07, 92.44, 92.07, 92.39, 12000),
        ("2026-10-29 16:01", 92.29, 92.29, 78.11, 80.85, 80000),
        ("2026-10-29 16:02", 80.55, 81.10, 76.00, 76.94, 45000),
        ("2026-10-29 16:14", 75.47, 76.40, 75.02, 75.99, 15000),
        ("2026-10-29 16:15", 75.99, 92.07, 74.56, 74.91, 18000),
        ("2026-10-29 16:16", 74.92, 92.07, 74.89, 76.00, 16000),
        ("2026-10-29 16:17", 75.96, 76.00, 75.30, 75.85, 11000),
        ("2026-10-30 09:30", 71.00, 72.00, 70.50, 71.50, 200000),
    ])
    clean, n = ep.scrub_extended_bars(m, close_i=1)
    assert n == 2
    assert clean.loc[pd.Timestamp("2026-10-29 16:15", tz="America/New_York"), "high"] == pytest.approx(75.99)
    assert clean.loc[pd.Timestamp("2026-10-29 16:16", tz="America/New_York"), "high"] == pytest.approx(76.00)
    # 16:01 transition bar and 09:30 regular open are untouched
    assert clean.loc[pd.Timestamp("2026-10-29 16:01", tz="America/New_York"), "high"] == pytest.approx(92.29)


def test_scrub_removes_rally_close16_phantom_low_without_touching_real_spike():
    # Exact pattern observed on AAPL 2024-05-02: close16=173.08, jumps to 185, 16:53 prints l=173.03
    m = _make_m1([
        ("2024-05-02 15:59", 173.00, 173.15, 172.95, 173.08, 90000),
        ("2024-05-02 16:30", 173.10, 184.50, 173.10, 183.90, 500000),  # real jump: close moved to 183.90!
        ("2024-05-02 16:52", 184.90, 185.30, 184.70, 185.15, 40000),
        ("2024-05-02 16:53", 185.18, 185.20, 173.03, 184.63, 45000),   # late close16 low print!
        ("2024-05-02 16:54", 184.65, 185.00, 184.50, 184.80, 30000),
    ])
    clean, n = ep.scrub_extended_bars(m, close_i=1)
    assert n == 1
    assert clean.loc[pd.Timestamp("2024-05-02 16:30", tz="America/New_York"), "high"] == pytest.approx(184.50)
    assert clean.loc[pd.Timestamp("2024-05-02 16:53", tz="America/New_York"), "low"] == pytest.approx(184.63)


def test_leg_gains_with_fill_distinguishes_non_fill_from_filled():
    bars = (
        np.array([80.0, 80.5, 81.0, 79.0]),
        np.array([80.2, 81.0, 89.0, 79.5]),
        np.array([79.8, 80.0, 85.0, 78.0]),
        np.array([80.0, 80.8, 86.0, 78.5]),
    )
    idx = {bb: 3 for bb in ep.BUYBACK}
    idx["next_open"] = 3
    g_miss, f_miss = ep.leg_gains_with_fill(bars, 1, idx, 12.0)  # +12% on 80 = 89.6 -> missed
    assert f_miss == 0 and all(v == 0.0 for v in g_miss.values())
    g_hit, f_hit = ep.leg_gains_with_fill(bars, 1, idx, 8.0)     # +8% on 80 = 86.4 -> hit on bar 2 (h=89)
    assert f_hit == 1 and g_hit["lim3"] > 0.0
