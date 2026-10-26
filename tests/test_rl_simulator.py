"""Action RL simulator (Q.7): fills, exits and costs on hand-made paths."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from rl import grammar, simulator as sim  # noqa: E402

C = sim.COST


def path(bars):
    a = np.array(bars, dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def leg(bars, side, k=0.0, ex=3, y=0.0, z=0.0, s=0.0, entry_end=99, hz_end=None, p0=100.0, sigma=1.0, mkt=None):
    o, h, l, c = path(bars)
    return sim._leg(o, h, l, c, p0, p0 if mkt is None else mkt, sigma, side, k, ex, y, z, s, entry_end,
                    len(o) - 1 if hz_end is None else hz_end)


def test_grammar_size_and_ids():
    a = grammar.actions()
    assert len(a) == 2 * 4 * 12 * 6
    assert a["action_id"].is_unique and a["name"].is_unique


def test_reduce_at_market_then_limit_buy_back():
    # sold at 100; buy back 2% lower (y=2, sigma 1%) at 98 on the third bar
    g, f, code = leg([(100, 100.5, 99.5, 100), (100, 100, 98.5, 99), (99, 99, 97.5, 98)], side=0, ex=0, y=2.0)
    assert (f, code) == (1, 0)
    assert g == pytest.approx(2.0 - 2 * C)


def test_limit_entry_needs_the_price_through_it_and_fills_at_the_open_on_a_gap():
    # a sale limit at +1% (101): the first bar touches 101.03 (not through by 0.05%), the second gaps to 102
    g, f, code = leg([(100, 101.03, 99.9, 100.5), (102, 102.5, 101.5, 102), (102, 102, 101, 101)], side=0, k=1.0)
    assert (f, code) == (1, 4)                                 # no exit: closed at the horizon close, 101
    assert g == pytest.approx((102 - 101) / 100 * 100 - 2 * C)


def test_entry_not_reached_is_no_trade():
    assert leg([(100, 100.5, 99.5, 100)] * 3, side=1, k=2.0) == (0.0, 0, 4)


def test_bracket_counts_the_stop_when_a_bar_reaches_both():
    # bought at 100; target +2% (102), stop -1% (99); the second bar spans 98.5..102.5
    g, f, code = leg([(100, 100.2, 99.8, 100), (100, 102.5, 98.5, 101)], side=1, ex=2, y=2.0, s=1.0)
    assert code == 3
    assert g == pytest.approx(-1.0 - 2 * C)


def test_trailing_stop_follows_the_best_price():
    # bought at 100, trail 1%: the high runs to 105, then a drop through 103.95 (= 105 x 0.99) sells
    bars = [(100, 101, 99.9, 101), (101, 105, 100.9, 104.8), (104.8, 104.9, 103.0, 103.2)]
    g, f, code = leg(bars, side=1, ex=1, z=1.0)
    assert code == 1
    assert g == pytest.approx(105 * 0.99 - 100 - 2 * C)


def test_reduce_trailing_buys_back_on_a_rebound_off_the_low():
    # sold at 100, buy back on a 1% rebound from the low: low 95, then 95.95 reached
    bars = [(100, 100, 97, 97.5), (97.5, 97.6, 95, 95.2), (95.2, 96.5, 95.1, 96.3)]
    g, f, code = leg(bars, side=0, ex=1, z=1.0)
    assert code == 1
    assert g == pytest.approx(100 - 95 * 1.01 - 2 * C)


def test_market_entry_after_the_close_fills_at_the_next_open():
    # decided on a 100 close; sold at the next open 103; bought back at the 101 close: +2 against p0
    g, f, code = leg([(103, 104, 100.5, 101)], side=0, mkt=103.0)
    assert (f, code) == (1, 4)
    assert g == pytest.approx((103 - 101) / 100 * 100 - 2 * C)


def test_simulate_marks_unavailable_horizons():
    o, h, l, c = path([(100, 101, 99, 100)] * 5)
    a = grammar.actions()
    hz_ends = np.array([1, 2, 3, 4, -1, -1], np.int64)       # d10 and d20 beyond the data
    g, f, cd = sim.simulate(o, h, l, c, 100.0, 100.0, 1.0, 4, hz_ends, a.side_code.values, a.entry_k.values, a.exit_code.values,
                            a.y.values, a.z.values, a.s.values, a.horizon_code.values)
    assert np.isnan(g[a.horizon.isin(["d10", "d20"]).values]).all()
    assert not np.isnan(g[a.horizon.isin(["next_open", "d1"]).values]).any()
