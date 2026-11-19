"""keyword_refresh's collateral check / auto rule and coverage_watch's weekly flags (2026-11-19)."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import coverage_watch as cw  # noqa: E402
import keyword_refresh as kr  # noqa: E402


def test_neighbours_find_the_owner_phrase():
    titles = ["Samsung Galaxy A20 review", "Galaxy A20 price cut", "Apple A20 chip leak"]
    nb = kr._neighbours(titles, "A20")
    assert nb["before galaxy"] == 2 and nb["after chip"] == 1


def test_auto_needs_a_clean_screen_and_claude():
    base = {"tier": "add", "stories": 3, "sites": 3}
    assert kr.auto_eligible({**base, "screen": {"risks": [], "claude": {"risky": False}}})
    assert not kr.auto_eligible({**base, "screen": {"risks": [{"code": "claude"}], "claude": {"risky": True}}})
    assert not kr.auto_eligible({**base, "screen": {"risks": []}})                       # Claude did not answer
    assert not kr.auto_eligible({**base, "stories": 2, "screen": {"risks": [], "claude": {"risky": False}}})
    assert not kr.auto_eligible({**base, "tier": "weak", "screen": {"risks": [], "claude": {"risky": False}}})


def _frames(days, kept_last_week):
    c, k = [], []
    for i, d in enumerate(days):
        c.append({"symbol": "AAA", "day": d, "n": 20})
        k.append({"symbol": "AAA", "day": d, "n": 10 if i < len(days) - 7 else kept_last_week})
    return pd.DataFrame(c), pd.DataFrame(k)


def test_coverage_flags_a_collapse_and_not_a_steady_week():
    asof = datetime(2026, 11, 19)
    days = pd.date_range("2026-09-01", "2026-11-18").strftime("%Y%m%d")
    c, k = _frames(days, 10)
    assert cw.evaluate(c, k, asof, ["AAA"])["AAA"]["flags"] == []
    c, k = _frames(days, 1)
    codes = {f["code"] for f in cw.evaluate(c, k, asof, ["AAA"])["AAA"]["flags"]}
    assert {"kept_drop", "keep_rate_drop"} <= codes
