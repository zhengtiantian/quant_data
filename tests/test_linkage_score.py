"""N.6 step 7: a judgment is scored from the first price after it, against SPY over the same bars. No Mongo:
the bars are put in the cache score_one reads first."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "news_linkage"))
import score  # noqa: E402

SESSIONS = ["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"]   # Fri, Mon..Fri
UTC = timezone.utc


def test_entry_is_the_first_price_after_the_judgment():
    # New York is UTC-5 in November: the session runs 14:30-21:00 UTC
    assert score.entry_point(datetime(2026, 11, 9, 13, 0, tzinfo=UTC), SESSIONS) == (1, "open")    # before the open
    assert score.entry_point(datetime(2026, 11, 9, 16, 0, tzinfo=UTC), SESSIONS) == (1, "close")   # during the session
    assert score.entry_point(datetime(2026, 11, 9, 21, 30, tzinfo=UTC), SESSIONS) == (2, "open")   # after the close
    assert score.entry_point(datetime(2026, 11, 8, 21, 0, tzinfo=UTC), SESSIONS) == (1, "open")    # Sunday night
    assert score.entry_point(datetime(2026, 11, 14, 12, 0, tzinfo=UTC), SESSIONS) is None          # no session yet


def test_excess_return_and_direction():
    stock = {s: (100.0, 100.0) for s in SESSIONS} | {"2026-11-09": (100.0, 90.0), "2026-11-13": (100.0, 85.0)}
    spy = {s: (500.0, 500.0) for s in SESSIONS} | {"2026-11-13": (500.0, 505.0)}
    at = datetime(2026, 11, 8, 21, 0, tzinfo=UTC)            # Sunday: entry at Monday's open (100)
    cache = {("NVDA", at.date()): stock, ("SPY", at.date()): spy}
    s = score.score_one(None, "NVDA", at, -1, cache, now=datetime(2026, 11, 15, tzinfo=UTC))
    assert s["entry"] == {"session": "2026-11-09", "price": "open"}
    assert s["h1"] == {"ret": -10.0, "spy": 0.0, "excess": -10.0, "right": True}         # Monday's close
    assert s["h5"]["excess"] == -16.0 and s["h5"]["right"] is True                        # Friday: -15% vs +1%
    assert "h20" not in s                                                                 # not known yet
    # a session counts only once it has closed
    s = score.score_one(None, "NVDA", at, -1, cache, now=datetime(2026, 11, 13, 18, 0, tzinfo=UTC))
    assert "h5" not in s


def test_track_counts_directional_calls_and_neutral_moves():
    docs = [{"answer": {"direction": -1}, "score": {"h5": {"excess": -4.0, "right": True}}},
            {"answer": {"direction": 2}, "score": {"h5": {"excess": -2.0, "right": False}}},
            {"answer": {"direction": 0}, "score": {"h5": {"excess": 3.0, "right": None}}}]
    t = score.track(docs)["h5"]
    assert (t["calls"], t["right"], t["mean_called_excess"]) == (2, 1, 1.0)               # (+4 - 2) / 2
    assert (t["neutral"], t["neutral_mean_abs_excess"]) == (1, 3.0)
