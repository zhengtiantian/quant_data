"""N.6 steps 1 and 3: storyline assignment and the trigger rules, on synthetic articles and vectors
(no Mongo, no embedding model)."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "news_linkage"))
import storylines as sl  # noqa: E402
import triggers as tg  # noqa: E402

T0 = datetime(2026, 11, 9, 8, 0, tzinfo=timezone.utc)
DF = (1000, {"stock": 300, "nvidia": 200, "shares": 250, "falls": 100})   # common words; anything else is rare


def art(i, title, hours, vec, site="a.com", strength=1, sym="NVDA"):
    return sl.Article(id=f"id{i:04d}", symbol=sym, at=T0 + timedelta(hours=hours), title=title, text=title,
                      url=f"https://{site}/{i}", domain=site, src="kept", strength=strength,
                      person=None)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_same_story_joins_and_a_generic_article_does_not():
    # cosine 0.89 between each pair: close enough to join, not a near-identical repeat (>= 0.95)
    vecs = np.stack([unit([1, 0, 0]), unit([1, 0.5, 0]), unit([1, 0, 0.5])])
    arts = [art(0, "Nvidia board probes Megaspeed ties", 0, None),
            art(1, "Megaspeed links widen Nvidia probe", 1, None),        # shares the rare word "megaspeed"
            art(2, "Nvidia stock falls", 2, None)]                        # close in meaning, no rare word
    lines = sl.build(arts, vecs, DF)
    assert arts[0].storyline == arts[1].storyline
    assert arts[2].storyline != arts[0].storyline
    assert len(lines) == 2


def test_marks_repeat_filler_progress():
    v = unit([1, 0, 0])
    arts = [art(0, "Huang resigns as CEO", 0, None),
            art(1, "Huang resigns as CEO", 1, None),                          # same title: repeat
            art(2, "Is Nvidia a buy after Huang resigns?", 2, None),          # commentary title: filler
            art(3, "Huang resigns; board names Kress interim", 3, None, strength=0)]  # low strength: filler
    sl.build(arts, np.stack([v, v, unit([1, 0.45, 0]), unit([1, 0, 0.45])]), DF)
    assert len({a.storyline for a in arts}) == 1
    assert [a.mark for a in arts] == ["progress", "repeat", "filler", "filler"]


def _line(counts_by_hour, sites=4):
    """A storyline with `n` progress articles at each hour offset (from distinct sites in turn)."""
    arts, k = [], 0
    for h, n in counts_by_hour:
        for _ in range(n):
            a = art(k, f"t{k}", h, None, site=f"s{k % sites}.com")
            a.mark, a.storyline = "progress", "L1"
            arts.append(a)
            k += 1
    arts.sort(key=lambda a: a.at)
    return sl.Storyline("L1", "NVDA", arts)


def test_burst_against_its_own_baseline():
    steady = _line([(-24 * d - 1, 3) for d in range(1, 8)] + [(-2, 4)])        # ~3 a day, 4 today: quiet
    assert tg.hot_at(steady, T0) is None
    spike = _line([(-24 * d - 1, 1) for d in range(1, 8)] + [(-2, 9)])         # ~1 a day, 9 today: hot
    h = tg.hot_at(spike, T0)
    assert h and h.reason == "burst" and h.n24 == 9


def test_new_storyline_needs_count_and_sites():
    assert tg.hot_at(_line([(-3, 5)], sites=4), T0).reason == "new"
    assert tg.hot_at(_line([(-3, 5)], sites=2), T0) is None              # five copies from two sites


def test_refire_only_on_escalation_and_symbol_gap():
    line = _line([(-24 * 3, 1), (-20, 6), (-1, 30)])
    trs = tg.replay([line], T0 - timedelta(hours=19), T0)
    # fires once at the first check (6 in 24h), then once more when 30 >= 2 x 6 and 6 hours have passed
    assert [h.reason for t in trs for h in t.hot][0] in ("burst", "new")
    assert len(trs) == 2 and trs[1].hot[0].reason.endswith("+escalation")


def test_cross_article_counts_only_at_the_strongest_effect():
    weak = _line([(-3, 5)])
    for a in weak.articles:
        a.src, a.sentiment, a.via = "cross", -0.5, "AMD"        # impact -1: context for the judge only
    assert tg.hot_at(weak, T0) is None
    for a in weak.articles:
        a.sentiment = -1.0                                       # impact -2: counts
    assert tg.hot_at(weak, T0).reason == "new"


def test_loader_adds_cross_articles_and_themes():
    from conftest import FakeCollection
    when = "2026-11-09T06:00:00+00:00"
    g = lambda **kw: {"relevant": True, "signal_strength": "high", "sentiment": 0.4, "themes": [], "impacts": {}, **kw}
    na = [
        {"_id": "k1", "symbol": "NVDA", "title": "Nvidia ships Rubin", "publishedAt": when, "student_gd1": g(themes=["ai_capex"])},
        {"_id": "o1", "symbol": "NVDA", "title": "Nvidia board probes ties", "publishedAt": when, "student_gd1": g()},
        {"_id": "x1", "symbol": "AMD", "title": "AMD wins a Meta order", "publishedAt": when,
         "student_gd1": g(impacts={"NVDA": -1, "AMD": 1}, themes=["ai_capex"])},
        {"_id": "x0", "symbol": "AMD", "title": "AMD opens an office", "publishedAt": when, "student_gd1": g(impacts={"NVDA": 0})},
        {"_id": "u1", "symbol": "AMD", "title": "AMD old story", "publishedAt": when, "origin": "universe_backfill",
         "student_gd1": g(impacts={"NVDA": -2})},
    ]
    v2 = [{"_id": "k1", "symbol": "NVDA", "title": "Nvidia ships Rubin", "publishedAt": when, "llm_sentiment_final": 0.5,
           "llm_signal_strength_a": "high"}]
    db = {"news_articles": FakeCollection(na), "news_articles_company_matched_v2": FakeCollection(v2)}
    arts = {a.id: a for a in sl.load_articles(["NVDA"], T0 - timedelta(days=1), T0, db=db)}
    assert set(arts) == {"k1", "o1", "x1"}
    assert arts["k1"].src == "kept" and arts["k1"].themes == ["ai_capex"]
    x = arts["x1"]
    assert (x.src, x.symbol, x.via, x.sentiment, x.themes) == ("cross", "NVDA", "AMD", -0.5, ["ai_capex"])
