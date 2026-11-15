"""N.6 step 5: the live job finds a storyline's last firing in the stored judgments, though rebuilding over
a sliding window may have renumbered it."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "news_linkage"))
import live  # noqa: E402
import storylines as sl  # noqa: E402

T0 = datetime(2026, 11, 12, 14, 0, tzinfo=timezone.utc)


def line(lid, ids, repeats=()):
    arts = [sl.Article(id=i, symbol="NVDA", at=T0 - timedelta(hours=len(ids) - k), title=i, text=i, url="", domain="a.com",
                       src="kept", mark="repeat" if i in repeats else "progress", storyline=lid) for k, i in enumerate(ids)]
    return sl.Storyline(lid, "NVDA", arts)


def test_same_id_or_most_of_the_articles_is_the_same_storyline():
    history = [{"at": T0 - timedelta(hours=6), "storylines": [{"id": "NVDA:old", "n24": 9, "members": ["a", "b", "c", "d"]}]},
               {"at": T0 - timedelta(days=2), "storylines": [{"id": "NVDA:x", "n24": 3, "members": ["q"]}]}]
    # renumbered, but holds 3 of the 4 articles of the firing 6 hours ago
    assert live.prev_firing(line("NVDA:new", ["b", "c", "d", "e", "f"]), history) == (T0 - timedelta(hours=6), 9)
    # the same id, whatever its articles
    assert live.prev_firing(line("NVDA:x", ["z"]), history) == (T0 - timedelta(days=2), 3)
    # only one of the four: another storyline
    assert live.prev_firing(line("NVDA:other", ["a", "y"]), history) is None


def test_repeats_are_not_members():
    l = line("L", ["a", "b", "c"], repeats={"b"})
    assert live._members(l) == ["a", "c"]
