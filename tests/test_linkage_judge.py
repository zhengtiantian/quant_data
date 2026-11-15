"""N.6 step 4: the Claude judge's guards. No Claude call: subprocess is faked."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "news_linkage"))
import judge  # noqa: E402


def test_check_rejects_unknown_or_missing_citations():
    ids = {"A1": object(), "B2": {"title": "x"}}
    ok = {"direction": -1, "cites": ["A1", "B2"]}
    assert judge.check(ok, ids) == []
    assert judge.check({"direction": -1, "cites": ["A9"]}, ids) == ["unknown citation A9"]
    assert "no citations" in judge.check({"direction": 0, "cites": []}, ids)
    assert "direction out of range" in judge.check({"direction": 3, "cites": ["A1"]}, ids)
    assert judge.check(None, ids) == ["no JSON answer"]


def test_ask_never_passes_an_api_key_and_never_uses_bare(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    seen = {}

    def fake_run(cmd, input, capture_output, text, timeout, cwd, env):
        seen.update(cmd=cmd, env=env, cwd=cwd)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"structured_output": {"direction": 0}}), stderr="")

    with mock.patch.object(judge, "claude_bin", return_value="/bin/claude"), mock.patch.object(judge.subprocess, "run", fake_run):
        res = judge.ask("packet")
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert "--bare" not in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--tools") + 1] == ""          # no tools
    assert "--strict-mcp-config" in seen["cmd"]
    assert judge._answer(res) == {"direction": 0}


def test_answer_falls_back_to_result_text():
    assert judge._answer({"result": '{"direction": 1}'}) == {"direction": 1}
    assert judge._answer({"result": "not json"}) is None


def test_theme_backdrop_other_companies_before_the_cutoff():
    from datetime import datetime, timezone
    from conftest import FakeCollection
    at = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)
    g = lambda th, s="high": {"relevant": True, "signal_strength": s, "themes": th}
    row = lambda i, sym, hour, th, **kw: {"_id": i, "symbol": sym, "title": f"story {i}", "content": "body",
                                         "publishedAt": f"2026-11-09T{hour:02d}:00:00+00:00", "student_gd1": g(th), **kw}
    docs = [row(1, "AMD", 8, ["ai_capex"]), row(2, "NVDA", 9, ["ai_capex"]),            # the stock itself: left out
            row(3, "MU", 13, ["ai_capex"]),                                              # after the cutoff
            row(4, "TSM", 7, ["export_controls"]), row(5, "ORCL", 6, ["crypto"]),       # theme not asked for
            {**row(6, "AVGO", 5, ["ai_capex"]), "student_gd1": g(["ai_capex"], "low")},  # weak
            row(7, "META", 4, ["ai_capex"]), row(8, "GOOGL", 3, ["ai_capex"]), row(9, "MSFT", 2, ["ai_capex"]),
            row(10, "AMZN", 10, ["ai_capex"], title="Seen in the packet")]
    db = {"news_articles": FakeCollection(docs)}
    out = judge._theme_backdrop(db, "NVDA", at, ["ai_capex", "export_controls"], {"seen in the packet"})
    # newest first until PER_THEME (3) per theme, then in time order: MSFT (4th ai_capex) and AMZN (seen) dropped
    assert [x["symbol"] for x in out] == ["GOOGL", "META", "TSM", "AMD"]


def test_packet_with_a_theme_backdrop(monkeypatch):
    # 2026-11-15: a non-empty backdrop (dict rows next to the articles) broke the packet
    import types
    from datetime import datetime, timedelta, timezone
    from conftest import FakeCollection
    import storylines as sl
    monkeypatch.setitem(sys.modules, "injection_guard", types.SimpleNamespace(screen=lambda t: SimpleNamespace(text=t)))
    monkeypatch.setattr(judge, "_background", lambda *a, **k: [])
    monkeypatch.setattr(judge, "_closes", lambda *a, **k: [])
    monkeypatch.setattr(judge, "_spy", lambda *a, **k: [])
    at = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)
    arts = [sl.Article(id=f"a{i}", symbol="NVDA", at=at - timedelta(hours=i + 1), title=f"Nvidia story {i}", text="",
                       url=f"https://s{i}.com/x", domain=f"s{i}.com", src="cross" if i else "gd1", via="AMD" if i else None,
                       themes=["ai_capex"], mark="progress") for i in range(2)]
    line = sl.Storyline("L1", "NVDA", sorted(arts, key=lambda a: a.at))
    line.head = "Nvidia story"
    db = {"news_articles": FakeCollection([{"_id": 1, "symbol": "MU", "title": "Micron lifts capex", "content": "body",
                                            "publishedAt": "2026-11-09T06:00:00+00:00",
                                            "student_gd1": {"relevant": True, "signal_strength": "high", "themes": ["ai_capex"]}}])}
    text, ids = judge.packet("NVDA", at, [line], [], db=db)
    assert "(filed under AMD)" in text and "themes: ai_capex" in text
    assert "Theme backdrop" in text and "filed under MU, theme ai_capex" in text
    assert any(k.startswith("T") for k in ids)
