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
