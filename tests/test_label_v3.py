"""N.6 label student (gd1_label_v3): its answer parsed and normalised; `leadership` survives normalisation, the
other companies' effects are kept only when there is one per listed company."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "labeling"))
import mlx_students as ms  # noqa: E402
import student_gd1_shadow as sh  # noqa: E402


def test_parse_full_answer():
    p = ms.parse_label_v3("leadership|-0.6|high|export_controls,geopolitics_war,made_up|+1,-1", ["AMD", "TSM"])
    assert p == {"event_type": "leadership", "sentiment": -0.6, "signal_strength": "high",
                 "themes": ["export_controls", "geopolitics_war"], "impacts": {"AMD": 1, "TSM": -1}}


def test_empty_parts_and_mismatched_impacts():
    assert ms.parse_label_v3("other|0.0|low|-|-", [])["impacts"] == {}
    assert ms.parse_label_v3("product|0.2|medium|-|0", ["AMD", "TSM"])["impacts"] is None      # one value, two companies
    assert ms.parse_label_v3("not an answer", []) is None


def test_normalised_like_production_but_leadership_kept():
    assert sh.v3_label("leadership|-1.4|high|-|-", "CEO resigns", [])["event_type"] == "leadership"
    assert sh.v3_label("leadership|-1.4|high|-|-", "CEO resigns", [])["sentiment"] == -1.0           # clamped
    assert sh.v3_label("other|0.1|loud|-|-", "Acme to acquire Widget Co", [])["event_type"] == "MA"   # title override
    assert sh.v3_label("other|0.1|loud|-|-", "x", [])["signal_strength"] == "low"


def test_input_lists_the_other_companies():
    u = ms.label_v3_input("NVDA", "NVIDIA Corporation", "Title", "Body", ["AMD"])
    assert u.startswith("NVDA (NVIDIA Corporation)\nTitle: Title\nContent: Body\nOthers: AMD (")
    assert "Others" not in ms.label_v3_input("NVDA", "NVIDIA Corporation", "Title", "Body", [])
