"""N.8 (2026-11-19): context rules for the five universe companies with an ambiguous short name, and the
ticker mapping of names.json (share classes, renamed companies, reused tickers)."""
import contextlib
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UNI = ROOT / "collectors" / "news" / "gdelt" / "universe_rules"
sys.path.insert(0, str(ROOT / "collectors" / "news" / "gdelt"))
os.environ["USE_SLM_FILTER"] = "false"

from special_rules import RuleManager  # noqa: E402
from special_rules.headline import gate_terms, title_gate  # noqa: E402


@pytest.fixture(scope="module")
def rm():
    with contextlib.redirect_stdout(io.StringIO()):
        return RuleManager(str(UNI / "company_rules"))


def _kept(rm, sym: str, headline: str, url: str = "https://example.com/news/a") -> bool:
    """A GKG row (entity lists, then the headline) through the headline gate and the rule, as the filter runs it."""
    raw = f"Org One;Org Two;{headline}"
    if not title_gate(gate_terms(rm, sym, datetime(2024, 6, 1)), raw, url):
        return False
    with contextlib.redirect_stdout(io.StringIO()):
        return rm.should_include(sym, {"title": "", "content": raw, "url": url, "date": "20240601"})


@pytest.mark.parametrize("sym, headline, keep", [
    ("MRK", "FDA authorizes Merck's at-home antiviral COVID-19 pill", True),
    ("MRK", "Merck & Co., Inc. (NYSE:MRK) Shares Sold by Essex Savings Bank", True),
    ("MRK", "Germany's Merck KGaA Names Kai Beckmann as Incoming Chair, CEO", False),
    ("MRK", "Merck Opens Climate-Neutral €150 Million Filter Manufacturing Facility in Ireland", False),
    ("MRK", "Merck Foundation CEO meets Kenya First Lady", False),
    ("MRK", "Tata Electronics, Merck sign pact to boost semiconductor materials in India", False),
    ("VTR", "Ventas Declares First Quarter 2023 Dividend of $0.45 per Common Share", True),
    ("VTR", "Lowe's predice una disminución en sus ventas para el 2022", False),
    ("VTR", "Ventas de autos caen en México", False),
    ("PSA", "Public Storage Q3 Net Income Declines, Revenue Rises", True),
    ("PSA", "Police arrest man in killing at public storage facility", False),
    ("ROL", "Rollins, Inc. Acquisition Expands Northwest Exterminating into South Florida", True),
    ("ROL", "Rollins (ROL) Stock Is Up, What You Need To Know", True),
    ("ROL", "Atlanta-Based Rollins, Inc. Receives First Geotab CEO Visionary Award", True),
    ("ROL", "Rollins Says USDA Thinking Hard About Financial Relief for Farmers", False),
    ("ROL", "Secretary of Agriculture Brooke Rollins Shares Message With FFA Members", False),
    ("ROL", "Henry Rollins Shares Rock 'n' Roll Artifacts With Nashville", False),
    ("DOV", "Dover Corp. acquires Acme Cryogenics of Allentown for $295M", True),
    ("DOV", "Dover Fueling Solutions Launches DMP Probe", True),
    ("DOV", "Dover police investigate late-night reports of suspected gunfire", False),
    ("DOV", "ILC Dover Acquires Solo Containment", False),
    ("BBWI", "L Brands Sells Majority Stake in Victoria's Secret's UK Business", True),
    ("PSKY", "ViacomCBS Reaches Deal to Sell CNET for $500 Million", True),
])
def test_context_rules(rm, sym, headline, keep):
    assert _kept(rm, sym, headline) is keep


def test_context_pattern_reads_the_real_title_at_download(rm):
    # fetch_article re-runs the rule on the downloaded page: title and full text, no GKG row
    art = {"title": "Dover police seek robbery suspect", "content": "Dover Corporation is not involved. " * 3, "url": "https://x.com/a"}
    with contextlib.redirect_stdout(io.StringIO()):
        assert rm.should_include("DOV", art) is False
        assert rm.should_include("DOV", {**art, "title": "Dover (NYSE:DOV) beats estimates"}) is True


def test_news_symbols_point_at_collected_companies():
    names = json.loads((UNI / "names.json").read_text())
    have = {p.stem for p in (UNI / "company_rules").glob("*.json")}
    have |= {p.stem for p in (ROOT / "collectors" / "news" / "gdelt" / "company_rules").glob("*.json")}
    mapped = {t: v["news_symbol"] for t, v in names.items() if "news_symbol" in v}
    assert {"GOOG", "FOX", "NWS", "UA", "FB", "LB", "PARA", "INFO", "SATS"} <= set(mapped)
    for t, to in mapped.items():
        assert to is None or to in have, (t, to)
        assert not (UNI / "company_rules" / f"{t}.json").exists(), t      # its own rule stays dropped
    assert json.loads((UNI / "rule_aliases.json").read_text()) == {t: to for t, to in sorted(mapped.items()) if to}


def test_news_symbols_helper():
    sys.path.insert(0, str(ROOT / "research" / "models"))
    import q17_rank
    m = q17_rank.news_symbols()
    assert m["GOOG"] == "GOOGL" and m["LB"] == "BBWI" and m["INFO"] is None and m["MRK"] == "MRK"
