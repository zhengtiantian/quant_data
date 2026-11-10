"""Hard rules in front of the relevance model (research/labeling/prefilter.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "labeling"))
import prefilter as pf  # noqa: E402


@pytest.fixture(autouse=True)
def blocklist(monkeypatch):
    monkeypatch.setattr(pf, "_blocked", frozenset({"biztoc.com"}))


@pytest.mark.parametrize("title,url", [
    ("BizToc", "https://biztoc.com/x/abc"),
    ("No Title", "https://example.com/a"),
    ("404", "https://example.com/a"),
    ("Amazon.com", "https://www.amazon.com/dp/1"),
    ("Fox News", "https://www.foxnews.com/video/1"),
    ("Business Insider", "https://www.businessinsider.com/x"),
    ("Xcel Energy", "https://example.com/xcel"),
    ("iPhone", "https://example.com/iphone"),
    ("", "https://example.com/a"),
])
def test_generic_titles(title, url):
    assert pf.junk_reason(title, url) == "generic_title"


@pytest.mark.parametrize("title", [
    "NOV - NOV - Stock Price & News",
    "Smurfit Westrock Plc (SW) Stock Price, News, Quote & History",
    "American Airlines Group Inc (AAL:UW): Stock Price, Quote and News",
    "Brown-Forman Corp (BF.B) Stock Price & 30 Year Financial Data",
    "AAPL Stock Price | Apple Inc. Stock Quote (U.S.: Nasdaq) | MarketWatch",
    "Robert Painter Net Worth, Biography, and Insider Trading",
    "Buy Agilent 1260 Infinity II LC System Read Reviews",
    "American Airlines Archives",
    "Research Reports & Trade Ideas",
    "Access Denied - Reuters",
])
def test_non_news_pages(title):
    assert pf.junk_reason(title, "https://example.com/a") == "non_news_page"


@pytest.mark.parametrize("title", [
    "Apple stock price falls 5% after earnings miss",
    "Why Tesla's stock price and valuation keep rising",
    "Amcor promotes Peter Konieczny to CEO",
    "Iron Mountain Buys Lane Archive Technologies in Philippines",
    "Access denied and insults; US mutual funds get taste of activist life",
    "Morgan Stanley cuts SK hynix price target on memory glut",
    "$75 Costco coupon offer is a scam",
    "Tim Cook's net worth passes $2 billion as Apple hits record",
])
def test_headlines_reach_the_model(title):
    assert pf.junk_reason(title, "https://www.reuters.com/a") is None


def test_blocked_domain():
    assert pf.junk_reason("Jim Cramer says this is the stock to buy now", "https://biztoc.com/x/1") == "blocked_domain"
    assert pf.junk_reason("Jim Cramer says this is the stock to buy now", "https://www.cnbc.com/1") is None


def test_bad_url_is_not_an_error():
    assert pf.junk_reason("Morgan Stanley cuts SK hynix price target", "http://[bad") is None


@pytest.mark.parametrize("title", ["Agilent Technologies", "Agilent Technologies Inc. (A)", "The Agilent Technologies, Inc."])
def test_title_that_is_only_the_company_name(title):
    names = ["Agilent Technologies, Inc.", "Agilent"]
    assert pf.junk_reason(title, "https://example.com/a", names) == "generic_title"
    assert pf.junk_reason("Agilent Technologies raises full-year guidance", "https://example.com/a", names) is None


@pytest.mark.parametrize("symbol,title,expected", [
    ("MCO", "Moody's downgrades US banks, warns of possible cuts to others", "source_only"),
    ("MCO", "Moody's cuts India's GDP growth forecast for 2020 to 5.4%", "source_only"),
    ("SPGI", "S&P Global upgrades India's rating to BBB, says outlook stable", "source_only"),
    ("MCO", "Moody's raises its full-year profit forecast after strong quarter", None),
    ("MCO", "Moody's to cut 200 jobs in restructuring", None),
    ("SPGI", "S&P Global buys carbon data firm for $1.2 billion", None),
    ("TSM", "TSMC cuts sales forecast for 2023 amid weak global economy", None),
    ("JPM", "Moody's downgrades US banks, warns of possible cuts to others", None),
])
def test_rating_agency_as_the_source(symbol, title, expected):
    assert pf.junk_reason(title, "https://www.reuters.com/a", (), symbol) == expected


def test_generic_title_over_a_full_body_is_left_to_the_model():
    body = "Netflix unveiled the Netflix Binge Scale today, which takes a crack at the speed members watch. " * 8
    assert pf.junk_reason("Deadline", "http://deadline.com/print-article/1/", (), "NFLX", body) is None
    assert pf.junk_reason("Deadline", "http://deadline.com/print-article/1/", (), "NFLX", "short") == "generic_title"
    assert pf.junk_reason("NOV - NOV - Stock Price & News", "https://fool.com/q", (), "NOV", body) == "non_news_page"
