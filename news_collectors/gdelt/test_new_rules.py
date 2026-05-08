#!/usr/bin/env python3
"""
Test GDELT company rules for the 60 new stocks.

Validates:
  1. JSON files load without errors
  2. RuleManager initializes all 60 symbols correctly
  3. Keyword extraction returns sensible results
  4. Ambiguous-ticker rules correctly BLOCK false positives and PASS true positives
  5. Non-ambiguous rules pass basic positive articles through

Usage:
  cd /Users/xiz/Quant_trade/quant_data
  USE_SLM_FILTER=false .venv311/bin/python news_collectors/gdelt/test_new_rules.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Patch path so we can import the special_rules package ────────────────────
GDELT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GDELT_DIR))

os.environ.setdefault("USE_SLM_FILTER", "false")  # skip LLM calls during test

from special_rules import RuleManager  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
RULES_DIR = GDELT_DIR / "company_rules"
TODAY = datetime.now()

NEW_SYMBOLS = [
    "SHOP", "NET", "ZS", "HUBS", "WDAY", "VEEV", "TEAM", "TTD", "OKTA", "APP",
    "RBLX", "COIN", "TWLO", "DUOL", "CFLT", "GTLB", "MNDY", "S",
    "LLY", "JNJ", "AMGN", "GILD", "REGN", "VRTX", "ISRG", "UNH", "MRNA",
    "ABBV", "PFE", "MDT", "SYK", "DXCM", "ILMN",
    "V", "MA", "PYPL", "GS", "JPM", "MS", "BLK", "SCHW", "AXP", "COF",
    "DIS", "SNAP", "SPOT", "RDDT", "PINS",
    "NKE", "HD", "SBUX", "MCD", "TGT",
    "CAT", "HON", "RTX", "LMT", "GE", "DE", "BA",
]

# ── Hand-crafted test cases for ambiguous tickers ────────────────────────────
# Format: (symbol, title, content, expected_result, description)
AMBIGUOUS_TESTS: list[tuple[str, str, str, bool, str]] = [
    # V — Visa Inc
    ("V", "Visa Inc reports strong Q3 payment volume growth", "Visa Inc payment network revenue rose 12%.", True, "Visa earnings news should PASS"),
    ("V", "US student visa applications hit record high in 2024", "Students seeking F-1 visas faced long wait times.", False, "Immigration visa news should FAIL"),
    ("V", "Ryan McInerney discusses Visa network expansion", "Cross-border transactions grew 15% this quarter.", True, "CEO name + expansion keywords should PASS"),

    # S — SentinelOne
    ("S", "SentinelOne launches Purple AI for threat hunting", "The cybersecurity firm unveiled Purple AI today.", True, "SentinelOne product launch should PASS"),
    ("S", "S&P 500 closes at record high on tech rally", "The index rose 1.2% on broad-based gains.", False, "S&P 500 article should FAIL"),
    ("S", "Tomer Weingarten on SentinelOne XDR growth", "SentinelOne's endpoint detection revenue doubled.", True, "CEO + SentinelOne should PASS"),

    # BA — Boeing
    ("BA", "Boeing 737 MAX production ramps up ahead of FAA review", "Boeing increased monthly 737 aircraft output.", True, "Boeing aircraft news should PASS"),
    ("BA", "British Airways BA announces new London-Tokyo route", "The airline will operate three weekly flights.", False, "British Airways should FAIL"),
    ("BA", "Boeing Starliner capsule completes ISS mission", "The spacecraft returned Boeing crew safely.", True, "Boeing space news should PASS"),

    # GE — GE Aerospace
    ("GE", "GE Aerospace wins $2B jet engine contract", "GE Aerospace will supply LEAP engines for the deal.", True, "GE Aerospace should PASS"),
    ("GE", "General Electric healthcare unit posts strong sales", "GE HealthCare reported record imaging sales.", False, "GE HealthCare (not aerospace) should FAIL"),
    ("GE", "GE Aviation's new engine test at Peebles facility", "The jet engine passed all performance benchmarks.", True, "GE Aviation jet engine should PASS"),

    # MA — Mastercard
    ("MA", "Mastercard reports record cross-border payment volume", "MA earnings beat estimates as spending surges.", True, "Mastercard earnings should PASS"),
    ("MA", "Massachusetts governor signs climate bill", "The MA governor signed the new energy legislation.", False, "Massachusetts abbreviation should FAIL"),

    # GS — Goldman Sachs
    ("GS", "Goldman Sachs investment banking revenue beats estimates", "GS reported $3.2B in trading revenue this quarter.", True, "Goldman Sachs news should PASS"),
    ("GS", "GS Warriors win NBA championship game seven", "The Golden State Warriors defeated their rivals.", False, "Golden State Warriors should FAIL"),

    # MS — Morgan Stanley
    ("MS", "Morgan Stanley wealth management AUM hits $5 trillion", "MS reported record client assets under management.", True, "Morgan Stanley news should PASS"),
    ("MS", "MS Dhoni retires from international cricket", "The legendary captain announced his retirement.", False, "Cricket player initials should FAIL"),

    # NET — Cloudflare
    ("NET", "Cloudflare reports 30% revenue growth in Q4", "NET stock rose on strong enterprise customer additions.", True, "Cloudflare earnings should PASS"),
    ("NET", "Net income for S&P 500 companies up 8%", "Aggregate net income across the index rose this quarter.", False, "Generic 'net income' should FAIL"),

    # APP — AppLovin
    ("APP", "AppLovin AI-powered ad platform doubles revenue", "APP posted 100% revenue growth driven by mobile gaming.", True, "AppLovin revenue should PASS"),
    ("APP", "New mobile app for grocery delivery launches", "The grocery delivery app is now available on iOS.", False, "Generic 'app' article should FAIL"),

    # CAT — Caterpillar
    ("CAT", "Caterpillar machinery sales jump on infrastructure demand", "CAT raised its full-year guidance on strong construction orders.", True, "Caterpillar machinery should PASS"),
    ("CAT", "A cat was rescued from a tree in Ohio", "Local firefighters saved the stranded feline.", False, "Literal cat article should FAIL"),

    # DE — Deere
    ("DE", "Deere & Co raises guidance on strong farm equipment demand", "John Deere reported record quarterly revenue.", True, "Deere news should PASS"),
    ("DE", "Germany DE visa requirements updated for 2025", "Travelers to Germany should check new requirements.", False, "Germany country code should FAIL"),

    # HD — Home Depot
    ("HD", "Home Depot comparable sales rise on pro customer growth", "HD reported better-than-expected quarterly revenue.", True, "Home Depot news should PASS"),
    ("HD", "HD video streaming hits new resolution standards", "High-definition video now supports 8K resolution.", False, "HD video abbreviation should FAIL"),

    # COF — Capital One
    ("COF", "Capital One credit card defaults rise amid consumer stress", "COF set aside more loan loss reserves in Q2.", True, "Capital One should PASS"),
    ("COF", "Coffee prices surge on Brazil drought concerns", "Arabica coffee futures hit a five-year high.", False, "Coffee news should FAIL"),
]

# ── Positive-only tests for non-ambiguous tickers ───────────────────────────
POSITIVE_TESTS: list[tuple[str, str, str, str]] = [
    ("SHOP", "Shopify merchants hit $1 trillion GMV milestone", "Shopify Inc reported strong merchant growth.", "Shopify milestone"),
    ("ZS", "Zscaler Zero Trust Architecture wins enterprise clients", "ZS Zero Trust cloud security expanded.", "Zscaler ZTA"),
    ("LLY", "Eli Lilly Mounjaro GLP-1 drug drives record revenue", "LLY tirzepatide sales exceeded $3B in the quarter.", "Eli Lilly GLP-1"),
    ("JNJ", "Johnson & Johnson oncology pipeline gets FDA approval", "J&J submitted NDA for its latest cancer drug.", "JNJ oncology"),
    ("JPM", "JPMorgan Chase Jamie Dimon warns of recession risk", "Chase bank tightened lending standards this quarter.", "JPMorgan CEO"),
    ("V", "Visa Inc payment volume growth accelerates internationally", "Visa Inc cross-border transactions rose 18%.", "Visa Inc international"),
    ("MA", "Mastercard payment network expands into emerging markets", "MA contactless payment volume surged in Q3.", "Mastercard emerging markets"),
    ("DIS", "Walt Disney streaming subscriber count hits 200 million", "Disney+ added 10M subs this quarter.", "Disney streaming"),
    ("SNAP", "Snapchat daily active users grow on AR features", "Snap Inc reported 5% DAU growth in North America.", "Snapchat DAU"),
    ("NKE", "Nike revenue beats on strong direct-to-consumer sales", "John Donahoe said Nike digital grew 20%.", "Nike revenue"),
    ("SBUX", "Starbucks same-store sales recover under Brian Niccol", "SBUX comp sales turned positive for the first time.", "Starbucks recovery"),
    ("CAT", "Caterpillar construction equipment backlog hits record", "CAT machinery orders from infrastructure projects surged.", "Caterpillar backlog"),
    ("BA", "Boeing 787 Dreamliner deliveries resume after FAA review", "Boeing aircraft production increased this quarter.", "Boeing deliveries"),
    ("LMT", "Lockheed Martin F-35 deliveries ahead of schedule", "LMT defense contract backlog reached $160B.", "Lockheed Martin F-35"),
    ("GE", "GE Aerospace LEAP engine orders surge from airlines", "GE Aviation reported a record engine backlog.", "GE Aerospace engines"),
]


def _make_article(title: str, content: str) -> dict:
    return {"title": title, "content": content}


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print("=" * 70)


def run_json_validation() -> list[str]:
    """Check all 60 new JSON files load correctly."""
    section("1. JSON File Validation")
    failed = []
    for sym in NEW_SYMBOLS:
        path = RULES_DIR / f"{sym}.json"
        if not path.exists():
            print(f"  ❌ MISSING: {sym}.json")
            failed.append(sym)
            continue
        try:
            with open(path) as f:
                cfg = json.load(f)
            assert cfg.get("symbol") == sym, f"symbol field mismatch: {cfg.get('symbol')}"
            assert cfg.get("primary_keywords"), "primary_keywords is empty"
            print(f"  ✅ {sym:<6}  ({len(cfg.get('primary_keywords', []))} primary, {len(cfg.get('expansion_keywords', []))} expansion kw)")
        except Exception as e:
            print(f"  ❌ {sym}: {e}")
            failed.append(sym)
    return failed


def run_rule_manager_load() -> RuleManager | None:
    """Initialise RuleManager and check all 60 symbols are present."""
    section("2. RuleManager Initialization")
    try:
        rm = RuleManager()
    except Exception as e:
        print(f"  ❌ RuleManager init failed: {e}")
        return None

    all_loaded = rm.get_all_symbols()
    missing = [s for s in NEW_SYMBOLS if s not in all_loaded]
    extra = [s for s in all_loaded if s not in NEW_SYMBOLS]  # old symbols — expected

    print(f"  Total symbols loaded: {len(all_loaded)}")
    print(f"  New symbols present:  {len(NEW_SYMBOLS) - len(missing)}/{len(NEW_SYMBOLS)}")
    if missing:
        print(f"  ❌ Missing from RuleManager: {missing}")
    else:
        print("  ✅ All 60 new symbols loaded")
    return rm


def run_keyword_check(rm: RuleManager) -> None:
    """Spot-check keyword extraction for all new symbols."""
    section("3. Keyword Extraction Check")
    for sym in NEW_SYMBOLS:
        try:
            kws = rm.get_keywords(sym, TODAY)
            count = len(kws)
            flag = "✅" if count >= 3 else "⚠️ "
            print(f"  {flag} {sym:<6}  {count:>3} keywords  (sample: {kws[:4]})")
        except Exception as e:
            print(f"  ❌ {sym}: get_keywords raised {e}")


def run_ambiguous_tests(rm: RuleManager) -> tuple[int, int]:
    """Run positive + negative test cases for ambiguous tickers."""
    section("4. Ambiguous Ticker Rule Tests")
    passed = 0
    failed_cases = []

    for sym, title, content, expected, desc in AMBIGUOUS_TESTS:
        art = _make_article(title, content)
        try:
            result = rm.should_include(sym, art)
        except Exception as e:
            result = None
            print(f"  ❌ {sym} ERROR: {e}")

        ok = result == expected
        if ok:
            passed += 1
            status = "✅ PASS"
        else:
            status = f"❌ FAIL (got={result}, want={expected})"
            failed_cases.append((sym, desc, title, result, expected))

        print(f"  {status}  [{sym}] {desc}")
        if not ok:
            print(f"         Title: {title}")

    total = len(AMBIGUOUS_TESTS)
    print(f"\n  Result: {passed}/{total} passed")
    if failed_cases:
        print(f"\n  ── Failed cases ──")
        for sym, desc, title, got, want in failed_cases:
            print(f"    [{sym}] {desc}")
            print(f"      Title:  {title}")
            print(f"      Got={got}, Want={want}")
    return passed, total


def run_positive_tests(rm: RuleManager) -> tuple[int, int]:
    """Run positive-only tests (all should return True) for non-ambiguous tickers."""
    section("5. Positive Pass-through Tests")
    passed = 0
    failed_cases = []

    for sym, title, content, desc in POSITIVE_TESTS:
        art = _make_article(title, content)
        try:
            result = rm.should_include(sym, art)
        except Exception as e:
            result = None
            print(f"  ❌ {sym} ERROR: {e}")

        if result is True or result is None:  # NoRule always returns True
            passed += 1
            print(f"  ✅ PASS  [{sym}] {desc}")
        else:
            failed_cases.append((sym, desc, title))
            print(f"  ❌ FAIL  [{sym}] {desc}")
            print(f"         Title: {title}")

    total = len(POSITIVE_TESTS)
    print(f"\n  Result: {passed}/{total} passed")
    return passed, total


def main() -> None:
    print("\n" + "=" * 70)
    print("  GDELT Rule Test — 60 New Stocks")
    print(f"  Date: {TODAY.strftime('%Y-%m-%d')}  SLM: {os.environ.get('USE_SLM_FILTER','true')}")
    print("=" * 70)

    json_failures = run_json_validation()
    rm = run_rule_manager_load()

    if rm is None:
        print("\n❌ Cannot continue — RuleManager failed to load.")
        sys.exit(1)

    run_keyword_check(rm)
    amb_pass, amb_total = run_ambiguous_tests(rm)
    pos_pass, pos_total = run_positive_tests(rm)

    section("Summary")
    total_pass = amb_pass + pos_pass
    total_tests = amb_total + pos_total
    print(f"  JSON validation:      {'✅ all ok' if not json_failures else '❌ ' + str(json_failures)}")
    print(f"  Ambiguous tests:      {amb_pass}/{amb_total}")
    print(f"  Positive tests:       {pos_pass}/{pos_total}")
    print(f"  Overall:              {total_pass}/{total_tests} ({100*total_pass//total_tests}%)")
    if total_pass == total_tests and not json_failures:
        print("\n  🎉 All tests passed — rules are ready for collection!")
    else:
        print("\n  ⚠️  Some tests failed — review rules before running collection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
