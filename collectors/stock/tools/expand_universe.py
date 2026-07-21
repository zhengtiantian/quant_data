#!/usr/bin/env python3
"""Expand stock universe from 40 → 100 symbols.

Adds new symbols to MongoDB stock_universe collection (upsert, safe to re-run).
Then triggers 10-year price history backfill for new symbols only.

Usage:
  LOCAL_MONGO_URI="mongodb://root:root@127.0.0.1:37018/" \
  .venv311/bin/python stock_collector/expand_universe.py [--dry-run] [--prices-only]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
import os

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env", override=False)

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")

# 60 new symbols to bring the universe from 40 → 100
# Chosen for: high news coverage, S&P 500 membership, sector diversity
NEW_SYMBOLS: list[dict] = [
    # ── IT / Tech additions (18) ─────────────────────────────────────────
    {"symbol": "SHOP", "name": "Shopify Inc.",           "sector": "Information Technology", "industry": "Internet Commerce"},
    {"symbol": "NET",  "name": "Cloudflare Inc.",         "sector": "Information Technology", "industry": "Software"},
    {"symbol": "ZS",   "name": "Zscaler Inc.",            "sector": "Information Technology", "industry": "Software"},
    {"symbol": "HUBS", "name": "HubSpot Inc.",            "sector": "Information Technology", "industry": "Software"},
    {"symbol": "WDAY", "name": "Workday Inc.",            "sector": "Information Technology", "industry": "Software"},
    {"symbol": "VEEV", "name": "Veeva Systems Inc.",      "sector": "Information Technology", "industry": "Software"},
    {"symbol": "TEAM", "name": "Atlassian Corp.",         "sector": "Information Technology", "industry": "Software"},
    {"symbol": "TTD",  "name": "The Trade Desk Inc.",     "sector": "Information Technology", "industry": "Software"},
    {"symbol": "OKTA", "name": "Okta Inc.",               "sector": "Information Technology", "industry": "Software"},
    {"symbol": "APP",  "name": "AppLovin Corp.",          "sector": "Information Technology", "industry": "Software"},
    {"symbol": "RBLX", "name": "Roblox Corp.",            "sector": "Communication Services", "industry": "Interactive Media"},
    {"symbol": "COIN", "name": "Coinbase Global Inc.",    "sector": "Information Technology", "industry": "Financial Technology"},
    {"symbol": "TWLO", "name": "Twilio Inc.",             "sector": "Information Technology", "industry": "Software"},
    {"symbol": "DUOL", "name": "Duolingo Inc.",           "sector": "Information Technology", "industry": "Software"},
    {"symbol": "CFLT", "name": "Confluent Inc.",          "sector": "Information Technology", "industry": "Software"},
    {"symbol": "GTLB", "name": "GitLab Inc.",             "sector": "Information Technology", "industry": "Software"},
    {"symbol": "MNDY", "name": "monday.com Ltd.",         "sector": "Information Technology", "industry": "Software"},
    {"symbol": "S",    "name": "SentinelOne Inc.",        "sector": "Information Technology", "industry": "Software"},

    # ── Healthcare / Biotech (15) ─────────────────────────────────────────
    {"symbol": "LLY",  "name": "Eli Lilly and Co.",       "sector": "Health Care", "industry": "Pharmaceuticals"},
    {"symbol": "JNJ",  "name": "Johnson & Johnson",        "sector": "Health Care", "industry": "Pharmaceuticals"},
    {"symbol": "AMGN", "name": "Amgen Inc.",               "sector": "Health Care", "industry": "Biotechnology"},
    {"symbol": "GILD", "name": "Gilead Sciences Inc.",     "sector": "Health Care", "industry": "Biotechnology"},
    {"symbol": "REGN", "name": "Regeneron Pharmaceuticals","sector": "Health Care", "industry": "Biotechnology"},
    {"symbol": "VRTX", "name": "Vertex Pharmaceuticals",   "sector": "Health Care", "industry": "Biotechnology"},
    {"symbol": "ISRG", "name": "Intuitive Surgical Inc.",  "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "UNH",  "name": "UnitedHealth Group Inc.",  "sector": "Health Care", "industry": "Managed Care"},
    {"symbol": "MRNA", "name": "Moderna Inc.",             "sector": "Health Care", "industry": "Biotechnology"},
    {"symbol": "ABBV", "name": "AbbVie Inc.",              "sector": "Health Care", "industry": "Pharmaceuticals"},
    {"symbol": "PFE",  "name": "Pfizer Inc.",              "sector": "Health Care", "industry": "Pharmaceuticals"},
    {"symbol": "MDT",  "name": "Medtronic plc",            "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "SYK",  "name": "Stryker Corp.",            "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "DXCM", "name": "Dexcom Inc.",              "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "ILMN", "name": "Illumina Inc.",            "sector": "Health Care", "industry": "Life Sciences Tools"},

    # ── Financials (10) ───────────────────────────────────────────────────
    {"symbol": "V",    "name": "Visa Inc.",                "sector": "Financials", "industry": "Payment Services"},
    {"symbol": "MA",   "name": "Mastercard Inc.",          "sector": "Financials", "industry": "Payment Services"},
    {"symbol": "PYPL", "name": "PayPal Holdings Inc.",     "sector": "Financials", "industry": "Payment Services"},
    {"symbol": "GS",   "name": "Goldman Sachs Group Inc.", "sector": "Financials", "industry": "Investment Banking"},
    {"symbol": "JPM",  "name": "JPMorgan Chase & Co.",     "sector": "Financials", "industry": "Banking"},
    {"symbol": "MS",   "name": "Morgan Stanley",           "sector": "Financials", "industry": "Investment Banking"},
    {"symbol": "BLK",  "name": "BlackRock Inc.",           "sector": "Financials", "industry": "Asset Management"},
    {"symbol": "SCHW", "name": "Charles Schwab Corp.",     "sector": "Financials", "industry": "Brokerage"},
    {"symbol": "AXP",  "name": "American Express Co.",     "sector": "Financials", "industry": "Payment Services"},
    {"symbol": "COF",  "name": "Capital One Financial",    "sector": "Financials", "industry": "Banking"},

    # ── Communication Services additions (5) ─────────────────────────────
    {"symbol": "DIS",  "name": "Walt Disney Co.",          "sector": "Communication Services", "industry": "Entertainment"},
    {"symbol": "SNAP", "name": "Snap Inc.",                "sector": "Communication Services", "industry": "Social Media"},
    {"symbol": "SPOT", "name": "Spotify Technology SA",    "sector": "Communication Services", "industry": "Entertainment"},
    {"symbol": "RDDT", "name": "Reddit Inc.",              "sector": "Communication Services", "industry": "Social Media"},
    {"symbol": "PINS", "name": "Pinterest Inc.",           "sector": "Communication Services", "industry": "Social Media"},

    # ── Consumer (5) ─────────────────────────────────────────────────────
    {"symbol": "NKE",  "name": "Nike Inc.",                "sector": "Consumer Discretionary", "industry": "Apparel"},
    {"symbol": "HD",   "name": "Home Depot Inc.",          "sector": "Consumer Discretionary", "industry": "Retail"},
    {"symbol": "SBUX", "name": "Starbucks Corp.",          "sector": "Consumer Discretionary", "industry": "Restaurants"},
    {"symbol": "MCD",  "name": "McDonald's Corp.",         "sector": "Consumer Discretionary", "industry": "Restaurants"},
    {"symbol": "TGT",  "name": "Target Corp.",             "sector": "Consumer Discretionary", "industry": "Retail"},

    # ── Industrials (7) ──────────────────────────────────────────────────
    {"symbol": "CAT",  "name": "Caterpillar Inc.",         "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "HON",  "name": "Honeywell International",  "sector": "Industrials", "industry": "Conglomerates"},
    {"symbol": "RTX",  "name": "RTX Corp.",                "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "LMT",  "name": "Lockheed Martin Corp.",    "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "GE",   "name": "GE Aerospace",             "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "DE",   "name": "Deere & Co.",              "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "BA",   "name": "Boeing Co.",               "sector": "Industrials", "industry": "Aerospace & Defense"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Print what would be added without writing")
    p.add_argument("--prices-only", action="store_true", help="Skip universe update, only backfill prices")
    return p.parse_args()


def get_existing_symbols(col) -> set[str]:
    return {d["symbol"] for d in col.find({}, {"symbol": 1, "_id": 0})}


def upsert_symbols(col, symbols: list[dict], dry_run: bool) -> list[str]:
    existing = get_existing_symbols(col)
    new = [s for s in symbols if s["symbol"] not in existing]

    if not new:
        print("All symbols already in universe.")
        return []

    print(f"\nAdding {len(new)} new symbols (skipping {len(symbols) - len(new)} already present):")
    for s in new:
        print(f"  {s['symbol']:<6} {s['name']:<40} {s['sector']}")

    if dry_run:
        print("\n[dry-run] No changes written.")
        return [s["symbol"] for s in new]

    ops = [
        UpdateOne(
            {"symbol": s["symbol"]},
            {"$set": {**s, "index": "SP500"}},
            upsert=True,
        )
        for s in new
    ]
    col.bulk_write(ops)
    print(f"\nInserted {len(new)} symbols into stock_universe.")
    return [s["symbol"] for s in new]


def backfill_prices(new_symbols: list[str]) -> None:
    if not new_symbols:
        return
    script = ROOT / "stock_collector" / "price_collector" / "10y_1d_history_collector.py"
    python = ROOT / ".venv311" / "bin" / "python"
    env = {**os.environ, "SYMBOLS_FILTER": ",".join(new_symbols)}
    print(f"\nBackfilling 10y price history for: {', '.join(new_symbols)}")
    subprocess.run([str(python), str(script)], env=env, check=False)


def main() -> None:
    args = parse_args()
    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["stock_universe"]

    existing_count = col.count_documents({})
    print(f"Current universe: {existing_count} symbols")

    if not args.prices_only:
        new_syms = upsert_symbols(col, NEW_SYMBOLS, args.dry_run)
    else:
        existing = get_existing_symbols(col)
        new_syms = [s["symbol"] for s in NEW_SYMBOLS if s["symbol"] in existing]

    print(f"\nUniverse after update: {col.count_documents({})} symbols")

    if not args.dry_run and new_syms:
        backfill_prices(new_syms)


if __name__ == "__main__":
    main()
