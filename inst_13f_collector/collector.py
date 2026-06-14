#!/usr/bin/env python3
"""Institutional 13F holdings collector (D.7).

Fetches the two most recent 13F-HR filings from major institutions via edgartools.
Stores per-(symbol, institution, period) rows, then derives two aggregate features:

  inst_holding_pct      : latest known holdings / total shares (sum across institutions,
                          always using each institution's OWN latest period)
  inst_holding_pct_chg  : QoQ change per institution averaged across institutions that
                          have two consecutive periods available

Run weekly (Sun 06:00); data changes quarterly.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path

import edgar
import yfinance as yf
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

edgar.set_identity(os.getenv("SEC_IDENTITY", "quant-research xiz00188@gmail.com"))

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"
RAW_COLLECTION = "inst_13f_raw"      # per (symbol, institution, period)
FEAT_COLLECTION = "inst_13f_holdings"  # aggregated per (symbol, period="latest")
UNIVERSE_COLLECTION = "stock_universe"
SLEEP_BETWEEN = float(os.getenv("INST13F_SLEEP", "1.0"))

INSTITUTIONS: dict[str, int] = {
    "Vanguard":    102909,
    "StateStreet": 93751,
    "Fidelity":    315066,
}
QUARTERS_BACK = 2


def load_universe(client: MongoClient) -> list[str]:
    docs = list(client[DB_NAME][UNIVERSE_COLLECTION].find({}, {"symbol": 1, "_id": 0}))
    return sorted(d["symbol"] for d in docs)


def fetch_total_shares(symbols: list[str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for sym in symbols:
        try:
            fi = yf.Ticker(sym).fast_info
            result[sym] = float(fi.shares) if fi.shares else None
        except Exception:
            result[sym] = None
        time.sleep(0.1)
    return result


def fetch_institution_quarters(
    name: str, cik: int, universe: set[str]
) -> list[tuple[str, dict[str, float]]]:
    """Return [(period_YYYY-MM, {symbol: shares})] for last QUARTERS_BACK quarters."""
    try:
        company = edgar.Company(cik)
        filings = company.get_filings(form="13F-HR")
    except Exception as e:
        print(f"  {name}: filing fetch error — {e}")
        return []

    results = []
    for idx in range(min(QUARTERS_BACK, len(filings))):
        filing = filings[idx]
        try:
            tf = filing.obj()
            period_str = str(tf.report_period)[:7]  # "YYYY-MM"
            df = tf.holdings
            if df is None or df.empty:
                continue
            sym_shares: dict[str, float] = {}
            for _, row in df.iterrows():
                ticker = str(row.get("Ticker") or "").strip()
                if ticker and ticker != "nan" and ticker in universe:
                    try:
                        sym_shares[ticker] = float(row.get("SharesPrnAmount") or 0)
                    except (TypeError, ValueError):
                        pass
            print(f"  {name} {period_str}: {len(sym_shares)}/{len(universe)} universe symbols found")
            results.append((period_str, sym_shares))
        except Exception as e:
            print(f"  {name} filing[{idx}] error: {e}")
        time.sleep(SLEEP_BETWEEN)

    return results


def main() -> None:
    print("=== Institutional 13F collector ===")
    client = MongoClient(MONGO_URI)
    raw_col = client[DB_NAME][RAW_COLLECTION]
    feat_col = client[DB_NAME][FEAT_COLLECTION]
    raw_col.create_index([("symbol", 1), ("institution", 1), ("period", 1)], unique=True, background=True)
    feat_col.create_index([("symbol", 1)], unique=True, background=True)

    universe_list = load_universe(client)
    universe = set(universe_list)
    print(f"Universe: {len(universe)} symbols")

    # Step 1: fetch per-institution quarters → upsert raw collection
    # inst_data[institution][period] = {symbol: shares}
    inst_data: dict[str, list[tuple[str, dict[str, float]]]] = {}
    for name, cik in INSTITUTIONS.items():
        print(f"\nFetching {name} (CIK {cik})...")
        quarters = fetch_institution_quarters(name, cik, universe)
        inst_data[name] = quarters

    now = datetime.now(UTC)
    raw_ops: list[UpdateOne] = []
    for name, quarters in inst_data.items():
        for period_str, sym_shares in quarters:
            for sym, shares in sym_shares.items():
                raw_ops.append(UpdateOne(
                    {"symbol": sym, "institution": name, "period": period_str},
                    {"$set": {"symbol": sym, "institution": name, "period": period_str,
                              "shares": shares, "collectedAt": now}},
                    upsert=True,
                ))
    if raw_ops:
        res = raw_col.bulk_write(raw_ops)
        print(f"\nRaw: upserted={res.upserted_count} modified={res.modified_count}")

    # Step 2: fetch total shares outstanding
    print("\nFetching total shares outstanding...")
    total_shares_map = fetch_total_shares(universe_list)

    # Step 3: compute aggregated features per symbol
    # For inst_holding_pct: use each institution's LATEST period
    # For inst_holding_pct_chg: per institution latest vs prev, then average
    feat_ops: list[UpdateOne] = []

    for sym in universe_list:
        total_shares = total_shares_map.get(sym)
        latest_shares_total = 0.0   # sum across all institutions, each using their own latest period
        chg_list: list[float] = []  # QoQ changes for institutions that have 2 periods

        for name, quarters in inst_data.items():
            if not quarters:
                continue
            # latest quarter for this institution
            period_0, shares_0 = quarters[0]
            latest_shares = shares_0.get(sym, 0.0)
            latest_shares_total += latest_shares

            # QoQ change for this institution
            if len(quarters) >= 2:
                period_1, shares_1 = quarters[1]
                prev_shares = shares_1.get(sym, 0.0)
                if total_shares and total_shares > 0:
                    cur_pct = latest_shares / total_shares
                    prv_pct = prev_shares / total_shares
                    chg_list.append(cur_pct - prv_pct)

        inst_holding_pct = (latest_shares_total / total_shares) if total_shares and total_shares > 0 else None
        inst_holding_pct_chg = (sum(chg_list) / len(chg_list)) if chg_list else None

        feat_ops.append(UpdateOne(
            {"symbol": sym},
            {"$set": {
                "symbol": sym,
                "inst_holding_pct": inst_holding_pct,
                "inst_holding_pct_chg": inst_holding_pct_chg,
                "total_shares": total_shares,
                "collectedAt": now,
            }},
            upsert=True,
        ))

    if feat_ops:
        res2 = feat_col.bulk_write(feat_ops)
        print(f"Features: upserted={res2.upserted_count} modified={res2.modified_count}")

    print(f"\nDone. Raw docs={raw_col.count_documents({})} "
          f"Feature docs={feat_col.count_documents({})}")


if __name__ == "__main__":
    main()
