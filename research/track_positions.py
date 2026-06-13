#!/usr/bin/env python3
"""Paper-trading position tracker (C.4 + C.5).

Simulates holding the daily top-N LONG signals and tracks them day-by-day on
the dense price history, applying exit rules. Idempotent: re-running rebuilds
the full state from `daily_signals` + `stock_prices_history`.

Reads:   quant_data.daily_signals          (entry candidates, per signal date)
         quant_data.stock_prices_history    (daily close, for returns/holding)
Writes:  quant_data.positions               (open + closed paper positions)
         quant_data.alerts                   (exit-trigger alerts)

Run:
  LOCAL_MONGO_URI="mongodb://root:root@127.0.0.1:37018/" \
  .venv311/bin/python research/track_positions.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"

ENTRY_TOP_N = int(os.getenv("POSITION_ENTRY_TOP_N", "5"))      # open the day's top-N LONG
MAX_HOLD_DAYS = int(os.getenv("POSITION_MAX_HOLD_DAYS", "60"))  # trading days
EXIT_SCORE = float(os.getenv("POSITION_EXIT_SCORE", "0.0"))     # exit if latest score < this
SENT_REVERSAL = float(os.getenv("POSITION_SENT_REVERSAL", "-0.1"))  # exit if sentiment_shift_5d < this


def _date_str(v) -> str:
    return str(v)[:10]


def load_signals(db):
    """{date: [LONG docs sorted by rank]} and {(symbol,date): doc}."""
    by_date: dict[str, list[dict]] = {}
    by_symbol_date: dict[tuple[str, str], dict] = {}
    for d in db.daily_signals.find():
        date = _date_str(d["trade_date"])
        by_symbol_date[(d["symbol"], date)] = d
        if d.get("signal_type") == "LONG":
            by_date.setdefault(date, []).append(d)
    for date in by_date:
        by_date[date].sort(key=lambda x: x.get("signal_rank", 9999))
    return by_date, by_symbol_date


def load_prices(db):
    """{symbol: [(date_str, close)] sorted} for symbols that appear in signals."""
    symbols = db.daily_signals.distinct("symbol")
    prices: dict[str, list[tuple[str, float]]] = {}
    cur = db.stock_prices_history.find(
        {"symbol": {"$in": symbols}},
        {"symbol": 1, "timestamp": 1, "close": 1, "_id": 0},
    )
    for r in cur:
        c = r.get("close")
        if c is None:
            continue
        prices.setdefault(r["symbol"], []).append((_date_str(r["timestamp"]), float(c)))
    for s in prices:
        prices[s].sort(key=lambda x: x[0])
    return prices


def build_positions(signals_by_date, signals_by_sym_date, prices):
    """Day-by-day simulation. Returns (positions, alerts)."""
    # union trading calendar across tracked symbols
    all_dates = sorted({d for series in prices.values() for d, _ in series})
    signal_dates = set(signals_by_date)
    first_signal = min(signal_dates) if signal_dates else None
    if first_signal is None:
        return [], []
    calendar = [d for d in all_dates if d >= first_signal]

    # per-symbol fast lookups: date -> (index, close)
    sym_idx = {s: {d: i for i, (d, _) in enumerate(series)} for s, series in prices.items()}

    def close_on_or_before(symbol, date):
        series = prices.get(symbol)
        if not series:
            return None, None
        lo, hi, best = 0, len(series) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if series[mid][0] <= date:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return (series[best][1], series[best][0]) if best is not None else (None, None)

    open_pos: dict[str, dict] = {}   # symbol -> position
    positions: list[dict] = []
    alerts: list[dict] = []

    def bars_held(symbol, entry_date, as_of_date):
        idx = sym_idx.get(symbol, {})
        if entry_date in idx and as_of_date in idx:
            return idx[as_of_date] - idx[entry_date]
        return 0

    for date in calendar:
        # 1. evaluate exits for open positions as of this date
        for symbol in list(open_pos):
            pos = open_pos[symbol]
            price, pdate = close_on_or_before(symbol, date)
            if price is None:
                continue
            held = bars_held(symbol, pos["entry_date"], pdate)
            ret = (price - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] else 0.0
            sig = signals_by_sym_date.get((symbol, date))  # only on signal dates

            trigger = None
            if held >= MAX_HOLD_DAYS:
                trigger = "max_hold"
            elif sig is not None:
                if sig.get("earnings_miss_signal") == 1:
                    trigger = "earnings_miss"
                elif sig.get("composite_score") is not None and sig["composite_score"] < EXIT_SCORE:
                    trigger = "score_below_exit"
                elif (sig.get("sentiment_shift_5d") is not None
                      and sig["sentiment_shift_5d"] < SENT_REVERSAL):
                    trigger = "sentiment_reversal"

            if trigger:
                pos.update(status="closed", exit_date=date, exit_price=round(price, 4),
                           exit_return=round(ret, 6), days_held=held, exit_trigger=trigger)
                positions.append(pos)
                alerts.append({
                    "alert_date": date, "symbol": symbol, "alert_type": trigger,
                    "entry_date": pos["entry_date"], "days_held": held,
                    "return_at_alert": round(ret, 6),
                    "message": _alert_msg(trigger, symbol, held, ret),
                    "created_at": datetime.now(timezone.utc),
                })
                del open_pos[symbol]

        # 2. open new positions on signal dates (top-N LONG not already held)
        if date in signals_by_date:
            for sig in signals_by_date[date][:ENTRY_TOP_N]:
                symbol = sig["symbol"]
                if symbol in open_pos:
                    continue
                price, pdate = close_on_or_before(symbol, date)
                if price is None:
                    continue
                open_pos[symbol] = {
                    "symbol": symbol, "entry_date": date, "entry_price": round(price, 4),
                    "entry_score": round(float(sig.get("composite_score") or 0.0), 6),
                    "entry_rank": int(sig.get("signal_rank") or 0),
                    "status": "open",
                }

    # 3. mark remaining open positions to the latest available price
    latest_date = calendar[-1] if calendar else None
    for symbol, pos in open_pos.items():
        price, pdate = close_on_or_before(symbol, latest_date)
        held = bars_held(symbol, pos["entry_date"], pdate) if pdate else 0
        ret = (price - pos["entry_price"]) / pos["entry_price"] if (price and pos["entry_price"]) else 0.0
        pos.update(current_price=round(price, 4) if price else None,
                   current_return=round(ret, 6), days_held=held,
                   as_of_date=pdate, exit_trigger=None)
        positions.append(pos)

    return positions, alerts


def _alert_msg(trigger, symbol, held, ret):
    reason = {
        "max_hold": f"held {held}d (max {MAX_HOLD_DAYS}d) — time exit",
        "score_below_exit": "signal score dropped below exit threshold",
        "earnings_miss": "negative earnings surprise",
        "sentiment_reversal": "5d sentiment reversed negative",
    }.get(trigger, trigger)
    return f"EXIT {symbol}: {reason} (return {ret:+.1%})"


def upsert(col, docs, keys):
    ops = [UpdateOne({k: d[k] for k in keys}, {"$set": d}, upsert=True) for d in docs]
    if ops:
        r = col.bulk_write(ops)
        return r.upserted_count, r.modified_count
    return 0, 0


def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    db.positions.create_index([("symbol", 1), ("entry_date", 1)], unique=True, background=True)
    db.positions.create_index([("status", 1)], background=True)
    db.alerts.create_index([("symbol", 1), ("entry_date", 1), ("alert_type", 1)], unique=True, background=True)

    signals_by_date, signals_by_sym_date = load_signals(db)
    prices = load_prices(db)
    positions, alerts = build_positions(signals_by_date, signals_by_sym_date, prices)

    up_p = upsert(db.positions, positions, ["symbol", "entry_date"])
    up_a = upsert(db.alerts, alerts, ["symbol", "entry_date", "alert_type"])

    open_n = sum(1 for p in positions if p["status"] == "open")
    closed_n = sum(1 for p in positions if p["status"] == "closed")
    print(f"Positions: {len(positions)} ({open_n} open, {closed_n} closed) "
          f"upserted={up_p[0]} modified={up_p[1]}")
    print(f"Alerts: {len(alerts)} upserted={up_a[0]} modified={up_a[1]}")

    cur_open = sorted([p for p in positions if p["status"] == "open"],
                      key=lambda x: x.get("current_return", 0), reverse=True)
    if cur_open:
        print("\nCurrent holdings (paper):")
        for p in cur_open:
            print(f"  {p['symbol']:<6} entry {p['entry_date']} @{p['entry_price']:<8} "
                  f"held {p['days_held']:>2}d  ret {p.get('current_return', 0):+.2%}")
    if alerts:
        print("\nRecent exit alerts:")
        for a in sorted(alerts, key=lambda x: x["alert_date"], reverse=True)[:8]:
            print(f"  {a['alert_date']} {a['message']}")


if __name__ == "__main__":
    main()
