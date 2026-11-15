"""N.6 step 7 (2026-11-15): score every stored judgment against the stock's excess return after it.

Entry is the first price after the judgment, so a move that happened before it is never credited:
  judged before the US open (09:30 New York)   that session's open
  judged during the session                    that session's close (the price at the moment is not stored)
  judged after the close or on a closed day    the next session's open
Exit after h sessions (h = 1, 5, 20): the close h sessions on, counting the entry session when entry was an
open. Excess = the stock's return minus SPY's over the same two prices. A directional call (direction != 0)
is right when the excess has its sign; a "no material effect" call is reported by the size of the move.
The same is done for every other company Claude named (`other_companies`) that has prices.

Called at the start of every live run (live.py): a judgment is rescored until its 20-session exit is known.
    python score.py              # score what can be scored now and print the track record
    python score.py --dry FILE   # score judgments from a JSONL file (the replay) without storing
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from pymongo import MongoClient

import storylines as sl

NY = ZoneInfo("America/New_York")
HORIZONS = (1, 5, 20)
COLLECTION = "news_linkage_judgments"


def _bars(db, symbol: str, start: datetime, days: int = 45) -> dict:
    lo, hi = (start - timedelta(days=1)).strftime("%Y-%m-%d"), (start + timedelta(days=days)).strftime("%Y-%m-%d")
    out = {}
    for r in db["stock_prices_history"].find({"symbol": symbol, "interval": "1d", "timestamp": {"$gte": lo, "$lt": hi}},
                                             {"timestamp": 1, "open": 1, "close": 1}):
        if r.get("close"):
            out[r["timestamp"][:10]] = (r.get("open"), float(r["close"]))
    return out


def entry_point(at: datetime, sessions: list[str]) -> tuple[int, str] | None:
    """(index into sessions, 'open' | 'close') of the first price after `at`."""
    loc = at.astimezone(NY)
    day, t = loc.strftime("%Y-%m-%d"), loc.time()
    for i, s in enumerate(sessions):
        if s < day:
            continue
        if s == day:
            if t < time(9, 30):
                return i, "open"
            if t < time(16, 0):
                return i, "close"
            continue
        return i, "open"
    return None


def _ret(bars: dict, sessions: list[str], i: int, kind: str, h: int) -> float | None:
    j = i + h - 1 if kind == "open" else i + h
    if j >= len(sessions):
        return None
    a, b = bars.get(sessions[i]), bars.get(sessions[j])
    if not a or not b:
        return None
    start = a[0] if kind == "open" else a[1]
    if not start and kind == "open" and i:      # no open stored: the close before (takes in the overnight gap)
        prev = bars.get(sessions[i - 1])
        start = prev[1] if prev else None
    return None if not start else b[1] / float(start) - 1


def _closed(session: str, now: datetime) -> bool:
    return datetime.combine(datetime.fromisoformat(session).date(), time(16, 0), NY) <= now


def score_one(db, symbol: str, at: datetime, direction: int, cache: dict, now: datetime | None = None) -> dict | None:
    """{entry, h1/h5/h20: {ret, spy, excess, right}} with the horizons whose exit is known."""
    now = now or datetime.now(timezone.utc)
    for s in (symbol, "SPY"):
        if (s, at.date()) not in cache:
            cache[(s, at.date())] = _bars(db, s, at)
    bars, spy = cache[(symbol, at.date())], cache[("SPY", at.date())]
    sessions = [s for s in sorted(spy) if _closed(s, now)]      # a bar counts once its session has closed
    ep = entry_point(at, sessions)
    if not ep or not bars:
        return None
    i, kind = ep
    out = {"entry": {"session": sessions[i], "price": kind}}
    for h in HORIZONS:
        r, m = _ret(bars, sessions, i, kind, h), _ret(spy, sessions, i, kind, h)
        if r is None or m is None:
            continue
        x = r - m
        out[f"h{h}"] = {"ret": round(100 * r, 2), "spy": round(100 * m, 2), "excess": round(100 * x, 2),
                        "right": None if direction == 0 else (x > 0) == (direction > 0)}
    return out


def score_doc(db, doc: dict, cache: dict) -> dict:
    a = doc.get("answer") or {}
    at = doc["at"] if isinstance(doc["at"], datetime) else datetime.fromisoformat(str(doc["at"]))
    s = score_one(db, doc["symbol"], at, int(a.get("direction") or 0), cache) or {}
    others = []
    for o in a.get("other_companies") or []:
        so = score_one(db, str(o.get("symbol") or "").upper(), at, int(o.get("direction") or 0), cache)
        if so:
            others.append({"symbol": o["symbol"], "direction": o.get("direction"), **so})
    s["others"] = others
    s["complete"] = f"h{HORIZONS[-1]}" in s
    s["scored_at"] = datetime.now(timezone.utc)
    return s


def update(db=None, now: datetime | None = None) -> int:
    """Score every valid judgment at least a day old whose 20-session exit is not known yet."""
    db = db if db is not None else MongoClient(sl.MONGO_URI, tz_aware=True)["quant_data"]
    now = now or datetime.now(timezone.utc)
    cache, n = {}, 0
    for doc in db[COLLECTION].find({"status": "ok", "score.complete": {"$ne": True}, "at": {"$lt": now - timedelta(hours=12)}}):
        db[COLLECTION].update_one({"_id": doc["_id"]}, {"$set": {"score": score_doc(db, doc, cache)}})
        n += 1
    return n


def track(docs: list[dict]) -> dict:
    """Per horizon: directional calls scored, how many right, and the mean excess in the called direction."""
    out = {}
    for h in HORIZONS:
        k = f"h{h}"
        calls = [(d["answer"]["direction"], d["score"][k]) for d in docs
                 if (d.get("score") or {}).get(k) and (d.get("answer") or {}).get("direction")]
        quiet = [abs(d["score"][k]["excess"]) for d in docs
                 if (d.get("score") or {}).get(k) and (d.get("answer") or {}).get("direction") == 0]
        out[k] = {"calls": len(calls), "right": sum(bool(s["right"]) for _, s in calls),
                  "mean_called_excess": round(sum((1 if dr > 0 else -1) * s["excess"] for dr, s in calls) / len(calls), 2) if calls else None,
                  "neutral": len(quiet), "neutral_mean_abs_excess": round(sum(quiet) / len(quiet), 2) if quiet else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", help="a JSONL of judgments (the replay): score and print, store nothing")
    args = ap.parse_args()
    db = MongoClient(sl.MONGO_URI, tz_aware=True)["quant_data"]
    if args.dry:
        docs, cache = [], {}
        for l in open(args.dry):
            r = json.loads(l)
            if not r.get("valid"):
                continue
            d = {"symbol": r["symbol"], "at": datetime.fromisoformat(r["at"]), "answer": r["answer"]}
            d["score"] = score_doc(db, d, cache)
            docs.append(d)
            s = d["score"]
            print(f"{d['at']:%m-%d %H:%M} {d['symbol']:5} dir {d['answer']['direction']:+d} entry {s.get('entry')} "
                  + " ".join(f"{k} {s[k]['excess']:+.1f}{'' if s[k]['right'] is None else (' ok' if s[k]['right'] else ' MISS')}"
                             for k in ("h1", "h5", "h20") if k in s))
    else:
        print("scored", update(db))
        docs = list(db[COLLECTION].find({"status": "ok", "score": {"$exists": True}}))
    print(json.dumps(track(docs), indent=1))


if __name__ == "__main__":
    sys.exit(main())
