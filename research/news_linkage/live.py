"""N.6 step 5 (2026-11-15): the live run, every 30 minutes (Airflow news_linkage).

Each run, as of now:
  1. the symbols followed in quant_ai (holdings + watchlist, read at run time, never listed here);
  2. their articles of the last LOOKBACK_DAYS from the three sources of storylines.py, embedded (cached),
     grouped into storylines;
  3. the rules of triggers.py decide which storylines fire. The state they need (when a storyline last
     fired, with how many articles) is read back from the stored judgments: a storyline is the one fired
     before when it has the same id or holds at least half of that firing's articles, since rebuilding
     over a sliding window can renumber a storyline;
  4. Claude (judge.py, CLI on the subscription) judges each symbol that fired, at most MAX_PER_RUN a run
     and none after RUN_BUDGET; whatever is left fires again at the next check.
Every judgment is stored in quant_data.news_linkage_judgments, valid or not; quant_ai shows the latest
valid one per symbol in the holdings page's Action cell as a hint (it never changes the advice). A Claude
failure is stored as `failed`, does not count as a firing, and is retried at the next check; the run then
exits 1 so Airflow's failure notice says the judge is down (an expired login, the subscription's limit).

    python live.py                          # the scheduled run
    python live.py --dry                    # what would fire now, without calling Claude or storing
    python live.py --dry --at 2026-11-12T14:00   # the same as of a past check (no look-ahead)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests
from pymongo import MongoClient

import judge
import storylines as sl
import triggers as tg

QUANT_AI_SYMBOLS = os.getenv("QUANT_AI_SYMBOLS_URL", "http://127.0.0.1:18000/api/markets/portfolio/symbols")
COLLECTION = "news_linkage_judgments"
LOOKBACK_DAYS = 30          # storylines are rebuilt over this window (14-day joins + 7-day baselines + margin)
MAX_PER_RUN = 6
RUN_BUDGET = timedelta(minutes=20)    # no new judgment starts after this; the DAG's timeout is 28 minutes
TITLE_DF_MAX_AGE = timedelta(days=30)
MEMBERS_KEPT = 500
SAME_LINE = 0.5             # share of a past firing's articles a storyline must hold to be that storyline


def followed() -> list[str]:
    r = requests.get(QUANT_AI_SYMBOLS, timeout=10)
    r.raise_for_status()
    d = r.json()
    return sorted(set(d.get("held") or []) | set(d.get("watchlist") or []))


def _title_df(now: datetime):
    fresh = sl.TITLE_DF.exists() and now - datetime.fromtimestamp(sl.TITLE_DF.stat().st_mtime, timezone.utc) < TITLE_DF_MAX_AGE
    return sl.title_df(end=now, refresh=not fresh)


def _members(line: sl.Storyline) -> list[str]:
    return [a.id for a in line.articles if a.mark != "repeat"][-MEMBERS_KEPT:]


def prev_firing(line: sl.Storyline, history: list[dict]) -> tuple[datetime, int] | None:
    """(time, n24) of the latest stored firing of this storyline. `history`: the symbol's counted
    judgments, newest first."""
    mine = set(_members(line))
    for doc in history:
        for s in doc.get("storylines") or []:
            theirs = s.get("members") or []
            if s.get("id") == line.id or (theirs and len(mine.intersection(theirs)) >= SAME_LINE * len(theirs)):
                return doc["at"], int(s.get("n24") or 0)
    return None


def _others(lines: list[sl.Storyline], hot_ids: set[str], t: datetime) -> list[sl.Storyline]:
    out = [l for l in lines if l.id not in hot_ids and any(t - timedelta(days=3) <= a.at < t for a in l.articles)
           and sum(a.mark == "progress" and a.at < t for a in l.articles) >= 3]
    return sorted(out, key=lambda l: -l.count("progress"))


def _line_doc(line: sl.Storyline, h: tg.Hot) -> dict:
    return {"id": line.id, "reason": h.reason, "n24": h.n24, "sites24": h.sites24, "base": h.base, "head": line.head,
            "first_at": line.first_at, "last_at": line.last_at, "progress": line.count("progress"),
            "n_articles": len(line.articles), "members": _members(line)}


def run(now: datetime, dry: bool = False) -> int:
    started = time.monotonic()
    db = MongoClient(sl.MONGO_URI, tz_aware=True)["quant_data"]
    coll = db[COLLECTION]
    syms = followed()
    arts = sl.load_articles(syms, now - timedelta(days=LOOKBACK_DAYS), now)
    vecs = sl.embed(arts)
    lines = sl.build(arts, vecs, _title_df(now))
    by_symbol: dict[str, list[sl.Storyline]] = {}
    for l in lines:
        by_symbol.setdefault(l.symbol, []).append(l)
    print(f"{now:%Y-%m-%d %H:%M} UTC: {len(syms)} symbols, {len(arts)} articles, {len(lines)} storylines "
          f"({time.monotonic() - started:.0f}s)", flush=True)

    fired = []
    for sym, ls in by_symbol.items():
        history = list(coll.find({"symbol": sym, "status": {"$in": ["ok", "invalid"]}, "at": {"$lt": now, "$gte": now - timedelta(days=LOOKBACK_DAYS)}},
                                 {"at": 1, "storylines.id": 1, "storylines.n24": 1, "storylines.members": 1}).sort("at", -1))
        if history and now - history[0]["at"] < tg.SYMBOL_GAP:
            continue
        hot = tg.fire(ls, now, lambda l: prev_firing(l, history))
        if hot:
            fired.append((sym, ls, hot))
    fired.sort(key=lambda x: -max(h.n24 for h in x[2]))

    failures = 0
    for k, (sym, ls, hot) in enumerate(fired):
        by_id = {l.id: l for l in ls}
        hot_lines = [by_id[h.storyline] for h in hot]
        for l in hot_lines:
            l.head = sl.headline(l, arts, vecs)
        desc = "; ".join(f"{h.reason} {h.n24}/24h {by_id[h.storyline].head[:70]}" for h in hot)
        if dry:
            print(f"  would judge {sym}: {desc}")
            continue
        if k >= MAX_PER_RUN or time.monotonic() - started > RUN_BUDGET.total_seconds():
            print(f"  {sym} left for the next check: {desc}")
            continue
        others = _others(ls, {l.id for l in hot_lines}, now)
        for l in others:
            l.head = sl.headline(l, arts, vecs)
        doc = {"_id": f"{sym}:{now:%Y%m%dT%H%M}", "symbol": sym, "at": now,
               "storylines": [_line_doc(by_id[h.storyline], h) for h in hot]}
        try:
            r = judge.judge(sym, now, hot_lines, others)
            for key in ("at", "symbol", "storylines"):      # the doc's own fields carry these
                r.pop(key, None)
            doc.update(r, status="ok" if r["valid"] else "invalid")
            a = r["answer"] or {}
            print(f"  {sym} {doc['status']} dir={a.get('direction')} conf={a.get('confidence')} {r['duration_ms']}ms: {desc}", flush=True)
        except Exception as e:  # noqa: BLE001 - one symbol's failure must not stop the others
            failures += 1
            doc.update(status="failed", error=f"{type(e).__name__}: {e}"[:500], judged_at=datetime.now(timezone.utc).isoformat())
            print(f"  {sym} FAILED {doc['error']}", flush=True)
        coll.replace_one({"_id": doc["_id"]}, doc, upsert=True)
    if not fired:
        print("  nothing fired")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="only list what would fire")
    ap.add_argument("--at", help="check as of this UTC time (needs --dry)")
    args = ap.parse_args()
    if args.at and not args.dry:
        sys.exit("--at is for --dry only: a past check must not store judgments")
    now = datetime.fromisoformat(args.at).replace(tzinfo=timezone.utc) if args.at else datetime.now(timezone.utc)
    sys.exit(run(now.replace(second=0, microsecond=0), dry=args.dry))


if __name__ == "__main__":
    main()
