"""Coverage watch (PROJECT_PLAN N.8 improvement plan, step 4; 2026-11-19, the user's request).

Why: a collection or matching gap stays silent. The v2 direct-mention gate of 2026-11-04 dropped most of
INTC's and ORCL's articles and was found by chance two weeks later; on 2026-11-19 this check's first
replay found AMD (382 of 730 collected articles in 10 days rejected because the gate's list had no "AMD")
and CSCO ("Cisco") the same way.

core (the rule files in collectors/news/gdelt/company_rules): per company the articles collected
(news_articles, not universe_backfill) and kept (news_articles_company_matched_v2) in the 7 days to
yesterday (UTC), against the median of the 8 weeks before ("usual"). Flags:
  kept_drop        kept below half the usual (usual >= MIN_USUAL)
  collected_drop   collected below half the usual (usual >= MIN_USUAL)
  zero             nothing kept although the usual is >= ZERO_USUAL
  rule_loss        the week's collected-but-not-kept articles replayed through the production checks: a
                   rule (the direct-mention gate, one prefilter rule) rejected >= max(RULE_MIN, RULE_SHARE of
                   collected) whose headline names the company as the collector does or by its ticker (AMD
                   382, CSCO 94 in 8-17 Nov) - a fault in a rule, not a judgement
  keep_rate_drop   kept / collected below half the usual rate with >= MIN_COLLECTED collected; a notice,
                   not an alert: in Nov 2026 it fired for ~4 companies a day, mostly the relevance model
                   being stricter on a grown collection
Each company carries the replay's reasons (prefilter rule / gate / relevance model) and, when flagged,
samples of kept and lost headlines.
universe: rows per month (the backfill fills complete months), the last complete month against the median
of the 6 before. Flagged when it is empty although the usual is >= ZERO_USUAL_MONTH, or below a quarter of
a usual >= MIN_USUAL_MONTH: these companies have 10-30 articles a month and swing by half from month to
month (on 2026-11-19 a below-half rule flagged 15 of 489, none of them a fault but SYY's empty October).

-> <OUT>/latest.json and <OUT>/history/<date>.json; quant_ai serves it to quant_ui's Keywords tab.
Usage: .venv311/bin/python tools/coverage_watch.py [--asof YYYY-MM-DD] [--dry-run]
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "labeling"))
os.environ.setdefault("MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")

OUT = Path(os.getenv("COVERAGE_WATCH_DIR", "/Volumes/Data4T/coverage_watch"))
CORE_RULES = ROOT / "collectors" / "news" / "gdelt" / "company_rules"
UNIVERSE_RULES = ROOT / "collectors" / "news" / "gdelt" / "universe_rules" / "company_rules"
WEEKS = 8
MIN_USUAL, MIN_COLLECTED, ZERO_USUAL = 10, 50, 3
RULE_MIN, RULE_SHARE = 10, 0.1
ALERTS = {"kept_drop", "collected_drop", "zero", "rule_loss", "month_drop"}
UNIVERSE_MONTHS, MIN_USUAL_MONTH, ZERO_USUAL_MONTH = 6, 30, 10
SAMPLES = 6


def _names(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            out[f.stem] = json.loads(f.read_text()).get("name") or f.stem
        except ValueError:
            continue
    return out


def counts(db, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily (symbol, day, n) of collected core articles and of kept ones, start <= day < end (YYYYMMDD)."""
    import slm_company_match_v2 as m2

    def agg(col, extra: dict) -> pd.DataFrame:
        rows = col.aggregate([{"$match": {"date": {"$gte": start, "$lt": end}, **extra}},
                              {"$group": {"_id": {"s": "$symbol", "d": {"$substr": ["$date", 0, 8]}}, "n": {"$sum": 1}}}],
                             allowDiskUse=True)
        return pd.DataFrame([{"symbol": r["_id"].get("s"), "day": r["_id"]["d"], "n": r["n"]} for r in rows],
                            columns=["symbol", "day", "n"])
    return (agg(db[m2.SOURCE_COLLECTION], {"origin": {"$ne": m2.UNIVERSE_ORIGIN}}), agg(db[m2.TARGET_COLLECTION], {}))


def evaluate(collected: pd.DataFrame, kept: pd.DataFrame, asof: datetime, symbols: list) -> dict:
    """{symbol: row} for the 7 days ending the day before `asof`; row["flags"] lists what is wrong."""
    days = pd.date_range(asof - timedelta(days=7 * (WEEKS + 1)), asof - timedelta(days=1)).strftime("%Y%m%d")

    def grid(df):
        g = df[df.symbol.isin(symbols)].pivot_table(index="day", columns="symbol", values="n", aggfunc="sum")
        return g.reindex(index=days, columns=symbols).fillna(0)
    c, k = grid(collected), grid(kept)
    wk = np.arange(len(days))[::-1] // 7                     # 0 = the last 7 days, 1 = the week before, ...
    cw, kw = c.groupby(wk).sum(), k.groupby(wk).sum()
    out = {}
    for s in symbols:
        c7, k7 = float(cw.loc[0, s]), float(kw.loc[0, s])
        uc, uk = float(cw.loc[1:, s].median()), float(kw.loc[1:, s].median())
        rates = (kw.loc[1:, s] / cw.loc[1:, s].replace(0, np.nan)).dropna()
        ur = float(rates.median()) if len(rates) else None
        r7 = k7 / c7 if c7 else None
        flags = []
        if uk >= MIN_USUAL and k7 < 0.5 * uk:
            flags.append({"code": "kept_drop", "text": f"kept {k7:.0f} in 7 days, usually {uk:.0f}"})
        if uc >= MIN_USUAL and c7 < 0.5 * uc:
            flags.append({"code": "collected_drop", "text": f"collected {c7:.0f} in 7 days, usually {uc:.0f}"})
        if ur and r7 is not None and c7 >= MIN_COLLECTED and r7 < 0.5 * ur:
            flags.append({"code": "keep_rate_drop", "text": f"kept {r7:.0%} of what was collected, usually {ur:.0%}"})
        if k7 == 0 and uk >= ZERO_USUAL:
            flags.append({"code": "zero", "text": f"nothing kept in 7 days, usually {uk:.0f}"})
        out[s] = {"collected_7d": int(c7), "kept_7d": int(k7), "usual_collected": round(uc, 1), "usual_kept": round(uk, 1),
                  "keep_rate_7d": round(r7, 3) if r7 is not None else None, "usual_keep_rate": round(ur, 3) if ur else None,
                  "weeks": [[int(cw.loc[i, s]), int(kw.loc[i, s])] for i in range(WEEKS, -1, -1)], "flags": flags}
    return out


def _collector_names(symbols: list):
    """{symbol: pattern} of the core rule's primary keywords and the ticker (>= 3 letters), as written."""
    out = {}
    for sym in symbols:
        try:
            kws = [k for k in json.loads((CORE_RULES / f"{sym}.json").read_text()).get("primary_keywords") or []
                   if isinstance(k, str) and len(k.strip()) >= 3]
        except (OSError, ValueError):
            continue
        kws += [sym] if len(sym) >= 3 and sym.isalpha() and sym not in kws else []
        if kws:
            out[sym] = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(kws, key=len, reverse=True)) + r")(?!\w)")
    return out


def why_lost(db, symbols: list, start: str, end: str) -> dict:
    """{symbol: {"why", "named_by_rule", "lost", "kept"}}: each collected-but-not-kept article of the window
    replayed through the production checks (prefilter, direct-mention gate); the rest the model rejected.
    named_by_rule: {rule: n} of those a rule rejected although the headline names the company as the
    collector does."""
    import slm_company_match_v2 as m2
    kept: dict = {}
    for r in db[m2.TARGET_COLLECTION].find({"symbol": {"$in": symbols}, "date": {"$gte": start, "$lt": end}},
                                           {"symbol": 1, "url": 1, "title": 1, "date": 1}):
        kept.setdefault(r["symbol"], {})[r["url"]] = r
    names = _collector_names(symbols)
    out = {s: {"why": Counter(), "named_by_rule": Counter(), "lost": []} for s in symbols}
    proj = {"symbol": 1, "url": 1, "date": 1, "title": 1, "data_quality": 1,
            "content": {"$substrCP": [{"$ifNull": ["$content", ""]}, 0, m2.CONTENT_CHARS]}}
    for d in db[m2.SOURCE_COLLECTION].aggregate([{"$match": {"symbol": {"$in": symbols}, "date": {"$gte": start, "$lt": end},
                                                              "origin": {"$ne": m2.UNIVERSE_ORIGIN}}}, {"$project": proj}],
                                                 allowDiskUse=True):
        s, o = d["symbol"], out[d["symbol"]]
        if d.get("url") in kept.get(s, {}):
            continue
        title = d.get("title") or ""
        pre = m2.prefilter_reject(d, s, "student")
        content = d.get("content") or ""
        thin = d.get("data_quality") == "title_only" or m2._is_thin_body(content)
        if pre:
            why = f"prefilter: {pre.get('model')}"
        elif s in m2._SCAN_PATTERNS and not m2._has_direct_mention(s, title, "" if thin else content):
            why = "gate: no direct mention"
        else:
            why = "relevance model"
        o["why"][why] += 1
        named = bool(names.get(s) and names[s].search(title))
        if named and why != "relevance model":
            o["named_by_rule"][why] += 1
        if len(o["lost"]) < SAMPLES * 4:
            o["lost"].append({"title": title, "url": d.get("url"), "date": (d.get("date") or "")[:8], "why": why, "named": named})
    for s in symbols:
        rows = sorted(kept.get(s, {}).values(), key=lambda r: r.get("date") or "", reverse=True)[:SAMPLES]
        o = out[s]
        o["kept"] = [{"title": r.get("title"), "url": r.get("url"), "date": (r.get("date") or "")[:8]} for r in rows]
        lost, picked = o["lost"], []
        for why, _n in o["why"].most_common():                # samples spread over the reasons, most frequent first
            picked += [x for x in lost if x["why"] == why][:max(2, SAMPLES // max(1, len(o["why"])))]
        o["lost"] = picked[:SAMPLES]
        o["why"], o["named_by_rule"] = dict(o["why"].most_common()), dict(o["named_by_rule"].most_common())
    return out


def universe_months(db, asof: datetime) -> dict:
    """{symbol: row}: the last complete month before `asof`'s month against the median of the 6 before."""
    import slm_company_match_v2 as m2
    first = (asof.replace(day=1) - pd.DateOffset(months=UNIVERSE_MONTHS + 1)).strftime("%Y%m01")
    this = asof.strftime("%Y%m01")
    rows = db[m2.SOURCE_COLLECTION].aggregate([{"$match": {"origin": m2.UNIVERSE_ORIGIN, "date": {"$gte": first, "$lt": this}}},
                                               {"$group": {"_id": {"s": "$symbol", "m": {"$substr": ["$date", 0, 6]}}, "n": {"$sum": 1}}}],
                                              allowDiskUse=True)
    df = pd.DataFrame([{"symbol": r["_id"].get("s"), "m": r["_id"]["m"], "n": r["n"]} for r in rows], columns=["symbol", "m", "n"])
    months = pd.period_range(pd.Timestamp(first), pd.Timestamp(this) - pd.Timedelta(days=1), freq="M").strftime("%Y%m")
    g = df.pivot_table(index="m", columns="symbol", values="n", aggfunc="sum").reindex(months).fillna(0)
    out = {}
    for s in g.columns:
        last, usual = float(g[s].iloc[-1]), float(g[s].iloc[:-1].median())
        bad = (last == 0 and usual >= ZERO_USUAL_MONTH) or (usual >= MIN_USUAL_MONTH and last < 0.25 * usual)
        flags = [{"code": "month_drop", "text": f"{last:.0f} articles in {months[-1][:4]}-{months[-1][4:]}, usually {usual:.0f} a month"}] if bad else []
        out[s] = {"month": f"{months[-1][:4]}-{months[-1][4:]}", "last_month": int(last), "usual_month": round(usual, 1),
                  "months": [int(x) for x in g[s].values], "flags": flags}
    return out


def run(asof: datetime, dry_run: bool = False) -> dict:
    from pymongo import MongoClient
    import slm_company_match_v2 as m2
    db = MongoClient(os.environ["MONGO_URI"])[m2.DB_NAME]
    core, uni = _names(CORE_RULES), _names(UNIVERSE_RULES)
    start = (asof - timedelta(days=7 * (WEEKS + 1))).strftime("%Y%m%d")
    end, w0 = asof.strftime("%Y%m%d"), (asof - timedelta(days=7)).strftime("%Y%m%d")
    collected, kept = counts(db, start, end)
    rows = evaluate(collected, kept, asof, sorted(core))
    detail = why_lost(db, sorted(core), w0, end)
    for s, r in rows.items():
        d = detail.get(s, {})
        r["name"], r["why"], r["named_by_rule"] = core.get(s, s), d.get("why", {}), d.get("named_by_rule", {})
        for rule, n in r["named_by_rule"].items():
            if n >= max(RULE_MIN, RULE_SHARE * r["collected_7d"]):
                r["flags"].append({"code": "rule_loss", "text": f"{rule} rejected {n} articles whose headline names the company"})
        for f in r["flags"]:
            f["level"] = "alert" if f["code"] in ALERTS else "notice"
        if r["flags"]:
            r["kept"], r["lost"] = d.get("kept", []), d.get("lost", [])
    flagged = sorted(s for s, r in rows.items() if r["flags"])
    alerts = sorted(s for s, r in rows.items() if any(f["level"] == "alert" for f in r["flags"]))
    urows = universe_months(db, asof)
    for s, r in urows.items():
        r["name"] = uni.get(s, s)
        for f in r["flags"]:
            f["level"] = "alert"
    rep = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "asof": asof.strftime("%Y-%m-%d"),
           "window": [f"{w0[:4]}-{w0[4:6]}-{w0[6:]}", (asof - timedelta(days=1)).strftime("%Y-%m-%d")],
           "core": rows, "core_flagged": flagged, "core_alerts": alerts,
           "universe": {s: r for s, r in urows.items() if r["flags"]},
           "universe_checked": len(urows), "universe_flagged": sorted(s for s, r in urows.items() if r["flags"])}
    print(f"coverage_watch {rep['asof']}: core {len(flagged)} flagged of {len(rows)} ({', '.join(flagged) or '-'}); "
          f"universe {len(rep['universe_flagged'])} of {len(urows)}", flush=True)
    for s in flagged:
        print(f"  {s}: " + "; ".join(f"[{f['level']}] {f['text']}" for f in rows[s]["flags"]) + f" | lost: {rows[s].get('why')}", flush=True)
    if not dry_run:
        (OUT / "history").mkdir(parents=True, exist_ok=True)
        text = json.dumps(rep, indent=1, ensure_ascii=False, default=str)
        (OUT / "history" / f"{rep['asof']}.json").write_text(text)
        tmp = OUT / "latest.json.tmp"
        tmp.write_text(text)
        tmp.rename(OUT / "latest.json")
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    day = datetime.strptime(a[a.index("--asof") + 1], "%Y-%m-%d") if "--asof" in a else datetime.now(timezone.utc).replace(tzinfo=None)
    run(day.replace(hour=0, minute=0, second=0, microsecond=0), dry_run="--dry-run" in a)
