"""Q.17.2 ablations: do the v3 student labels (2a) or sector-residual targets (2b) improve what the
news study and the 5-session barrier model find? (PROJECT_PLAN Q.17.2, 2026-11-18)

Nothing here touches the live files in ADVICE_DIR: every output goes to Q17_DIR/<step>/<variant>/.

Step 2a, labels only. Three variants of news_days.parquet, each fed to the unchanged news_events study
and to the unchanged barrier features:
  v2     production: news_articles_company_matched_v2 with the A labels (what is live today)
  v2dd   the same with the syndication dedupe of daily_symbol_features (symbol, day, title), so a
         difference between v2 and v3 is not just the dedupe
  v3     news_articles.student_gd1 (the GD1 students): kept when student_gd1.relevant is true (the
         key-person rule already folded in), prefilter.junk_reason() is None, and the company is named
         in the title or the first 1,500 characters (slm_company_match_v2._has_direct_mention), unless
         the key-person rule is why it is relevant; same dedupe; sentiment / event type / strength from
         the student
  v3n    v3 with the mention test widened to the company's short name, case-sensitive ("Intel", "Oracle",
         "KLA"): the keyword lists of some names leave the bare name out on purpose (it is ambiguous for
         collection), and keywords alone drop most of their relevant rows (INTC: 1,876 of 2,000)

  news      scan both collections once, build the three news_days, run the study on each, compare
  barrier   build the 5-session barrier dataset per variant -> one parquet for Colab (barrier_5.parquet)
  score     read Colab's walk-forward predictions: Brier per variant, paired date-bootstrap CI vs v2

Usage: .venv311/bin/python research/models/q17_ablation.py news|barrier|score
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "labeling"))

Q17_DIR = Path(os.getenv("Q17_DIR", "/Volumes/Data4T/q17"))
STEP = Q17_DIR / "2a"
VARIANTS = ("v2", "v2dd", "v3", "v3n")
CONTENT_CHARS = 1500                      # what slm_company_match_v2 reads (V2_CONTENT_CHARS)
# the 20-session kinds with |t| >= 3 in the live study (news_study.json, 2026-11-02): acceptance needs them kept
CORE_KINDS = ("guidance cut", "guidance raise", "bad earnings news", "high-impact bad news",
              "bad news, fell less", "fell 5%+ without bad news")


def _mongo():
    from dotenv import load_dotenv
    from pymongo import MongoClient
    load_dotenv(HERE.parents[1] / ".env")
    uri = (os.getenv("HOST_LOCAL_MONGO_URI") or os.getenv("LOCAL_MONGO_URI") or "mongodb://root:root@127.0.0.1:37018/")
    return MongoClient(uri.replace("host.docker.internal", "127.0.0.1"), serverSelectionTimeoutMS=8000)["quant_data"]


def _core() -> list[str]:
    import news_events as ne
    return sorted(p.name.split(".")[0] for p in ne.EP_DIR.glob("*.daily.parquet") if not p.name.startswith("SPY."))


_SUFFIX = re.compile(r"[,.]?\s+(?:Corporation|Corp\.?|Inc\.?|Incorporated|Technologies|Technology|Platforms|Holdings|Company|Co\.|plc|Ltd\.?|Group)$")


def _short_name(name: str) -> str:
    prev = None
    while prev != name:
        prev, name = name, _SUFFIX.sub("", name).strip()
    return name


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """daily_symbol_features._dedupe_syndication: syndicated copies differ only by URL host."""
    day = df["date"].astype(str).str[:8]
    return df.loc[~pd.DataFrame({"s": df["symbol"], "d": day, "t": df["title"].fillna("")}).duplicated()]


def _scan_v2(core: list[str]) -> pd.DataFrame:
    import news_events as ne
    col = _mongo()["news_articles_company_matched_v2"]
    rows = list(col.find({"symbol": {"$in": core}}, ne.PROJ, batch_size=5000))
    return pd.DataFrame(rows)


def _scan_v3(core: list[str]) -> tuple[pd.DataFrame, dict]:
    """One pass over news_articles (no symbol index there): the student's relevant rows for the core names,
    with production's prefilter and direct-mention gate applied as each row streams in."""
    import prefilter
    import slm_company_match_v2 as m2
    col = _mongo()["news_articles"]
    pipe = [{"$match": {"symbol": {"$in": core}, "student_gd1.relevant": True}},
            {"$project": {"_id": 0, "symbol": 1, "date": 1, "title": 1, "url": 1,
                          "body": {"$substrCP": [{"$ifNull": ["$content", ""]}, 0, CONTENT_CHARS]},
                          "s": "$student_gd1.sentiment", "e": "$student_gd1.event_type",
                          "st": "$student_gd1.signal_strength", "rr": "$student_gd1.rel_rule"}}]
    keep, stats = [], {"relevant": 0, "junk": 0, "no_mention": 0, "name_only": 0, "key_person_kept": 0}
    short = {s: re.compile(r"\b" + re.escape(_short_name(m2.COMPANY_UNIVERSE.get(s, (s, []))[0])) + r"\b") for s in core}
    for d in col.aggregate(pipe, allowDiskUse=True, batchSize=5000):
        stats["relevant"] += 1
        sym, title, body = d.get("symbol"), d.get("title") or "", d.get("body") or ""
        name, kws = m2.COMPANY_UNIVERSE.get(sym, (sym, []))
        if prefilter.junk_reason(title, d.get("url") or "", [name, *(k for k in kws if isinstance(k, str))], sym, body):
            stats["junk"] += 1
            continue
        key_person = str(d.get("rr") or "").startswith("key_person")
        kw = sym not in m2._SCAN_PATTERNS or m2._has_direct_mention(sym, title, body)
        nm = bool(short[sym].search(title) or short[sym].search(body))
        if not (kw or nm or key_person):
            stats["no_mention"] += 1
            continue
        stats["key_person_kept"] += int(key_person and not (kw or nm))
        stats["name_only"] += int(nm and not kw and not key_person)
        keep.append({"symbol": sym, "date": d.get("date"), "title": title, "llm_sentiment_final": d.get("s"),
                     "llm_event_type_a": d.get("e"), "llm_signal_strength_a": d.get("st"), "strict": bool(kw or key_person)})
        if stats["relevant"] % 100000 == 0:
            print(f"  v3 scan: {stats}", flush=True)
    return pd.DataFrame(keep), stats


def _news_days(arts: pd.DataFrame, core: list[str]) -> pd.DataFrame:
    import news_events as ne
    parts = []
    for s, g in arts.groupby("symbol"):
        if s not in core:
            continue
        r = ne._days_for(s, g, ne._sessions(s).index)
        if r is not None:
            parts.append(r)
    return pd.concat(parts, ignore_index=True)


def news() -> dict:
    import news_events as ne
    core = _core()
    print(f"{len(core)} core names", flush=True)
    v2 = _scan_v2(core)
    print(f"v2: {len(v2):,} rows", flush=True)
    v3, stats = _scan_v3(core)
    print(f"v3: {len(v3):,} rows kept; {stats}", flush=True)
    arts = {"v2": v2, "v2dd": _dedupe(v2), "v3": _dedupe(v3[v3.strict]), "v3n": _dedupe(v3)}
    rep = {"articles": {k: int(len(a)) for k, a in arts.items()}, "v3_filter": stats, "study": {}}
    for k, a in arts.items():
        d = STEP / k
        d.mkdir(parents=True, exist_ok=True)
        nd = _news_days(a, core)
        nd.to_parquet(d / "news_days.parquet")
        ne.OUT = d                               # study() reads and writes OUT
        print(f"\n== {k}: {int(nd.n.sum()):,} articles on {len(nd):,} stock-sessions", flush=True)
        rep["study"][k] = ne.study()
    rep["core_kinds_20d"] = {kind: {k: rep["study"][k][kind]["20d_vs_all"] for k in arts} for kind in CORE_KINDS}
    rep["core_kept"] = {k: all(abs(rep["study"][k][kind]["20d_vs_all"].get("t", 0)) >= 3.0 for kind in CORE_KINDS) for k in ("v3", "v3n")}
    (STEP / "news_report.json").write_text(json.dumps(rep, indent=1))
    print("\n20-session excess vs all days (mean %, t):")
    for kind, v in rep["core_kinds_20d"].items():
        print(f"  {kind:28s} " + " | ".join(f"{k}: {x.get('mean')} (t {x.get('t')})" for k, x in v.items()))
    print("all core kinds keep |t| >= 3:", rep["core_kept"])
    return rep


def barrier() -> Path:
    """The 5-session barrier dataset per variant, joined into one file: shared features once, the ten news
    features once per variant (suffixed), the label, and what the score needs."""
    import barrier_model as bm
    bm.SETS = {"5": bm.SETS["5"]}
    live_dataset = Path(os.getenv("ADVICE_DIR", "/Volumes/Data4T/daily_advice")) / "dataset.parquet"
    frames = {}
    for k in VARIANTS:
        d = STEP / k
        link = d / "dataset.parquet"
        if not link.exists():
            link.symlink_to(live_dataset)        # read only; build() reads OUT/dataset.parquet
        bm.OUT = d
        built = d / "barrier_dataset.parquet"
        frames[k] = pd.read_parquet(built) if built.exists() and os.getenv("Q17_REBUILD") != "1" else bm.build()
    shared = [c for c in bm.FEATS if c not in bm.NEWS_FEATS]
    base = frames["v2"][list(dict.fromkeys(["symbol", "date", "b5", "dd_252", "atr_pct", *shared]))].copy()
    for k in VARIANTS:
        f = frames[k][["symbol", "date", *bm.NEWS_FEATS]].rename(columns={c: f"{c}__{k}" for c in bm.NEWS_FEATS})
        base = base.merge(f, on=["symbol", "date"], how="left", validate="one_to_one")
    num = base.select_dtypes("float64").columns
    base[num] = base[num].astype("float32")
    out = STEP / "barrier_5.parquet"
    base.to_parquet(out)
    (STEP / "barrier_5_meta.json").write_text(json.dumps({"shared": shared, "news": list(bm.NEWS_FEATS), "variants": list(VARIANTS),
                                                          "set": bm.SETS["5"], "first_test_year": 2019, "seed": 7}, indent=1))
    print(f"{out}: {len(base):,} rows, {out.stat().st_size / 1e6:.1f} MB")
    return out


def diag() -> dict:
    """Where v3 loses the bad-news signal: the student's relevance or its labels. On production's article
    set (v2dd, joined by _id: a v2 row keeps its news_articles _id):
      dF   production labels, only rows the student calls relevant
      dL   all rows, the student's labels instead of production's"""
    import news_events as ne
    core = _core()
    db = _mongo()
    v2 = pd.DataFrame(list(db["news_articles_company_matched_v2"].find({"symbol": {"$in": core}}, {**ne.PROJ, "_id": 1}, batch_size=5000)))
    st = pd.DataFrame(list(db["news_articles"].aggregate([
        {"$match": {"symbol": {"$in": core}, "student_gd1": {"$exists": True}}},
        {"$project": {"_id": 1, "rel": "$student_gd1.relevant", "s": "$student_gd1.sentiment",
                      "e": "$student_gd1.event_type", "st": "$student_gd1.signal_strength"}}], allowDiskUse=True, batchSize=5000)))
    x = _dedupe(v2).merge(st, on="_id", how="left")
    rep = {"rows": int(len(x)), "with_student": int(x.s.notna().sum()), "student_relevant": int((x.rel == True).sum()), "study": {}}  # noqa: E712
    dl = x[x.s.notna()].assign(llm_sentiment_final=lambda d: d.s, llm_event_type_a=lambda d: d.e, llm_signal_strength_a=lambda d: d.st)
    agree = x[x.s.notna()]
    rep["label_agreement"] = {
        "event_type": round(float((agree.e == agree.llm_event_type_a).mean() * 100), 1),
        "sentiment_sign": round(float((np.sign(pd.to_numeric(agree.s, errors="coerce").round(1)) ==
                                       np.sign(pd.to_numeric(agree.llm_sentiment_final, errors="coerce").round(1))).mean() * 100), 1),
        "strength": round(float((agree.st == agree.llm_signal_strength_a).mean() * 100), 1),
        "neg_earnings_prod": int(((pd.to_numeric(agree.llm_sentiment_final, errors="coerce") <= -0.3) & (agree.llm_event_type_a == "earnings")).sum()),
        "neg_earnings_student": int(((pd.to_numeric(agree.s, errors="coerce") <= -0.3) & (agree.e == "earnings")).sum())}
    for k, a in {"dF": x[x.rel == True], "dL": dl}.items():  # noqa: E712
        d = STEP / k
        d.mkdir(parents=True, exist_ok=True)
        nd = _news_days(a, core)
        nd.to_parquet(d / "news_days.parquet")
        ne.OUT = d
        print(f"\n== {k}: {int(nd.n.sum()):,} articles on {len(nd):,} stock-sessions", flush=True)
        rep["study"][k] = ne.study()
    base = json.loads((STEP / "news_report.json").read_text())["study"]
    rep["core_kinds_20d"] = {kind: {**{k: base[k][kind]["20d_vs_all"] for k in ("v2dd", "v3")},
                                    **{k: rep["study"][k][kind]["20d_vs_all"] for k in ("dF", "dL")}} for kind in CORE_KINDS}
    (STEP / "diag_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "study"}, indent=1, default=str))
    return rep


def _brier_rows(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.sum((P - np.eye(3)[y]) ** 2, axis=1)


def score(path: Path | None = None, n_boot: int = 2000) -> dict:
    """Brier per variant on the same walk-forward days, and Brier(variant) - Brier(v2) with a 95% CI from
    resampling dates (days of one date move together)."""
    pr = pd.read_parquet(path or STEP / "colab" / "preds_5.parquet")
    y = pr["b5"].astype(int).values
    cls = ("lower", "none", "upper")
    rows = {k: _brier_rows(y, pr[[f"p_{c}__{k}" for c in cls]].values) for k in VARIANTS}
    rows["base"] = _brier_rows(y, pr[[f"base_{c}" for c in cls]].values)
    dates = pr["date"].values
    rng = np.random.default_rng(7)
    ud, inv = np.unique(dates, return_inverse=True)
    rep = {"days": int(len(pr)), "dates": int(len(ud)), "brier": {k: round(float(v.mean()), 4) for k, v in rows.items()}, "vs_v2": {}}
    cnt = np.bincount(inv)
    for k in ("v2dd", "v3", "v3n", "base"):
        diff = rows[k] - rows["v2"]
        per_date = np.bincount(inv, weights=diff)
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(ud), len(ud))
            boots.append(per_date[pick].sum() / cnt[pick].sum())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rep["vs_v2"][k] = {"diff": round(float(diff.mean()), 5), "ci95": [round(float(lo), 5), round(float(hi), 5)]}
    rep["live_reference"] = 0.6271
    rep["passes"] = {k: rep["brier"][k] < 0.6271 and rep["vs_v2"][k]["ci95"][1] < 0 for k in ("v3", "v3n")}
    (STEP / "barrier_score.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


# ---------------------------------------------------------------- 2b: sector residuals (target only)

STEP2B = Q17_DIR / "2b"
ETFS = ("XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SOXX", "SMH", "IGV", "XBI", "KRE")
SECTOR_ETF = {"Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV", "Energy": "XLE", "Industrials": "XLI",
              "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Utilities": "XLU", "Basic Materials": "XLB",
              "Real Estate": "XLRE", "Communication Services": "XLC"}
# industries with their own fund; SOXX rather than SMH: SMH is ~20% NVDA, so NVDA's residual would cancel itself
INDUSTRY_ETF = {"Semiconductors": "SOXX", "Semiconductor Equipment & Materials": "SOXX", "Software - Application": "IGV",
                "Software - Infrastructure": "IGV", "Biotechnology": "XBI", "Banks - Regional": "KRE"}
BETA_WINDOW, BETA_CLIP = 60, (0.2, 2.5)
SECTOR_FEATS = ["etf_r1", "etf_r5", "etf_r20", "resid_r1", "resid_r5", "resid_r20", "beta60"]


def fetch2b() -> None:
    """Sector / industry of the core names (yfinance .info) and the funds' daily bars, from 2015-06 as the
    stocks' (earnings_policy.fetch, auto_adjust=False). Kept under STEP2B, not EP_DIR: the stock list is
    read from EP_DIR's files."""
    import time
    import yfinance as yf
    (STEP2B / "etf").mkdir(parents=True, exist_ok=True)
    sec = {}
    for s in _core():
        info = yf.Ticker(s).info or {}
        sec[s] = {"sector": info.get("sector"), "industry": info.get("industry")}
        time.sleep(0.4)
    (STEP2B / "core_sectors.json").write_text(json.dumps(sec, indent=1))
    for e in ETFS:
        d = yf.Ticker(e).history(start="2015-06-01", interval="1d", auto_adjust=False)
        d.to_parquet(STEP2B / "etf" / f"{e}.daily.parquet")
        print(f"  {e}: {len(d):,} bars from {d.index.min():%Y-%m-%d}", flush=True)
    print(json.dumps({s: v for s, v in sec.items() if not v.get("sector")}, indent=1), "missing sector")


def _fund(sym: str, sec: dict) -> str:
    v = sec.get(sym) or {}
    return INDUSTRY_ETF.get(v.get("industry")) or SECTOR_ETF.get(v.get("sector")) or "SPY"


def _bars(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)[["Open", "Close"]]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d[~d.index.duplicated()].sort_index().dropna()


def _bench(sym: str, idx: pd.DatetimeIndex, sec: dict) -> tuple[pd.DataFrame, str]:
    """The stock's fund on its sessions, SPY where the fund has no bar yet (XLC before 2018-06)."""
    import news_events as ne
    spy = ne._sessions("SPY").reindex(idx)
    f = _fund(sym, sec)
    if f == "SPY":
        return spy, f
    e = _bars(STEP2B / "etf" / f"{f}.daily.parquet").reindex(idx)
    return e.where(e.Close.notna(), spy), f


def _beta(r: pd.Series, rb: pd.Series) -> pd.Series:
    """Rolling 60-session beta of the stock on its fund, known at the close before (shifted one session)."""
    cov = r.rolling(BETA_WINDOW, min_periods=40).cov(rb)
    var = rb.rolling(BETA_WINDOW, min_periods=40).var()
    return (cov / var).clip(*BETA_CLIP).shift(1)


def _forward_resid() -> pd.DataFrame:
    """news_events._forward with the fund in SPY's place and beta on it: x / n are 100 x (r - beta x r_fund)."""
    import news_events as ne
    sec = json.loads((STEP2B / "core_sectors.json").read_text())
    parts = []
    for s in _core():
        d = ne._sessions(s)
        b, _ = _bench(s, d.index, sec)
        beta = _beta(d.Close.pct_change(), b.Close.pct_change()).fillna(1.0)
        f = pd.DataFrame(index=d.index)
        f["day_ret"] = (d.Close / d.Close.shift() - 1) * 100
        for h in ne.HORIZONS:
            r = d.Close.shift(-(h - 1)) / d.Open - 1
            m = b.Close.shift(-(h - 1)) / b.Open - 1
            f[f"x{h}"] = (r - beta * m) * 100
            r = d.Close.shift(-h) / d.Open.shift(-1) - 1
            m = b.Close.shift(-h) / b.Open.shift(-1) - 1
            f[f"n{h}"] = (r - beta * m) * 100
        f["symbol"], f["session"] = s, d.index
        parts.append(f[f.index >= "2016-01-01"])
    return pd.concat(parts, ignore_index=True)


def sector_features(sym: str, idx: pd.DatetimeIndex, sec: dict) -> pd.DataFrame:
    """Known at the close of each session (the barrier model enters at the next open)."""
    import news_events as ne
    d = ne._sessions(sym).reindex(idx)
    b, _ = _bench(sym, idx, sec)
    r, rb = d.Close.pct_change(), b.Close.pct_change()
    beta = _beta(r, rb)
    f = pd.DataFrame(index=idx)
    for w in (1, 5, 20):
        er = b.Close / b.Close.shift(w) - 1
        f[f"etf_r{w}"] = er * 100
        f[f"resid_r{w}"] = ((d.Close / d.Close.shift(w) - 1) - beta.fillna(1.0) * er) * 100
    f["beta60"] = beta
    return f


def news2b() -> dict:
    """The news study on production's news days (2a's v2) with sector-residual returns as the target."""
    import shutil
    import news_events as ne
    d = STEP2B / "v2r"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(STEP / "v2" / "news_days.parquet", d / "news_days.parquet")
    ne.OUT, ne._forward = d, _forward_resid
    rep = ne.study()
    base = json.loads((STEP / "news_report.json").read_text())["study"]["v2"]
    out = {"core_kinds_20d": {k: {"spy_excess": base[k]["20d_vs_all"], "sector_residual": rep[k]["20d_vs_all"]} for k in CORE_KINDS},
           "core_kept": all(abs(rep[k]["20d_vs_all"].get("t", 0)) >= 3.0 for k in CORE_KINDS)}
    (STEP2B / "news_report.json").write_text(json.dumps({**out, "study": rep}, indent=1))
    for k, v in out["core_kinds_20d"].items():
        print(f"  {k:28s} SPY: {v['spy_excess'].get('mean')} (t {v['spy_excess'].get('t')}) | sector: {v['sector_residual'].get('mean')} (t {v['sector_residual'].get('t')})")
    print("all core kinds keep |t| >= 3:", out["core_kept"])
    return out


def barrier2b() -> Path:
    """2a's v2 barrier dataset (production news features) plus the seven sector features: variants v2 and v2s."""
    import barrier_model as bm
    sec = json.loads((STEP2B / "core_sectors.json").read_text())
    base = pd.read_parquet(STEP / "v2" / "barrier_dataset.parquet")
    parts = []
    for s, g in base.groupby("symbol"):
        f = sector_features(s, pd.DatetimeIndex(g.date), sec)
        f["symbol"], f["date"] = s, f.index
        parts.append(f.reset_index(drop=True))
    x = base[list(dict.fromkeys(["symbol", "date", "b5", "dd_252", "atr_pct", *bm.FEATS]))].merge(
        pd.concat(parts, ignore_index=True), on=["symbol", "date"], how="left", validate="one_to_one")
    num = x.select_dtypes("float64").columns
    x[num] = x[num].astype("float32")
    out = STEP2B / "barrier_5.parquet"
    x.to_parquet(out)
    (STEP2B / "barrier_5_meta.json").write_text(json.dumps({"variant_feats": {"v2": list(bm.FEATS), "v2s": list(bm.FEATS) + SECTOR_FEATS},
                                                            "first_test_year": 2019, "seed": 7}, indent=1))
    print(f"{out}: {len(x):,} rows, {out.stat().st_size / 1e6:.1f} MB; funds:",
          dict(pd.Series({s: _fund(s, sec) for s in base.symbol.unique()}).value_counts()))
    return out


def score2b(n_boot: int = 2000) -> dict:
    pr = pd.read_parquet(STEP2B / "colab" / "preds_5.parquet")
    y = pr["b5"].astype(int).values
    cls = ("lower", "none", "upper")
    rows = {k: _brier_rows(y, pr[[f"p_{c}__{k}" for c in cls]].values) for k in ("v2", "v2s")}
    rows["base"] = _brier_rows(y, pr[[f"base_{c}" for c in cls]].values)
    ud, inv = np.unique(pr["date"].values, return_inverse=True)
    cnt, rng = np.bincount(inv), np.random.default_rng(7)
    rep = {"days": int(len(pr)), "brier": {k: round(float(v.mean()), 4) for k, v in rows.items()}, "vs_v2": {}}
    for k in ("v2s", "base"):
        per = np.bincount(inv, weights=rows[k] - rows["v2"])
        boots = [per[p].sum() / cnt[p].sum() for p in (rng.integers(0, len(ud), len(ud)) for _ in range(n_boot))]
        rep["vs_v2"][k] = {"diff": round(float((rows[k] - rows["v2"]).mean()), 5),
                           "ci95": [round(float(v), 5) for v in np.percentile(boots, [2.5, 97.5])]}
    rep["passes"] = rep["brier"]["v2s"] < 0.6271 and rep["vs_v2"]["v2s"]["ci95"][1] < 0
    (STEP2B / "barrier_score.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    {"news": news, "barrier": barrier, "score": score, "diag": diag, "fetch2b": fetch2b, "news2b": news2b,
     "barrier2b": barrier2b, "score2b": score2b}[sys.argv[1] if len(sys.argv) > 1 else "news"]()
