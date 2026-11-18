"""Q.17.4: drop meta-labelling. After a >= 2 x ATR drop, does a small model say better than the grouped
rates what comes next? (PROJECT_PLAN Q.17.4, 2026-11-18)

Events: drop_rebound's close-to-close drops of >= 2 x ATR14 from 2018 for the 101 core names (the
decision is made at the drop's close and acted on at the next open). Train 2018-2023, test 2024-2026.

Targets (raw price, from the next open):
  lower5    the 5-session barrier's lower side (open - 1.5 x ATR) is touched first (barrier_model set "5")
  bad20     20-session excess vs SPY below -3% (news_events' n20)

Features, all known by the next open:
  drop      drop_rebound's 17 (size, gap / intraday split, report, market move, volume, trend, RSI ...)
  sector    the stock's fund that day (2b) and the stock's move net of it
  news      production labels (2a kept them) on the drop session and the next: articles, mean sentiment,
            bad / good earnings news, guidance cut, downgrade, high-impact bad news
  calendar  sessions to the next report (daily_advice), and Q.17.3's candidate: a target cut in the last
            20 sessions, net rating changes over 20
  themes    the 14 v3 themes on those two sessions (counts), only in the "+themes" variant

Models (fit on Colab): L2 logistic regression (primary, pre-set C = 0.1, standardised), the same with
themes, and a very shallow LightGBM (7 leaves) with themes, reported for reference only.
Baselines, rates from the training years: news_events' split (bad news on the drop or not) and
drop_rebound's table by cause (report x market-wide).
Accept: on 2024-2026 the primary model's Brier and log loss beat both baselines with date-bootstrap
95% CIs excluding 0, and its ECE < 3%; otherwise the grouped table goes live instead.

Usage: .venv311/bin/python research/models/q17_drops.py build|score
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q17_ablation as qa  # noqa: E402

STEP = qa.Q17_DIR / "4"
EVENTS = Path("/Volumes/Data4T/earnings_policy/drop_rebound/events.parquet")
SPLIT_YEAR = 2024
DROP_FEATS = ["drop_atr", "drop_pct", "gap_pct", "intraday_pct", "earnings", "mkt_ret", "mkt_rel", "market_wide",
              "vol_ratio", "atr_pct", "ret5", "ret20", "dist_sma50", "dist_sma200", "rsi", "close_loc", "drops_20d"]
SECTOR_FEATS = ["fund_ret", "own_vs_fund"]
NEWS_FEATS = ["news_n", "news_sent", "neg_earn", "pos_earn", "guide_cut", "downgrade", "neg_high", "bad_news"]
CAL_FEATS = ["to_report", "pt_cut20", "net20"]
THEMES = ("ai_capex", "ai_policy", "memory_cycle", "export_controls", "tariffs_trade", "rates_inflation", "elections_policy",
          "geopolitics_war", "energy_oil", "antitrust", "supply_chain", "consumer_demand", "health_policy", "crypto")
THEME_FEATS = [f"th_{t}" for t in THEMES]


def _themes(core: list[str]) -> pd.DataFrame:
    """v3 theme counts per stock and session, the student's relevant rows placed as news_events places them."""
    import news_events as ne
    col = qa._mongo()["news_articles"]
    rows = list(col.aggregate([
        {"$match": {"symbol": {"$in": core}, "student_gd1.relevant": True, "student_gd1.themes.0": {"$exists": True}}},
        {"$project": {"_id": 0, "symbol": 1, "date": 1, "title": 1, "themes": "$student_gd1.themes"}}], allowDiskUse=True, batchSize=5000))
    df = pd.DataFrame(rows)
    parts = []
    for s, g in df.groupby("symbol"):
        idx = ne._sessions(s).index
        ds = g["date"].astype(str)                       # news_events._days_for's placement, kept per row
        stamp = pd.to_datetime(ds.where(ds.str.len() == 14), format="%Y%m%d%H%M%S", errors="coerce", utc=True).dt.tz_convert("America/New_York")
        day = pd.to_datetime(ds.str[:8], format="%Y%m%d", errors="coerce")
        pre = stamp.notna() & ((stamp.dt.hour * 60 + stamp.dt.minute) < 9 * 60 + 30)
        local = stamp.dt.tz_localize(None).dt.normalize().where(stamp.notna(), day)
        pos = np.searchsorted(idx.values, local.values.astype("datetime64[ns]"))
        same = (pos < len(idx)) & (idx.values[np.minimum(pos, len(idx) - 1)] == local.values.astype("datetime64[ns]"))
        pos = np.where(same & ~pre.values, pos + 1, pos)
        ok = (pos < len(idx)) & local.notna().values
        g = g[ok].assign(session=idx[pos[ok]])
        ex = g[["session", "themes"]].explode("themes")
        ex = ex[ex.themes.isin(THEMES)]
        t = pd.crosstab(ex.session, ex.themes).reindex(columns=list(THEMES), fill_value=0)
        t.columns = THEME_FEATS
        t["symbol"] = s
        parts.append(t.reset_index())
    return pd.concat(parts, ignore_index=True)


def build() -> Path:
    import barrier_model as bm
    import news_events as ne
    import q17_analysts as an
    STEP.mkdir(parents=True, exist_ok=True)
    core = qa._core()
    ev = pd.read_parquet(EVENTS)
    ev = ev[(ev.date >= "2018-01-01") & ev.symbol.isin(core)].copy()
    ev["date"] = pd.to_datetime(ev["date"])
    # targets
    fw = ne._forward()[["symbol", "session", "n20"]].rename(columns={"session": "date"})
    ev = ev.merge(fw, on=["symbol", "date"], how="left")
    lab = []
    for s, g in ev.groupby("symbol"):
        d = bm.da._daily(s)
        b = bm.barrier_labels(d, *bm.SETS["5"])
        lab.append(pd.DataFrame({"symbol": s, "date": d.index, "b5": b}))
    ev = ev.merge(pd.concat(lab), on=["symbol", "date"], how="left")
    ev["lower5"] = (ev.b5 == 0).astype(float).where(ev.b5.notna())
    ev["bad20"] = (ev.n20 < -3).astype(float).where(ev.n20.notna())
    # sector
    sec = json.loads((qa.STEP2B / "core_sectors.json").read_text())
    sf = []
    for s, g in ev.groupby("symbol"):
        idx = ne._sessions(s).index
        b, _ = qa._bench(s, idx, sec)
        sf.append(pd.DataFrame({"symbol": s, "date": idx, "fund_ret": (b.Close.pct_change() * 100).values}))
    ev = ev.merge(pd.concat(sf), on=["symbol", "date"], how="left")
    ev["own_vs_fund"] = ev.drop_pct - ev.fund_ret
    # news on the drop session and the next (production labels)
    nd = pd.read_parquet(qa.STEP / "v2" / "news_days.parquet").rename(columns={"session": "date"})
    nd["date"] = pd.to_datetime(nd["date"])
    nxt = []
    for s, g in ev.groupby("symbol"):
        idx = ne._sessions(s).index
        pos = idx.get_indexer(g.date)
        nxt.append(pd.DataFrame({"symbol": s, "date": g.date.values, "date1": idx[np.minimum(pos + 1, len(idx) - 1)]}))
    ev = ev.merge(pd.concat(nxt), on=["symbol", "date"], how="left")
    cols = ["n", "sent", "neg_earn", "pos_earn", "guide_cut", "downgrade", "neg_high"]
    a0 = ev[["symbol", "date"]].merge(nd[["symbol", "date", *cols]], on=["symbol", "date"], how="left").fillna(0)
    a1 = ev[["symbol", "date1"]].merge(nd[["symbol", "date", *cols]].rename(columns={"date": "date1"}), on=["symbol", "date1"], how="left").fillna(0)
    ev["news_n"] = a0.n.values + a1.n.values
    ev["news_sent"] = np.where(ev.news_n > 0, (a0.sent.values * a0.n.values + a1.sent.values * a1.n.values) / ev.news_n.replace(0, np.nan), 0.0)
    for c in ("neg_earn", "pos_earn", "guide_cut", "downgrade", "neg_high"):
        ev[c] = a0[c].values + a1[c].values
    ev["bad_news"] = ((ev.neg_earn + ev.guide_cut + ev.downgrade) > 0).astype(int)
    # calendar and analysts
    bd = pd.read_parquet(qa.STEP / "v2" / "barrier_dataset.parquet")[["symbol", "date", "to_report"]]
    ev = ev.merge(bd, on=["symbol", "date"], how="left")
    acts = pd.read_parquet(an.STEP / "analyst_actions.parquet")
    sfa = an._sessions_frame(acts)
    roll = []
    for s, g in sfa.groupby("symbol"):
        g = g.sort_values("session")
        roll.append(pd.DataFrame({"symbol": s, "date": g.session.values,
                                  "pt_cut20": g.e_pt_cut.astype(int).rolling(20, min_periods=1).sum().values,
                                  "net20": g.net20.values}))
    ev = ev.merge(pd.concat(roll), on=["symbol", "date"], how="left")
    ev[["pt_cut20", "net20"]] = ev[["pt_cut20", "net20"]].fillna(0)
    # v3 themes
    th = _themes(core).rename(columns={"session": "date"})
    th["date"] = pd.to_datetime(th["date"])
    t0 = ev[["symbol", "date"]].merge(th, on=["symbol", "date"], how="left").fillna(0)
    t1 = ev[["symbol", "date1"]].merge(th.rename(columns={"date": "date1"}), on=["symbol", "date1"], how="left").fillna(0)
    for c in THEME_FEATS:
        ev[c] = t0[c].values + t1[c].values
    keep = ["symbol", "date", "year", "lower5", "bad20", *DROP_FEATS, *SECTOR_FEATS, *NEWS_FEATS, *CAL_FEATS, *THEME_FEATS]
    ev = ev[keep].reset_index(drop=True)
    out = STEP / "drops.parquet"
    ev.to_parquet(out)
    meta = {"split_year": SPLIT_YEAR, "targets": ["lower5", "bad20"],
            "models": {"logit": DROP_FEATS + SECTOR_FEATS + NEWS_FEATS + CAL_FEATS,
                       "logit_themes": DROP_FEATS + SECTOR_FEATS + NEWS_FEATS + CAL_FEATS + THEME_FEATS,
                       "tree_themes": DROP_FEATS + SECTOR_FEATS + NEWS_FEATS + CAL_FEATS + THEME_FEATS},
            "primary": "logit"}
    (STEP / "drops_meta.json").write_text(json.dumps(meta, indent=1))
    tr, te = ev[ev.year < SPLIT_YEAR], ev[ev.year >= SPLIT_YEAR]
    print(f"{out}: {len(ev):,} drops ({len(tr):,} train / {len(te):,} test, {te.date.nunique()} test dates); "
          f"lower5 {ev.lower5.mean():.3f}, bad20 {ev.bad20.mean():.3f}; any theme on {int((ev[THEME_FEATS].sum(axis=1) > 0).sum())} drops")
    return out


def _brier(y, p):
    return (p - y) ** 2


def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ece(y, p, bins=10):
    q = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(abs(y[q == b].mean() - p[q == b].mean()) * (q == b).mean() for b in range(bins) if (q == b).any()))


def score(n_boot: int = 2000) -> dict:
    ev = pd.read_parquet(STEP / "drops.parquet")
    pr = pd.read_parquet(STEP / "colab" / "preds.parquet")
    meta = json.loads((STEP / "drops_meta.json").read_text())
    tr, te = ev[ev.year < SPLIT_YEAR], ev[ev.year >= SPLIT_YEAR].reset_index(drop=True)
    pr = te[["symbol", "date"]].merge(pr, on=["symbol", "date"], how="left", validate="one_to_one")
    rng = np.random.default_rng(7)
    rep = {"train": int(len(tr)), "test": int(len(te)), "results": {}}
    for y in meta["targets"]:
        m = te[y].notna().values
        yy = te.loc[m, y].values
        base = {}
        g1 = tr.groupby("bad_news")[y].mean()
        base["news_split"] = te.loc[m, "bad_news"].map(g1).fillna(tr[y].mean()).values
        g2 = tr.groupby(["earnings", "market_wide"])[y].mean()
        base["cause_table"] = pd.Series(list(zip(te.loc[m, "earnings"], te.loc[m, "market_wide"]))).map(g2).fillna(tr[y].mean()).values
        base["pooled"] = np.full(m.sum(), tr[y].mean())
        preds = {k: pr.loc[m, f"p_{y}__{k}"].values for k in meta["models"]}
        dates = te.loc[m, "date"].values
        ud, inv = np.unique(dates, return_inverse=True)
        cnt = np.bincount(inv)
        res = {"events": int(m.sum()), "rate": round(float(yy.mean()), 3), "models": {}}
        for k, p in {**preds, **{f"base:{b}": v for b, v in base.items()}}.items():
            res["models"][k] = {"brier": round(float(_brier(yy, p).mean()), 4), "logloss": round(float(_logloss(yy, p).mean()), 4),
                                "ece": round(_ece(yy, p), 4)}
        for k, p in preds.items():
            for b in ("news_split", "cause_table"):
                for metric, fn in (("brier", _brier), ("logloss", _logloss)):
                    per = np.bincount(inv, weights=fn(yy, p) - fn(yy, base[b]))
                    boots = [per[q].sum() / cnt[q].sum() for q in (rng.integers(0, len(ud), len(ud)) for _ in range(n_boot))]
                    res["models"][k][f"{metric}_vs_{b}"] = {"diff": round(float(per.sum() / cnt.sum()), 5),
                                                             "ci95": [round(float(v), 5) for v in np.percentile(boots, [2.5, 97.5])]}
            r = res["models"][k]
            r["passes"] = bool(all(r[f"{mt}_vs_{b}"]["ci95"][1] < 0 for mt in ("brier", "logloss") for b in ("news_split", "cause_table"))
                               and r["ece"] < 0.03)
        rep["results"][y] = res
    rep["primary_passes"] = {y: rep["results"][y]["models"][meta["primary"]]["passes"] for y in meta["targets"]}
    (STEP / "score.json").write_text(json.dumps(rep, indent=1))
    for y, res in rep["results"].items():
        print(f"\n{y}: {res['events']} test drops, rate {res['rate']}")
        for k, r in res["models"].items():
            extra = "" if k.startswith("base:") else "  vs news " + str(r["brier_vs_news_split"]["ci95"]) + " vs cause " + str(r["brier_vs_cause_table"]["ci95"]) + f"  pass {r['passes']}"
            print(f"  {k:18s} brier {r['brier']}  logloss {r['logloss']}  ece {r['ece']}{extra}")
    print("\nprimary passes:", rep["primary_passes"])
    return rep


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
