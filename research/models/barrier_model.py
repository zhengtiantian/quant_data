"""Which comes first: a rebound to a sell level, a fall to a stop, or the time limit (PROJECT_PLAN Q.16.7,
2026-11-02). Aimed at getting out of INTC first.

The direction models (daily_advice.py) had no skill at "up or down". Practitioners label trades the way
they are managed instead (Lopez de Prado's triple barrier): from the next open, with ATR at the decision,

    upper  = open + UP x ATR      a rebound worth selling into
    lower  = open - DOWN x ATR    a further fall worth getting out before
    time   = H sessions           neither by then

whichever the daily highs and lows touch first (both in one session counts as the lower: the cautious
reading). For a holding under water the three map to the three actions: sell into the rebound with a
limit at the upper barrier, cut at the lower, or keep holding.

Features: daily_advice's ~35 plus the article-level news of news_events.py over the last 5 and 20
sessions (guidance cuts, bad / good earnings news, high-impact bad news, downgrades / upgrades), which the
event study found to move the next 20-60 sessions (a guidance cut: -1.6% vs all days over 20 sessions,
t -4.4; a 5% fall without bad news: +1.1%, t 3.0).

  train    per barrier set, walk-forward multiclass LightGBM (2019-2026), judged on log loss / Brier
           against the yearly pooled base rate and on deep-drawdown days (30%+ under the 1-year high), and the
           policy "cut when P(lower first) is high" against always holding with the barriers
  predict  today's probabilities for the given symbols -> barrier_latest.json (merged into latest.json)

Usage: .venv311/bin/python research/models/barrier_model.py train|predict [SYMBOLS...]
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_advice as da  # noqa: E402

OUT = da.OUT
SETS = {"5": (1.0, 1.5, 5), "20": (1.5, 2.0, 20), "60": (2.5, 3.0, 60)}      # (UP, DOWN, H) in ATR / sessions
NEWS_FEATS = ["guide_cut_20", "neg_earn_20", "neg_high_5", "neg_high_20", "downgrade_20", "upgrade_20", "pos_earn_20",
              "news_5", "sent_5", "bad_news_on_drop"]
NEWS_LABELS = {"guide_cut_20": ("guidance cuts, 20 sessions", "近 20 天指引下调消息"), "neg_earn_20": ("bad earnings news, 20 sessions", "近 20 天财报负面消息"),
               "neg_high_5": ("high-impact bad news, 5 sessions", "近 5 天高影响负面消息"), "neg_high_20": ("high-impact bad news, 20 sessions", "近 20 天高影响负面消息"),
               "downgrade_20": ("downgrades, 20 sessions", "近 20 天降级"), "upgrade_20": ("upgrades, 20 sessions", "近 20 天上调评级"),
               "pos_earn_20": ("good earnings news, 20 sessions", "近 20 天财报正面消息"), "news_5": ("news volume, 5 sessions", "近 5 天新闻量"),
               "sent_5": ("article sentiment, 5 sessions", "近 5 天新闻情绪"), "bad_news_on_drop": ("the last large drop came with bad news", "上次大跌伴随负面消息")}
import calendar_events as ce  # noqa: E402
# Scheduled events ahead (calendar_events.py, 2026-11-02): with them the 5-day barrier model's Brier
# score improved 0.6278 -> 0.6269 out of sample; the direction model did not improve.
FEATS = da.FEATS + NEWS_FEATS + ce.CAL_FEATS
CLASSES = ("lower", "none", "upper")


def barrier_labels(d: pd.DataFrame, up: float, down: float, h: int) -> np.ndarray:
    """0 = lower first, 1 = neither within h sessions, 2 = upper first; NaN where the window runs past the data."""
    o, hi, lo, c = d.Open.values, d.High.values, d.Low.values, d.Close.values
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[hi - lo, np.abs(hi - pc), np.abs(lo - pc)], axis=1)
    atr = pd.Series(tr).rolling(14).mean().values
    n = len(d)
    out = np.full(n, np.nan)
    for t in range(n - h - 1):
        if not np.isfinite(atr[t]):
            continue
        entry = o[t + 1]
        u, l = entry + up * atr[t], entry - down * atr[t]
        res = 1
        for j in range(t + 1, t + 1 + h):
            if lo[j] <= l:
                res = 0
                break
            if hi[j] >= u:
                res = 2
                break
        out[t] = res
    return out


def news_features(sym: str, index: pd.DatetimeIndex, nd: pd.DataFrame, drop_flag: pd.Series) -> pd.DataFrame:
    g = nd[nd.symbol == sym].set_index("session").reindex(index).fillna(0)
    f = pd.DataFrame(index=index)
    roll = lambda c, w: g[c].rolling(w, min_periods=1).sum()
    f["guide_cut_20"] = roll("guide_cut", 20)
    f["neg_earn_20"] = roll("neg_earn", 20)
    f["neg_high_5"] = roll("neg_high", 5)
    f["neg_high_20"] = roll("neg_high", 20)
    f["downgrade_20"] = roll("downgrade", 20)
    f["upgrade_20"] = roll("upgrade", 20)
    f["pos_earn_20"] = roll("pos_earn", 20)
    f["news_5"] = np.log1p(roll("n", 5))
    s = (g["sent"] * g["n"]).rolling(5, min_periods=1).sum() / g["n"].rolling(5, min_periods=1).sum().replace(0, np.nan)
    f["sent_5"] = s
    bad_day = ((g["guide_cut"] + g["neg_earn"] + g["downgrade"] + g["neg_high"]) > 0)
    # was the most recent >= 2 x ATR drop a day with bad news?
    f["bad_news_on_drop"] = (bad_day & drop_flag).astype(float).where(drop_flag).ffill().fillna(0)
    return f


def build() -> pd.DataFrame:
    base = pd.read_parquet(OUT / "dataset.parquet")
    nd = pd.read_parquet(OUT / "news_days.parquet")
    cal = ce.calendar(ce.sessions())
    parts = []
    for sym, g in base.groupby("symbol"):
        d = da._daily(sym)
        g = g.set_index("date").reindex(d.index).dropna(subset=["r1"])
        if g.empty:
            continue
        idx = g.index
        dd = d.loc[idx]
        drop_flag = (g["drop_atr"] >= 2)
        nf = news_features(sym, idx, nd, drop_flag)
        x = g.join(nf).join(cal.reindex(idx))
        x["report_in_5"] = x["to_report"].between(0, 7).astype(float)
        for k, (u, l, h) in SETS.items():
            x[f"b{k}"] = barrier_labels(d, u, l, h)[d.index.get_indexer(idx)]
        x["symbol"], x["date"] = sym, idx
        parts.append(x.reset_index(drop=True))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(OUT / "barrier_dataset.parquet")
    print(f"barrier dataset: {len(df):,} stock-days;", {k: df[f'b{k}'].value_counts(normalize=True).round(3).to_dict() for k in SETS})
    return df


def _lgb():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=400,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, random_state=da.SEED, verbose=-1)


def train() -> dict:
    from sklearn.metrics import log_loss
    df = build()
    df["year"] = df.date.dt.year
    rep = {"sets": {k: {"up_atr": u, "down_atr": l, "sessions": h} for k, (u, l, h) in SETS.items()}, "results": {}}
    for k, (u, l, h) in SETS.items():
        y = f"b{k}"
        preds = []
        for yr in range(da.FIRST_TEST_YEAR, int(df.year.max()) + 1):
            emb = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=int(h * 1.5))
            tr = df[(df.date < emb) & df[y].notna()]
            te = df[(df.year == yr) & df[y].notna()].copy()
            if te.empty:
                continue
            m = _lgb().fit(tr[FEATS], tr[y].astype(int))
            p = m.predict_proba(te[FEATS])
            for i, c in enumerate(CLASSES):
                te[f"p_{c}"] = p[:, i]
                te[f"base_{c}"] = (tr[y] == i).mean()
            preds.append(te)
        pr = pd.concat(preds, ignore_index=True)
        pr.to_parquet(OUT / f"barrier_wf_{k}.parquet")
        yy = pr[y].astype(int).values
        P = pr[[f"p_{c}" for c in CLASSES]].values
        B = pr[[f"base_{c}" for c in CLASSES]].values
        brier = lambda Q: float(np.mean(np.sum((Q - np.eye(3)[yy]) ** 2, axis=1)))
        res = {"days": int(len(pr)), "rates": {c: round(float((yy == i).mean() * 100), 1) for i, c in enumerate(CLASSES)},
               "logloss_model": round(log_loss(yy, P, labels=[0, 1, 2]), 4), "logloss_base": round(log_loss(yy, B, labels=[0, 1, 2]), 4),
               "brier_model": round(brier(P), 4), "brier_base": round(brier(B), 4)}
        deep = pr.dd_252 <= -30
        yd = yy[deep.values]
        res["deep_drawdown"] = {"days": int(deep.sum()), "rates": {c: round(float((yd == i).mean() * 100), 1) for i, c in enumerate(CLASSES)},
                                "brier_model": round(float(np.mean(np.sum((P[deep.values] - np.eye(3)[yd]) ** 2, axis=1))), 4),
                                "brier_base": round(float(np.mean(np.sum((B[deep.values] - np.eye(3)[yd]) ** 2, axis=1))), 4)}
        # does P(lower first) sort outcomes? quintiles of p_lower, actual lower-first rate
        pr["qq"] = pd.qcut(pr.p_lower, 5, labels=False, duplicates="drop")
        res["p_lower_quintiles"] = [{"pred": round(g.p_lower.mean() * 100, 1), "actual": round(float((g[y] == 0).mean() * 100), 1), "n": int(len(g))}
                                    for _, g in pr.groupby("qq")]
        # the getting-out policy: hold with both barriers (sell at the upper, cut at the lower, else exit at the
        # close after h) against cutting now when P(lower first) is in its top fifth (2023-2026 held out)
        dd = []
        for sym, g in pr.groupby("symbol"):
            d = da._daily(sym)
            pos = d.index.get_indexer(g.date)
            o1 = d.Open.values[np.minimum(pos + 1, len(d) - 1)]
            ch = d.Close.values[np.minimum(pos + h, len(d) - 1)]
            atr = (g.atr_pct.values / 100) * d.Close.values[pos]
            r_up, r_dn = (o1 + u * atr) / o1 - 1, (o1 - l * atr) / o1 - 1
            out = np.where(g[y].values == 2, r_up, np.where(g[y].values == 0, r_dn, ch / o1 - 1)) * 100
            dd.append(pd.DataFrame({"date": g.date.values, "barrier_ret": out, "p_lower": g.p_lower.values, "deep": (g.dd_252 <= -30).values}))
        pol = pd.concat(dd)
        cut = pol.p_lower >= pol.p_lower.quantile(0.8)
        late = pol.date >= pd.Timestamp("2023-01-01")
        res["policy"] = {
            "hold_with_barriers_mean_pct": round(float(pol.barrier_ret.mean()), 3),
            "top_fifth_p_lower_mean_pct": round(float(pol[cut].barrier_ret.mean()), 3),
            "rest_mean_pct": round(float(pol[~cut].barrier_ret.mean()), 3),
            "late_top_fifth_mean_pct": round(float(pol[cut & late].barrier_ret.mean()), 3),
            "late_rest_mean_pct": round(float(pol[~cut & late].barrier_ret.mean()), 3),
            "deep_top_fifth_mean_pct": round(float(pol[cut & pol.deep].barrier_ret.mean()), 3),
            "deep_rest_mean_pct": round(float(pol[~cut & pol.deep].barrier_ret.mean()), 3)}
        rep["results"][k] = res
        print(k, json.dumps(res), flush=True)
    import joblib
    final = {}
    for k, (u, l, h) in SETS.items():
        tr = df[df[f"b{k}"].notna()]
        final[k] = _lgb().fit(tr[FEATS], tr[f"b{k}"].astype(int))
    joblib.dump({"models": final, "features": FEATS, "sets": SETS, "trained_at": datetime.now(timezone.utc).isoformat()}, OUT / "barrier_models.joblib")
    (OUT / "barrier_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def _situation(deep: bool, badnews: bool) -> str:
    return ("deep" if deep else "normal") + ("_badnews" if badnews else "")


def situations() -> dict:
    """Per barrier set and situation (30%+ under the year's high or not; a recent bad-news drop or
    guidance cut or not), from the walk-forward days: which barrier came first, and holding to the time
    limit against managing with both barriers (mean and worst 5%). Also the 5-day model's P(lower)
    cut points, to say where today's reading sits."""
    out = {"sets": {}, "p_lower_cuts_5": None}
    for k, (u, l, h) in SETS.items():
        pr = pd.read_parquet(OUT / f"barrier_wf_{k}.parquet")
        if k == "5":
            out["p_lower_cuts_5"] = [round(float(pr.p_lower.quantile(q)), 4) for q in (0.2, 0.4, 0.6, 0.8)]
            # the raw probabilities are too spread (the calmest fifth said 13% and saw 21%): mapped
            # through isotonic fits to what happened, on the walk-forward predictions
            from sklearn.isotonic import IsotonicRegression
            grid = np.linspace(0, 1, 101)
            out["calibration_5"] = {c: [round(float(v), 4) for v in IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                                        .fit(pr[f"p_{c}"], (pr[f"b{k}"] == i).astype(float)).predict(grid)]
                                    for i, c in ((0, "lower"), (2, "upper"))}
        rows = []
        for sym, g in pr.groupby("symbol"):
            d = da._daily(sym)
            pos = d.index.get_indexer(g.date)
            o1 = d.Open.values[np.minimum(pos + 1, len(d) - 1)]
            ch = d.Close.values[np.minimum(pos + h, len(d) - 1)]
            atr = (g.atr_pct.values / 100) * d.Close.values[pos]
            lab = g[f"b{k}"].values
            bar = np.where(lab == 2, u * atr / o1, np.where(lab == 0, -l * atr / o1, ch / o1 - 1)) * 100
            bad = ((g.guide_cut_20 > 0) | ((g.bad_news_on_drop > 0) & (g.since_bigdrop <= 20))).values
            rows.append(pd.DataFrame({"hold": (ch / o1 - 1) * 100, "barrier": bar, "lab": lab,
                                      "sit": [_situation(a, b) for a, b in zip((g.dd_252 <= -30).values, bad)]}))
        x = pd.concat(rows)
        out["sets"][k] = {sit: {"n": int(len(g)), "upper_first": round(float((g.lab == 2).mean() * 100), 1),
                                "lower_first": round(float((g.lab == 0).mean() * 100), 1),
                                "hold_mean": round(float(g.hold.mean()), 2), "hold_worst5": round(float(np.percentile(g.hold, 5)), 1),
                                "barrier_mean": round(float(g.barrier.mean()), 2), "barrier_worst5": round(float(np.percentile(g.barrier, 5)), 1)}
                          for sit, g in x.groupby("sit")}
    rep = json.loads((OUT / "barrier_report.json").read_text())
    rep["situations"] = out
    (OUT / "barrier_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(out, indent=1))
    return out


def live(sym: str, d: pd.DataFrame, f: pd.DataFrame) -> dict:
    """Today's barriers for one symbol: the 5-day model's probabilities, and what similar situations did
    over 5 / 20 / 60 sessions held or managed with the barriers. d: daily bars, f: daily_advice features."""
    import joblib
    import news_events as ne
    bundle = joblib.load(OUT / "barrier_models.joblib")
    rep = json.loads((OUT / "barrier_report.json").read_text())
    idx = d.index[-80:]
    nd = ne.symbol_news_days(sym, idx)
    nf = news_features(sym, idx, nd, (f.loc[idx, "drop_atr"] >= 2))
    cal = ce.calendar(d.index)
    x = f.loc[[idx[-1]]].join(nf.loc[[idx[-1]]]).join(cal.loc[[idx[-1]]])
    x["report_in_5"] = float(0 <= f["to_report"].iloc[-1] <= 7)
    x = x[FEATS]
    ahead = cal.iloc[-1]
    close = float(d.Close.iloc[-1])
    atr = float(f["atr_pct"].iloc[-1]) / 100 * close
    deep = bool(f["dd_252"].iloc[-1] <= -30)
    bad = bool(nf["guide_cut_20"].iloc[-1] > 0 or (nf["bad_news_on_drop"].iloc[-1] > 0 and f["since_bigdrop"].iloc[-1] <= 20))
    sit = _situation(deep, bad)
    raw = bundle["models"]["5"].predict_proba(x)[0]
    cuts = rep["situations"]["p_lower_cuts_5"]
    cal = rep["situations"].get("calibration_5")
    if cal:
        lo = cal["lower"][int(round(raw[0] * 100))]
        hi = cal["upper"][int(round(raw[2] * 100))]
        p = [lo, max(0.0, 1 - lo - hi), hi]
    else:
        p = list(raw)
    out = {"close": round(close, 2), "atr": round(atr, 2), "situation": sit, "deep": deep, "bad_news": bad,
           "calendar": {"to_fomc": int(ahead.to_fomc), "to_opex": int(ahead.to_opex), "to_nfp": int(ahead.to_nfp),
                        "pre_holiday": bool(ahead.pre_holiday), "to_report_days": None if pd.isna(f["to_report"].iloc[-1]) else int(f["to_report"].iloc[-1])},
           "news_20": {c: float(nf[c].iloc[-1]) for c in ("guide_cut_20", "neg_earn_20", "downgrade_20", "upgrade_20")},
           "model_5": {"p_lower": round(float(p[0]) * 100, 1), "p_none": round(float(p[1]) * 100, 1), "p_upper": round(float(p[2]) * 100, 1),
                       "risk_fifth": int(sum(raw[0] > c for c in cuts)) + 1, "calibrated": bool(cal)},   # 1 = calmest fifth, 5 = riskiest
           "sets": {}}
    for k, (u, l, h) in SETS.items():
        st = rep["situations"]["sets"][k].get(sit) or rep["situations"]["sets"][k].get("normal")
        out["sets"][k] = {"up_atr": u, "down_atr": l, "sessions": h, "sell_at": round(close + u * atr, 2), "stop_at": round(close - l * atr, 2), **st}
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"build": build, "train": train, "situations": situations}[cmd]()
