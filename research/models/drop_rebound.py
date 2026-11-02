"""After a large drop: does the price come back to a sell limit, and which limit to place (PROJECT_PLAN
Q.14 / Q.13, 2026-11-02).

quant_ai's daily action trims a holding over its cap with two limits at +1.5 and +2 x ATR, quoting
"reached in 61% / 50% of >= 2 x ATR drops" for every stock and every drop. That rate is one pooled
average, and its study was not kept. This script rebuilds it from the 102-stock daily bars and asks
what the action needs instead of a constant:

  events   every close-to-close drop of >= 2 x ATR14 (2016-2026), with what was known at that close:
           size of the drop, gap vs intraday part, an earnings report that night/morning, the
           market's move, volume, trend, volatility
  a1       reach rates split by cause: earnings drops vs others, market-wide vs own (test A1)
  fill     walk-forward (train on earlier years, test the next) LightGBM: P(high reaches +k x ATR
           within 15 sessions), k = 0.5..3, against the pooled rate the action uses today
  value    what each limit is worth for a trim that must happen: proceeds = the limit if reached,
           else the close 15 sessions later; per k, against selling at the next open; a model picks
           k per event, scored walk-forward against the fixed +1.5 / +2 x ATR pair
  overnight  overnight (close -> open) vs intraday (open -> close) returns, all days and after drops (A2)
  preearn  return of the session before an after-the-close report vs other sessions (A3)
  vol      volatility-managed exposure (Moreira-Muir) per stock vs constant exposure (A4)

Usage: .venv311/bin/python research/models/drop_rebound.py all     (writes drop_rebound/*.parquet + report.json)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EP_DIR = Path(os.getenv("EP_DIR", "/Volumes/Data4T/earnings_policy"))
OUT = EP_DIR / "drop_rebound"
MARKET = "SPY"
DROP_ATR = 2.0             # the event: a close-to-close drop of at least this many ATR14
HORIZON = 15               # sessions a limit stays good for (daily_action.VALID_SESSIONS)
LEVELS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
FIRST_TEST_YEAR = 2019
SEED = 7


# ------------------------------------------------------------------ data

def symbols():
    return sorted(p.name.split(".")[0] for p in EP_DIR.glob("*.daily.parquet") if not p.name.startswith(MARKET + "."))


def daily(sym):
    d = pd.read_parquet(EP_DIR / f"{sym}.daily.parquet")[["Open", "High", "Low", "Close", "Volume"]]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d[~d.index.duplicated()].sort_index().dropna()


def report_days(sym, index):
    """Sessions whose close-to-close move contains a report: the session after an after-the-close
    report, the session of a before-the-open one."""
    try:
        rows = json.loads((EP_DIR / f"{sym}.earnings.json").read_text())
    except FileNotFoundError:
        return set()
    out = set()
    for r in rows:
        if r.get("eps") is None:                        # not reported yet
            continue
        t = pd.Timestamp(r["time"])
        day = t.tz_localize(None).normalize() if t.tzinfo is None else t.tz_convert("America/New_York").tz_localize(None).normalize()
        after_close = t.hour >= 16 if t.tzinfo else False
        pos = index.searchsorted(day)
        if after_close:
            pos += 1 if pos < len(index) and index[pos] == day else 0
        if pos < len(index):
            out.add(index[pos])
    return out


def atr14(d):
    prev = d["Close"].shift()
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev).abs(), (d["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def rsi14(c):
    diff = c.diff()
    up, down = diff.clip(lower=0).rolling(14).mean(), (-diff.clip(upper=0)).rolling(14).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


# ------------------------------------------------------------------ events

def build_events():
    mk = daily(MARKET)
    mk_ret = mk["Close"].pct_change() * 100
    rows = []
    for sym in symbols():
        d = daily(sym)
        if len(d) < 300:
            continue
        a = atr14(d)
        prev_c, prev_a = d["Close"].shift(), a.shift()
        drop = d["Close"] - prev_c
        drop_atr = -drop / prev_a
        reports = report_days(sym, d.index)
        hi, lo, cl, op = d["High"].values, d["Low"].values, d["Close"].values, d["Open"].values
        sma50, sma200 = d["Close"].rolling(50).mean(), d["Close"].rolling(200).mean()
        vol20 = d["Volume"].rolling(20).mean().shift()
        rsi = rsi14(d["Close"])
        for i in np.where((drop_atr >= DROP_ATR).values)[0]:
            if i < 200 or i + HORIZON >= len(d):
                continue
            t = d.index[i]
            c0, at = cl[i], a.iloc[i]                   # the decision is made at this close, with this ATR
            fut_hi, fut_lo, fut_cl = hi[i + 1:i + 1 + HORIZON], lo[i + 1:i + 1 + HORIZON], cl[i + 1:i + 1 + HORIZON]
            r = {"symbol": sym, "date": t, "year": t.year, "close": c0, "atr": at,
                 "drop_atr": drop_atr.iloc[i], "drop_pct": (c0 / prev_c.iloc[i] - 1) * 100,
                 "gap_pct": (op[i] / prev_c.iloc[i] - 1) * 100, "intraday_pct": (c0 / op[i] - 1) * 100,
                 "earnings": int(t in reports),
                 "mkt_ret": mk_ret.get(t, np.nan),
                 "vol_ratio": d["Volume"].iloc[i] / vol20.iloc[i] if vol20.iloc[i] else np.nan,
                 "atr_pct": at / c0 * 100,
                 "ret5": (prev_c.iloc[i] / cl[i - 5] - 1) * 100, "ret20": (prev_c.iloc[i] / cl[i - 20] - 1) * 100,
                 "dist_sma50": (c0 / sma50.iloc[i] - 1) * 100, "dist_sma200": (c0 / sma200.iloc[i] - 1) * 100,
                 "rsi": rsi.iloc[i], "close_loc": (c0 - lo[i]) / (hi[i] - lo[i]) if hi[i] > lo[i] else 0.5,
                 "drops_20d": int((drop_atr.iloc[i - 20:i] >= DROP_ATR).sum()),
                 "next_open": op[i + 1], "close_h": fut_cl[-1],
                 "further_1atr_3d": int(fut_lo[:3].min() <= c0 - at),
                 "max_up_atr": (fut_hi.max() - c0) / at}
            for k in LEVELS:
                hit = np.where(fut_hi >= c0 + k * at)[0]
                r[f"hit_{k}"] = int(len(hit) > 0)
                r[f"days_{k}"] = int(hit[0]) + 1 if len(hit) else np.nan
            rows.append(r)
    ev = pd.DataFrame(rows)
    ev["mkt_rel"] = ev["drop_pct"] - ev["mkt_ret"]          # the stock's own part of the drop
    ev["market_wide"] = (ev["mkt_ret"] <= -1.5).astype(int)
    OUT.mkdir(exist_ok=True)
    ev.to_parquet(OUT / "events.parquet")
    return ev


# ------------------------------------------------------------------ helpers

def rate_ci(x, n_boot=2000, seed=SEED):
    """Mean of a 0/1 series with a 95% bootstrap interval resampled by date (drops on one day are one draw)."""
    x = x.dropna()
    if not len(x):
        return {"n": 0}
    days = x.groupby(level=0).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), (n_boot, len(days)))
    s, c = days["sum"].values[idx].sum(1), days["count"].values[idx].sum(1)
    lo, hi = np.percentile(s / c * 100, [2.5, 97.5])
    return {"n": int(len(x)), "pct": round(x.mean() * 100, 1), "ci95": [round(lo, 1), round(hi, 1)]}


def mean_ci(x, n_boot=2000, seed=SEED):
    x = x.dropna()
    if not len(x):
        return {"n": 0}
    days = x.groupby(level=0).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), (n_boot, len(days)))
    m = days["sum"].values[idx].sum(1) / days["count"].values[idx].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"n": int(len(x)), "mean": round(x.mean(), 3), "ci95": [round(lo, 3), round(hi, 3)]}


# ------------------------------------------------------------------ A1: reach rates by cause

def a1(ev):
    e = ev.set_index("date")
    groups = {"all": e, "earnings": e[e.earnings == 1], "no_earnings": e[e.earnings == 0],
              "no_earnings_market_wide": e[(e.earnings == 0) & (e.market_wide == 1)],
              "no_earnings_own": e[(e.earnings == 0) & (e.market_wide == 0)],
              "oos_2023_on": e[e.year >= 2023]}
    out = {}
    for name, g in groups.items():
        out[name] = {"events": len(g), "further_1atr_3d": rate_ci(g["further_1atr_3d"]),
                     **{f"reach_{k}": rate_ci(g[f"hit_{k}"]) for k in LEVELS}}
    # The biggest drops (INTC 2026-10-30 was a report and > 5 x ATR)
    big = e[e.drop_atr >= 4]
    out["drop_ge_4atr_earnings"] = {"events": int((big.earnings == 1).sum()),
                                    **{f"reach_{k}": rate_ci(big[big.earnings == 1][f"hit_{k}"]) for k in (1.0, 1.5, 2.0)}}
    out["drop_ge_4atr_other"] = {"events": int((big.earnings == 0).sum()),
                                 **{f"reach_{k}": rate_ci(big[big.earnings == 0][f"hit_{k}"]) for k in (1.0, 1.5, 2.0)}}
    return out


# ------------------------------------------------------------------ fill model (walk-forward)

FEATURES = ["drop_atr", "drop_pct", "gap_pct", "intraday_pct", "earnings", "mkt_ret", "mkt_rel", "market_wide",
            "vol_ratio", "atr_pct", "ret5", "ret20", "dist_sma50", "dist_sma200", "rsi", "close_loc", "drops_20d"]


def _long(ev):
    """One row per (event, level): the fill model takes the level as a feature, decreasing in it."""
    parts = []
    for k in LEVELS:
        p = ev[FEATURES + ["symbol", "date", "year"]].copy()
        p["level"] = k
        p["hit"] = ev[f"hit_{k}"].values
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def fill_walkforward(ev):
    import lightgbm as lgb
    from sklearn.metrics import brier_score_loss, roc_auc_score
    lg = _long(ev)
    feats = FEATURES + ["level"]
    mono = [0] * len(FEATURES) + [-1]
    preds = []
    for y in range(FIRST_TEST_YEAR, int(ev.year.max()) + 1):
        tr, te = lg[lg.year < y], lg[lg.year == y].copy()
        if te.empty:
            continue
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=60,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, monotone_constraints=mono,
                               random_state=SEED, verbose=-1)
        m.fit(tr[feats], tr["hit"])
        te["p_model"] = m.predict_proba(te[feats])[:, 1]
        te["p_pooled"] = te["level"].map(tr.groupby("level")["hit"].mean())        # today's constant, refit each year
        preds.append(te)
    pr = pd.concat(preds, ignore_index=True)
    report = {}
    for k in LEVELS:
        s = pr[pr.level == k]
        report[str(k)] = {"n": len(s), "rate": round(s.hit.mean() * 100, 1),
                          "brier_model": round(brier_score_loss(s.hit, s.p_model), 4),
                          "brier_pooled": round(brier_score_loss(s.hit, s.p_pooled), 4),
                          "auc_model": round(roc_auc_score(s.hit, s.p_model), 3)}
    # calibration of the model at the action's two levels
    cal = {}
    for k in (1.5, 2.0):
        s = pr[pr.level == k].copy()
        s["bin"] = pd.qcut(s.p_model, 5, duplicates="drop")
        cal[str(k)] = [{"pred": round(g.p_model.mean() * 100, 1), "actual": round(g.hit.mean() * 100, 1), "n": len(g)}
                       for _, g in s.groupby("bin", observed=True)]
    pr.to_parquet(OUT / "fill_preds.parquet")
    imp = pd.Series(m.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    return {"by_level": report, "calibration": cal, "importance_last_fold": (imp / imp.sum() * 100).round(1).head(10).to_dict()}


# ------------------------------------------------------------------ value of each limit for a trim

def _proceeds(ev, k):
    """% versus selling at the next open: the limit if reached within HORIZON, else the close then."""
    limit = ev["close"] + k * ev["atr"]
    sold = np.where(ev[f"hit_{k}"] == 1, np.maximum(limit, ev["next_open"]), ev["close_h"])   # a gap above the limit fills at the open
    return (sold / ev["next_open"] - 1) * 100


def value_walkforward(ev):
    import lightgbm as lgb
    for k in LEVELS:
        ev[f"gain_{k}"] = _proceeds(ev, k)
    ev["gain_pair"] = (ev["gain_1.5"] + ev["gain_2.0"]) / 2           # today's action: half at +1.5, half at +2
    out = {"fixed": {}}
    e = ev.set_index("date")
    for col in [f"gain_{k}" for k in LEVELS] + ["gain_pair"]:
        out["fixed"][col] = mean_ci(e[col])
    out["fixed_earnings"] = {c: mean_ci(e[e.earnings == 1][c]) for c in ("gain_1.0", "gain_1.5", "gain_2.0", "gain_pair")}
    out["fixed_no_earnings"] = {c: mean_ci(e[e.earnings == 0][c]) for c in ("gain_1.0", "gain_1.5", "gain_2.0", "gain_pair")}
    # model: predict each level's gain, pick the best; selling at the open (0) when none is predicted above it
    picks = []
    for y in range(FIRST_TEST_YEAR, int(ev.year.max()) + 1):
        tr, te = ev[ev.year < y], ev[ev.year == y].copy()
        if te.empty:
            continue
        pred = {}
        for k in LEVELS:
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=60,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, objective="huber",
                                  random_state=SEED, verbose=-1)
            m.fit(tr[FEATURES], tr[f"gain_{k}"])
            pred[k] = m.predict(te[FEATURES])
        P = np.column_stack([pred[k] for k in LEVELS])
        best = P.argmax(1)
        te["pick"] = np.where(P.max(1) > 0, np.array(LEVELS)[best], 0.0)
        te["gain_model"] = [0.0 if p == 0 else row[f"gain_{p}"] for p, (_, row) in zip(te["pick"], te.iterrows())]
        picks.append(te)
    pk = pd.concat(picks)
    pk.to_parquet(OUT / "value_picks.parquet")
    p = pk.set_index("date")
    out["walkforward"] = {"years": f"{FIRST_TEST_YEAR}-{int(ev.year.max())}", "model": mean_ci(p["gain_model"]),
                          "pair_same_events": mean_ci(p["gain_pair"]), "sell_at_open": 0.0,
                          "diff_model_minus_pair": mean_ci(p["gain_model"] - p["gain_pair"]),
                          "pick_counts": p["pick"].value_counts().sort_index().to_dict(),
                          "model_p10": round(np.percentile(p["gain_model"], 10), 2),
                          "pair_p10": round(np.percentile(p["gain_pair"], 10), 2)}
    return out


# ------------------------------------------------------------------ A2: overnight vs intraday

def overnight():
    rows = []
    for sym in symbols():
        d = daily(sym)
        a = atr14(d)
        drop_atr = -(d["Close"] - d["Close"].shift()) / a.shift()
        rows.append(pd.DataFrame({"symbol": sym,
                                  "overnight": (d["Open"].shift(-1) / d["Close"] - 1) * 100,
                                  "intraday": (d["Close"].shift(-1) / d["Open"].shift(-1) - 1) * 100,
                                  "after_drop": (drop_atr >= DROP_ATR).astype(int)}, index=d.index))
    df = pd.concat(rows).dropna()
    df = df[df.index >= "2016-01-01"]
    out = {}
    for name, g in {"all_days": df, "day_after_2atr_drop": df[df.after_drop == 1]}.items():
        out[name] = {"overnight_mean_pct": mean_ci(g["overnight"]), "intraday_mean_pct": mean_ci(g["intraday"]),
                     "overnight_up_share": round((g["overnight"] > 0).mean() * 100, 1),
                     "intraday_up_share": round((g["intraday"] > 0).mean() * 100, 1)}
    return out


# ------------------------------------------------------------------ A3: the session before an after-the-close report

def preearnings():
    rows = []
    for sym in symbols():
        d = daily(sym)
        try:
            reps = json.loads((EP_DIR / f"{sym}.earnings.json").read_text())
        except FileNotFoundError:
            continue
        ret = (d["Close"] / d["Close"].shift() - 1) * 100
        ret2 = (d["Close"] / d["Close"].shift(2) - 1) * 100
        amc = set()
        for r in reps:
            if r.get("eps") is None:
                continue
            t = pd.Timestamp(r["time"])
            if t.tzinfo is not None and t.hour >= 16:
                amc.add(t.tz_convert("America/New_York").tz_localize(None).normalize())
        flag = pd.Series(d.index.isin(list(amc)).astype(int), index=d.index)
        rows.append(pd.DataFrame({"symbol": sym, "ret": ret, "ret2": ret2, "report_day": flag}))
    df = pd.concat(rows).dropna()
    df = df[df.index >= "2016-01-01"]
    rep, other = df[df.report_day == 1], df[df.report_day == 0]
    return {"report_day_session_ret": mean_ci(rep["ret"]), "other_sessions_ret": mean_ci(other["ret"]),
            "report_day_two_sessions_ret": mean_ci(rep["ret2"]), "other_two_sessions_ret": mean_ci(other["ret2"]),
            "report_day_up_share": round((rep["ret"] > 0).mean() * 100, 1), "other_up_share": round((other["ret"] > 0).mean() * 100, 1),
            "reports": int(len(rep))}


# ------------------------------------------------------------------ A4: volatility-managed exposure

def volmanaged():
    """Per stock: exposure c / sigma^2 (EWMA, known at the previous close), c set so the average exposure
    is 1 (in-sample scale, as Moreira-Muir; it only scales, the Sharpe ratio does not depend on it).
    Also the exposure 1/sigma (risk parity in time), the variant a position cap would use."""
    res = []
    for sym in symbols():
        d = daily(sym)
        r = d["Close"].pct_change().dropna()
        r = r[r.index >= "2016-01-01"]
        if len(r) < 500:
            continue
        sig = r.ewm(halflife=20).std().shift()
        ok = sig.notna() & (sig > 0)
        r, sig = r[ok], sig[ok]
        sharpe = lambda x: x.mean() / x.std() * np.sqrt(252)
        w2, w1 = 1 / sig ** 2, 1 / sig
        res.append({"symbol": sym, "buyhold": sharpe(r), "inv_var": sharpe(r * w2 / w2.mean()), "inv_vol": sharpe(r * w1 / w1.mean()),
                    "maxdd_buyhold": (r.add(1).cumprod() / r.add(1).cumprod().cummax() - 1).min() * 100,
                    "maxdd_inv_vol": ((r * w1 / w1.mean()).add(1).cumprod() / (r * w1 / w1.mean()).add(1).cumprod().cummax() - 1).min() * 100})
    df = pd.DataFrame(res)
    df.to_parquet(OUT / "volmanaged.parquet")
    better = lambda c: int((df[c] > df["buyhold"]).sum())
    return {"stocks": len(df), "median_sharpe": {c: round(df[c].median(), 3) for c in ("buyhold", "inv_var", "inv_vol")},
            "stocks_improved": {"inv_var": better("inv_var"), "inv_vol": better("inv_vol")},
            "median_maxdd_pct": {"buyhold": round(df.maxdd_buyhold.median(), 1), "inv_vol": round(df.maxdd_inv_vol.median(), 1)},
            "mean_sharpe_diff_inv_vol": round((df.inv_vol - df.buyhold).mean(), 3),
            "t_inv_vol": round((df.inv_vol - df.buyhold).mean() / ((df.inv_vol - df.buyhold).std() / np.sqrt(len(df))), 2)}


# ------------------------------------------------------------------ main

def main(cmd):
    OUT.mkdir(exist_ok=True)
    rep_path = OUT / "report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    ev = build_events() if cmd in ("events", "all") or not (OUT / "events.parquet").exists() else pd.read_parquet(OUT / "events.parquet")
    rep["events"] = {"n": len(ev), "years": f"{ev.year.min()}-{ev.year.max()}", "earnings_share": round(ev.earnings.mean() * 100, 1)}
    if cmd in ("a1", "all"):
        rep["a1"] = a1(ev)
    if cmd in ("fill", "all"):
        rep["fill"] = fill_walkforward(ev)
    if cmd in ("value", "all"):
        rep["value"] = value_walkforward(ev)
    if cmd in ("overnight", "all"):
        rep["overnight"] = overnight()
    if cmd in ("preearn", "all"):
        rep["preearn"] = preearnings()
    if cmd in ("vol", "all"):
        rep["vol"] = volmanaged()
    rep_path.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
