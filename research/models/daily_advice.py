"""Daily advice per holding from learned models (PROJECT_PLAN Q.16, 2026-11-02).

What the user asked for: advice on the action that a model produces every day, several times a day,
from price, events and news, over the horizons they hold for (1-5 days, 2-4 weeks, and months when a
position is deep under water), starting with INTC. Not hand rules: the models learn from 102 stocks'
daily history (2016-2026) what followed each kind of day, and the advice is what they predict.

  build    one row per stock and session: ~35 features known at that close (returns, trend, drawdown,
           volatility, the last large drop, the earnings cycle from the report dates, peers (a chip basket),
           SPY, VIX, and the news store's daily sentiment / volume / negative-event counts), and labels:
           the return from the next open to the close 5, 20 and 60 sessions later, and the lowest low
           on the way
  train    per horizon, walk-forward by year (train on earlier years with an embargo, test the next,
           2019-2026): LightGBM quantile models (10th / 50th / 90th percentile of the return) and a
           classifier for P(up); an MLP (neural network) on the same features as challenger. Reported:
           rank IC, calibration of P(up), coverage of the 10th percentile, and the decision test (do days
           the model calls "reduce" go on to do worse than days it calls "hold", and "add" better?).
           Then the final models on every labelled day
  predict  today's features for the given symbols (fresh bars from yfinance, news features from the
           store) -> /Volumes/Data4T/daily_advice/latest.json, read by quant_ai's daily action

Usage: .venv311/bin/python research/models/daily_advice.py build|train|predict [SYMBOLS...]
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

EP_DIR = Path(os.getenv("EP_DIR", "/Volumes/Data4T/earnings_policy"))
OUT = Path(os.getenv("ADVICE_DIR", "/Volumes/Data4T/daily_advice"))
FEATURES_PARQUET = Path("/Volumes/Data4T/factor_lab/features.parquet")
HORIZONS = (5, 20, 60)
QUANTILES = (0.1, 0.5, 0.9)
FIRST_TEST_YEAR = 2019
SEED = 7
SEMIS = ["INTC", "TSM", "MU", "AMD", "NVDA", "AVGO", "AMAT", "LRCX", "KLAC", "QCOM", "ADI", "TXN", "MCHP", "ASML", "ARM", "SMCI", "STX"]
NEWS_COLS = ["avg_sentiment_3d", "avg_sentiment_5d", "sentiment_shift_5d", "news_count_5d", "news_burst_20d",
             "negative_event_count_5d", "high_signal_count_3d"]
# What each feature means, for the explanation on the page
LABELS = {
    "r1": ("today's move", "当天涨跌"), "r5": ("5-day move", "5 天涨跌"), "r20": ("20-day move", "20 天涨跌"),
    "r60": ("60-day move", "60 天涨跌"), "dist_sma20": ("vs 20-day average", "相对 20 日均线"),
    "dist_sma50": ("vs 50-day average", "相对 50 日均线"), "dist_sma200": ("vs 200-day average", "相对 200 日均线"),
    "dd_60": ("below the 60-day high", "距 60 日高点"), "dd_252": ("below the 1-year high", "距一年高点"),
    "up_60low": ("above the 60-day low", "距 60 日低点"), "atr_pct": ("daily range (ATR)", "日波动幅度（ATR）"),
    "vol20": ("20-day volatility", "20 日波动率"), "vol_ratio": ("volatility now vs 3 months", "近期波动 / 3 个月波动"),
    "rsi14": ("RSI", "RSI"), "gap": ("opening gap", "开盘跳空"), "close_loc": ("close within the day's range", "收盘在当日区间的位置"),
    "volu_ratio": ("volume vs 20-day average", "成交量 / 20 日均量"), "down_streak": ("down days in a row", "连跌天数"),
    "drop_atr": ("today's drop in ATR", "当天跌幅（ATR 倍数）"), "since_bigdrop": ("sessions since a large drop", "距上次大跌的交易日"),
    "bigdrop_atr": ("size of the last large drop", "上次大跌的幅度"), "since_report": ("sessions since the report", "财报后第几个交易日"),
    "report_ret": ("move on the last report", "上次财报当天涨跌"), "to_report": ("days to the next report", "距下次财报天数"),
    "spy_r5": ("S&P 500, 5 days", "标普 500 近 5 天"), "spy_r20": ("S&P 500, 20 days", "标普 500 近 20 天"),
    "semi_r5": ("chip stocks, 5 days", "芯片股近 5 天"), "semi_r20": ("chip stocks, 20 days", "芯片股近 20 天"),
    "rel_semi20": ("vs chip stocks, 20 days", "相对芯片股（20 天）"), "vix": ("VIX", "VIX"), "vix_pct": ("VIX percentile", "VIX 百分位"),
    "avg_sentiment_3d": ("news sentiment, 3 days", "新闻情绪（3 天）"), "avg_sentiment_5d": ("news sentiment, 5 days", "新闻情绪（5 天）"),
    "sentiment_shift_5d": ("change in news sentiment", "新闻情绪变化"), "news_count_5d": ("news volume, 5 days", "新闻量（5 天）"),
    "news_burst_20d": ("news burst", "新闻爆发度"), "negative_event_count_5d": ("negative events, 5 days", "负面事件数（5 天）"),
    "high_signal_count_3d": ("high-impact news, 3 days", "高影响新闻（3 天）"),
}
FEATS = list(LABELS)


# ------------------------------------------------------------------ data

def _daily(sym: str) -> pd.DataFrame:
    d = pd.read_parquet(EP_DIR / f"{sym}.daily.parquet")[["Open", "High", "Low", "Close", "Volume"]]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d[~d.index.duplicated()].sort_index().dropna()


def symbols() -> list:
    return sorted(p.name.split(".")[0] for p in EP_DIR.glob("*.daily.parquet") if not p.name.startswith("SPY."))


def report_days(sym: str):
    """(sessions holding a report's reaction, scheduled future report dates) from the earnings file."""
    try:
        rows = json.loads((EP_DIR / f"{sym}.earnings.json").read_text())
    except FileNotFoundError:
        return [], []
    past, future = [], []
    for r in rows:
        t = pd.Timestamp(r["time"])
        if t.tzinfo is None:
            continue
        local = t.tz_convert("America/New_York")
        day = local.tz_localize(None).normalize()
        (future if r.get("eps") is None else past).append((day, local.hour >= 16))
    return past, sorted(d for d, _ in future)


def features(d: pd.DataFrame, spy: pd.Series, semi: pd.Series, past_reports, future_reports, news: pd.DataFrame | None,
             macro: pd.DataFrame | None) -> pd.DataFrame:
    c, o, h, l, v = d.Close, d.Open, d.High, d.Low, d.Volume
    pc = c.shift()
    ret = c.pct_change()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f = pd.DataFrame(index=d.index)
    for n in (1, 5, 20, 60):
        f[f"r{n}"] = (c / c.shift(n) - 1) * 100
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = (c / c.rolling(n).mean() - 1) * 100
    f["dd_60"] = (c / c.rolling(60).max() - 1) * 100
    f["dd_252"] = (c / c.rolling(252, min_periods=120).max() - 1) * 100
    f["up_60low"] = (c / c.rolling(60).min() - 1) * 100
    f["atr_pct"] = atr / c * 100
    f["vol20"] = ret.rolling(20).std() * 100
    f["vol_ratio"] = ret.rolling(20).std() / ret.rolling(60).std()
    up, dn = ret.clip(lower=0).rolling(14).mean(), (-ret.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    f["gap"] = (o / pc - 1) * 100
    f["close_loc"] = ((c - l) / (h - l)).where(h > l, 0.5)
    f["volu_ratio"] = v / v.rolling(20).mean().shift()
    down = (ret < 0).astype(int)
    f["down_streak"] = down.groupby((down == 0).cumsum()).cumsum()
    drop = (pc - c) / atr.shift()
    f["drop_atr"] = drop
    big = drop >= 2
    last_big = pd.Series(np.where(big, np.arange(len(d)), np.nan), index=d.index).ffill()
    f["since_bigdrop"] = (np.arange(len(d)) - last_big).clip(upper=250).fillna(250)
    f["bigdrop_atr"] = pd.Series(np.where(big, drop, np.nan), index=d.index).ffill().fillna(0)
    # the earnings cycle: sessions since the last report's reaction, its move, days to the next report
    sess = []
    for day, after_close in past_reports:
        pos = d.index.searchsorted(day)
        if after_close and pos < len(d) and d.index[pos] == day:
            pos += 1
        if pos < len(d):
            sess.append(pos)
    mark = np.full(len(d), np.nan)
    rep_ret = np.full(len(d), np.nan)
    for p in sess:
        mark[p] = p
        rep_ret[p] = ret.iloc[p] * 100
    f["since_report"] = (np.arange(len(d)) - pd.Series(mark).ffill().values).clip(max=120)
    f["report_ret"] = pd.Series(rep_ret).ffill().values
    all_days = sorted({d.index[min(p, len(d) - 1)] for p in sess} | set(future_reports))
    nxt = pd.Series(pd.NaT, index=d.index)
    j = 0
    for i, t in enumerate(d.index):
        while j < len(all_days) and all_days[j] <= t:
            j += 1
        nxt.iloc[i] = all_days[j] if j < len(all_days) else pd.NaT
    f["to_report"] = ((nxt - d.index.to_series()).dt.days).clip(upper=120)
    f["spy_r5"] = (spy / spy.shift(5) - 1).reindex(d.index) * 100
    f["spy_r20"] = (spy / spy.shift(20) - 1).reindex(d.index) * 100
    f["semi_r5"] = ((1 + semi).rolling(5).apply(np.prod, raw=True) - 1).reindex(d.index) * 100
    f["semi_r20"] = ((1 + semi).rolling(20).apply(np.prod, raw=True) - 1).reindex(d.index) * 100
    f["rel_semi20"] = f["r20"] - f["semi_r20"]
    if macro is not None:
        f["vix"] = macro["macro_vix"].reindex(d.index).ffill()
        f["vix_pct"] = macro["macro_vix_pctile_252d"].reindex(d.index).ffill()
    else:
        f["vix"] = f["vix_pct"] = np.nan
    for col in NEWS_COLS:
        f[col] = news[col].reindex(d.index) if news is not None and col in news else np.nan
    for col in ("news_count_5d", "negative_event_count_5d", "high_signal_count_3d"):
        f[col] = np.log1p(f[col])
    return f[FEATS]


def labels(d: pd.DataFrame) -> pd.DataFrame:
    o1 = d.Open.shift(-1)                       # decided at the close, done at the next open
    lab = pd.DataFrame(index=d.index)
    for hz in HORIZONS:
        lab[f"y{hz}"] = (d.Close.shift(-hz) / o1 - 1) * 100
        low = d.Low[::-1].rolling(hz, min_periods=hz).min()[::-1].shift(-1)
        lab[f"mdd{hz}"] = (low / o1 - 1) * 100
    return lab


def point_in_time(df: pd.DataFrame) -> pd.DataFrame:
    """With UNIVERSE_MEMBERSHIP set (research/universe/build_universe.py), keep a stock's day only while
    it was in the index then: the universe the user faced on that day, not today's survivors."""
    path = os.getenv("UNIVERSE_MEMBERSHIP")
    if not path:
        return df
    m = pd.read_parquet(path)
    # renamed tickers: the old name's membership belongs to the bars stored under the new one
    renames = {"FB": "META", "ANTM": "ELV", "ABC": "COR", "BLL": "BALL", "CDAY": "DAY", "COG": "CTRA", "PEAK": "DOC", "HCP": "DOC",
               "FLT": "CPAY", "WLTW": "WTW", "RE": "EG", "FISV": "FI", "DISCA": "WBD", "CBS": "PARA", "VIAC": "PARA", "PKI": "RVTY",
               "SYMC": "GEN", "NLOK": "GEN", "GPS": "GAP", "ADS": "BFH", "KORS": "CPRI", "TSO": "ANDV", "HRS": "LHX", "UTX": "RTX"}
    m["symbol"] = m.ticker.replace(renames).str.replace(".", "-", regex=False)
    keep = np.zeros(len(df), bool)
    for sym, iv in m.groupby("symbol"):
        rows = (df.symbol == sym).values
        if not rows.any():
            continue
        d = df.date.values
        inside = np.zeros(len(df), bool)
        for st, en in zip(iv.start, iv.end):
            inside |= (d >= np.datetime64(st)) & ((d < np.datetime64(en)) if pd.notna(en) else True)
        keep |= rows & inside
    print(f"point in time: {keep.sum():,} of {len(df):,} stock-days were index members on the day")
    return df[keep].reset_index(drop=True)


def news_frame() -> dict:
    f = pd.read_parquet(FEATURES_PARQUET, columns=["trade_date", "symbol"] + NEWS_COLS + ["macro_vix", "macro_vix_pctile_252d"])
    f = f[f.trade_date.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)].copy()
    f["date"] = pd.to_datetime(f.trade_date)
    f = f.sort_values("date").drop_duplicates(["symbol", "date"], keep="last")      # a day can be stored twice
    macro = f[f.symbol == "AAPL"].set_index("date")[["macro_vix", "macro_vix_pctile_252d"]].sort_index()
    by = {s: g.set_index("date")[NEWS_COLS].sort_index() for s, g in f.groupby("symbol")}
    return by, macro


def semi_returns(daily: dict) -> pd.Series:
    rets = pd.DataFrame({s: daily[s].Close.pct_change() for s in SEMIS if s in daily})
    return rets.mean(axis=1)


def build() -> pd.DataFrame:
    syms = symbols()
    daily = {s: _daily(s) for s in syms}
    spy = _daily("SPY").Close
    semi = semi_returns(daily)
    news, macro = news_frame()
    parts = []
    for s in syms:
        d = daily[s]
        if len(d) < 300:
            continue
        past, future = report_days(s)
        f = features(d, spy, semi, past, future, news.get(s), macro)
        x = f.join(labels(d))
        x["symbol"], x["date"] = s, x.index
        parts.append(x[x.index >= "2016-01-01"])
    df = pd.concat(parts, ignore_index=True)
    df = point_in_time(df)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "dataset.parquet")
    print(f"dataset: {len(df):,} stock-days, {df.symbol.nunique()} stocks, {df.date.min().date()}..{df.date.max().date()}")
    return df


# ------------------------------------------------------------------ models

def _lgb(objective, alpha=None):
    import lightgbm as lgb
    kw = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, verbose=-1)
    if objective == "binary":
        return lgb.LGBMClassifier(**kw)
    return lgb.LGBMRegressor(objective="quantile", alpha=alpha, **kw)


def advice_from(p_up20: float, q50_20: float, q10_20: float, p_up5: float, thr: dict) -> str:
    """The model's call: reduce when the 20-day odds and median both point down, add when both point up
    with a bounded downside; otherwise hold. Thresholds are set in train() from the walk-forward."""
    if p_up20 <= thr["reduce_p"] and q50_20 < 0:
        return "reduce"
    if p_up20 >= thr["add_p"] and q50_20 > 0 and q10_20 > thr["add_q10"]:
        return "add"
    return "hold"


def train() -> dict:
    from scipy.stats import spearmanr
    df = pd.read_parquet(OUT / "dataset.parquet")
    df[FEATS] = df[FEATS].replace([np.inf, -np.inf], np.nan)          # a zero-volume stretch divides by zero
    df["year"] = df.date.dt.year
    preds = []
    for y in range(FIRST_TEST_YEAR, int(df.year.max()) + 1):
        te = df[df.year == y].copy()
        if te.empty:
            continue
        for hz in HORIZONS:
            embargo = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=int(hz * 1.5))
            tr = df[(df.date < embargo) & df[f"y{hz}"].notna()]
            m = _lgb("binary").fit(tr[FEATS], (tr[f"y{hz}"] > 0).astype(int))
            te[f"p_up{hz}"] = m.predict_proba(te[FEATS])[:, 1]
            for q in QUANTILES:
                mq = _lgb("quantile", q).fit(tr[FEATS], tr[f"y{hz}"])
                te[f"q{int(q * 100)}_{hz}"] = mq.predict(te[FEATS])
            if hz == 20:                                         # the neural-network challenger
                from sklearn.neural_network import MLPRegressor
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.impute import SimpleImputer
                sub = tr.sample(min(len(tr), 120000), random_state=SEED)
                mlp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=60, random_state=SEED))
                mlp.fit(sub[FEATS], sub["y20"].clip(-60, 60))
                te["mlp_20"] = mlp.predict(te[FEATS])
        preds.append(te)
        print(f"  walk-forward {y}: {len(te):,} test days", flush=True)
    pr = pd.concat(preds, ignore_index=True)
    pr.to_parquet(OUT / "walkforward.parquet")

    rep = {"test_years": f"{FIRST_TEST_YEAR}-{int(df.year.max())}", "test_days": int(len(pr)), "horizons": {}}
    for hz in HORIZONS:
        s = pr.dropna(subset=[f"y{hz}"])
        ic = s.groupby("date").apply(lambda g: spearmanr(g[f"q50_{hz}"], g[f"y{hz}"])[0] if len(g) > 10 else np.nan).dropna()
        ic_n = ic.iloc[::hz]                                      # non-overlapping dates for the t-stat
        s["bin"] = pd.qcut(s[f"p_up{hz}"], 5, labels=False, duplicates="drop")
        cal = [{"pred": round(g[f"p_up{hz}"].mean() * 100, 1), "actual": round((g[f"y{hz}"] > 0).mean() * 100, 1), "n": len(g)}
               for _, g in s.groupby("bin")]
        rep["horizons"][hz] = {"ic": round(float(ic.mean()), 4), "ic_t": round(float(ic_n.mean() / (ic_n.std() / math.sqrt(len(ic_n)))), 2),
                               "q10_breached_pct": round(float((s[f"y{hz}"] < s[f"q10_{hz}"]).mean() * 100), 1),
                               "q90_exceeded_pct": round(float((s[f"y{hz}"] > s[f"q90_{hz}"]).mean() * 100), 1),
                               "base_up_pct": round(float((s[f"y{hz}"] > 0).mean() * 100), 1), "calibration": cal}
        if hz == 20:
            mic = s.groupby("date").apply(lambda g: spearmanr(g["mlp_20"], g["y20"])[0] if len(g) > 10 else np.nan).dropna()
            mn = mic.iloc[::20]
            rep["horizons"][hz]["mlp_ic"] = round(float(mic.mean()), 4)
            rep["horizons"][hz]["mlp_ic_t"] = round(float(mn.mean() / (mn.std() / math.sqrt(len(mn)))), 2)

    # the decision test: thresholds from the test years before 2023, judged on 2023 on (one look)
    s = pr.dropna(subset=["y20", "y5"]).copy()
    early, late = s[s.date < "2023-01-01"], s[s.date >= "2023-01-01"]
    thr = {"reduce_p": float(early.p_up20.quantile(0.15)), "add_p": float(early.p_up20.quantile(0.85)),
           "add_q10": float(early.q10_20.quantile(0.30))}
    s["call"] = [advice_from(a, b, c, e, thr) for a, b, c, e in zip(s.p_up20, s.q50_20, s.q10_20, s.p_up5)]

    def bucket(g):
        days = g.groupby("date")["y20"].mean()
        return {"days": int(len(g)), "mean20": round(float(g.y20.mean()), 2), "up20": round(float((g.y20 > 0).mean() * 100), 1),
                "mean5": round(float(g.y5.mean()), 2), "mdd20": round(float(g.mdd20.mean()), 2)}
    out = {}
    for period, part in (("2019-2022", s[s.date < "2023-01-01"]), ("2023-2026 (held out for thresholds)", s[s.date >= "2023-01-01"])):
        out[period] = {k: bucket(part[part.call == k]) for k in ("reduce", "hold", "add")}
        r, h_ = part[part.call == "reduce"].y20, part[part.call == "hold"].y20
        out[period]["reduce_minus_hold_t"] = round(float((r.mean() - h_.mean()) / math.sqrt(r.var() / len(r) + h_.var() / len(h_))), 2)
        a_ = part[part.call == "add"].y20
        out[period]["add_minus_hold_t"] = round(float((a_.mean() - h_.mean()) / math.sqrt(a_.var() / len(a_) + h_.var() / len(h_))), 2)
    deep = s[s.dd_252 <= -30]                                    # INTC's kind of day: 30%+ below the year's high
    out["deep_drawdown_days"] = {k: bucket(deep[deep.call == k]) for k in ("reduce", "hold", "add") if (deep.call == k).any()}
    rep["decision"] = {"thresholds": {k: round(v, 3) for k, v in thr.items()}, **out}

    rep["calibration"] = calib = calibrate(pr)
    rep["direction_reliable"] = reliable = _reliable(rep["decision"])

    # final models on every labelled day
    import joblib
    final = {}
    for hz in HORIZONS:
        tr = df[df[f"y{hz}"].notna()]
        final[f"p_up{hz}"] = _lgb("binary").fit(tr[FEATS], (tr[f"y{hz}"] > 0).astype(int))
        for q in QUANTILES:
            final[f"q{int(q * 100)}_{hz}"] = _lgb("quantile", q).fit(tr[FEATS], tr[f"y{hz}"])
    joblib.dump({"models": final, "features": FEATS, "thresholds": thr, "calibration": calib, "direction_reliable": reliable,
                 "trained_at": datetime.now(timezone.utc).isoformat(),
                 "trained_through": str(df.date.max().date())}, OUT / "models.joblib")
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return rep


def calibrate(pr: pd.DataFrame) -> dict:
    """The raw models are over-confident (P(up) 84% came true 62% of the time; the 10th percentile was
    breached ~20%, not 10%). From the walk-forward predictions: P(up) mapped through an isotonic fit to
    what happened, and each quantile's distance from the median scaled so it is breached as often as
    its level says (10% / 90%)."""
    from sklearn.isotonic import IsotonicRegression
    out = {}
    for hz in HORIZONS:
        s = pr.dropna(subset=[f"y{hz}"])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(s[f"p_up{hz}"], (s[f"y{hz}"] > 0).astype(float))
        grid = np.linspace(0, 1, 101)
        lo_gap = (s[f"q50_{hz}"] - s[f"q10_{hz}"]).clip(lower=0.1)
        hi_gap = (s[f"q90_{hz}"] - s[f"q50_{hz}"]).clip(lower=0.1)
        k10 = float(np.quantile((s[f"q50_{hz}"] - s[f"y{hz}"]) / lo_gap, 0.90))   # P(y < q50 - k10 * gap) = 10%
        k90 = float(np.quantile((s[f"y{hz}"] - s[f"q50_{hz}"]) / hi_gap, 0.90))
        out[str(hz)] = {"p_grid": [round(float(v), 4) for v in iso.predict(grid)], "k10": round(k10, 3), "k90": round(k90, 3)}
    return out


def _reliable(decision: dict) -> bool:
    """A direction call is shown only when, in the years its thresholds did not see (2023-2026),
    "reduce" days did significantly worse than "hold" days and "add" days significantly better, each
    on at least 200 days. 2026-11-02: neither held (reduce did better than hold), so the action says hold."""
    late = decision.get("2023-2026 (held out for thresholds)") or {}
    return (late.get("reduce_minus_hold_t", 0) <= -2 and late.get("add_minus_hold_t", 0) >= 2
            and (late.get("reduce") or {}).get("days", 0) >= 200 and (late.get("add") or {}).get("days", 0) >= 200)


def _apply_calibration(raw: dict, cal: dict) -> dict:
    p = cal["p_grid"][int(round(min(max(raw["p_up_raw"], 0.0), 1.0) * 100))] * 100
    lo = raw["q50"] - cal["k10"] * max(raw["q50"] - raw["q10_raw"], 0.1)
    hi = raw["q50"] + cal["k90"] * max(raw["q90_raw"] - raw["q50"], 0.1)
    return {"p_up": round(p, 1), "q10": round(lo, 2), "q50": raw["q50"], "q90": round(hi, 2)}


# ------------------------------------------------------------------ today

def _fresh(sym: str) -> pd.DataFrame:
    import yfinance as yf
    d = yf.Ticker(sym).history(period="2y", interval="1d", auto_adjust=False)[["Open", "High", "Low", "Close", "Volume"]]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d[~d.index.duplicated()].dropna()


def _latest_news(sym: str) -> pd.DataFrame | None:
    """The news features for the last days from the live store (the parquet is a snapshot)."""
    try:
        from dotenv import load_dotenv
        from pymongo import MongoClient
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        uri = (os.getenv("HOST_LOCAL_MONGO_URI") or os.getenv("LOCAL_MONGO_URI") or "mongodb://root:root@127.0.0.1:37018/")
        col = MongoClient(uri.replace("host.docker.internal", "127.0.0.1"), serverSelectionTimeoutMS=4000)["quant_data"]["daily_symbol_features"]
        rows = list(col.find({"symbol": sym, "trade_date": {"$type": "string", "$regex": "^20"}}, {"_id": 0, "trade_date": 1, **{c: 1 for c in NEWS_COLS}},
                             sort=[("trade_date", -1)], limit=30))
        if not rows:
            return None
        f = pd.DataFrame(rows)
        f.index = pd.to_datetime(f.pop("trade_date"))
        return f[~f.index.duplicated(keep="first")].sort_index()
    except Exception:  # noqa: BLE001 - no store: the news features stay missing, which the models handle
        return None


NEG = r"plunge|plummet|tumble|slump|sink|fall|fell|drop|slide|crash|cut|weak|miss|downgrade|warn|delay|probe|lawsuit|sell-?off"
CAUSES = OUT / "drop_causes.json"


def recent_events(sym: str, d: pd.DataFrame, lookback: int = 30) -> list:
    """Each session in the last `lookback` down 5%+ or 2 x ATR+, with its cause: the most negative
    headlines of that day and the day before, put in one sentence (English and Chinese) by the local
    model from the headlines alone; cached per symbol and day."""
    import re
    cache = json.loads(CAUSES.read_text()) if CAUSES.exists() else {}
    pc = d.Close.shift()
    tr = pd.concat([d.High - d.Low, (d.High - pc).abs(), (d.Low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().shift()
    ret = d.Close.pct_change() * 100
    out = []
    big = [i for i in range(max(1, len(d) - lookback), len(d))
           if ret.iloc[i] <= -5 or (atr.iloc[i] and (pc.iloc[i] - d.Close.iloc[i]) / atr.iloc[i] >= 2)]
    for i in sorted(sorted(big, key=lambda i: ret.iloc[i])[:3]):            # the three largest, in date order
        t = d.index[i]
        r = ret.iloc[i]
        key = f"{sym}:{t.date()}"
        if key not in cache:
            heads = _drop_headlines(sym, d.index[i - 1], t)
            cache[key] = {**_summarise(sym, t, r, heads), "sources": heads[:3]} if heads else None
        ev = cache.get(key) or {"cause_en": "No news found that explains it", "cause_zh": "新闻库里没有找到能解释这次下跌的消息", "sources": []}
        out.append({"date": str(t.date()), "ret_pct": round(float(r), 1), **ev})
    CAUSES.write_text(json.dumps(cache, indent=1, default=str))
    return out


def _drop_headlines(sym: str, prev: pd.Timestamp, day: pd.Timestamp) -> list:
    import re
    try:
        from dotenv import load_dotenv
        from pymongo import MongoClient
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        uri = (os.getenv("HOST_LOCAL_MONGO_URI") or os.getenv("LOCAL_MONGO_URI") or "mongodb://root:root@127.0.0.1:37018/")
        db = MongoClient(uri.replace("host.docker.internal", "127.0.0.1"), serverSelectionTimeoutMS=4000)["quant_data"]
        q = {"symbol": sym, "date": {"$gte": prev.strftime("%Y%m%d"), "$lt": day.strftime("%Y%m%d") + "z"}}
        proj = {"_id": 0, "title": 1, "url": 1, "llm_sentiment_final": 1, "llm_event_type_a": 1}
        rows = list(db["news_articles_company_matched_v2"].find(q, proj).limit(400)) + list(db["news_articles"].find(q, proj).limit(400))
    except Exception:  # noqa: BLE001
        return []
    seen, out = set(), []
    for r in rows:
        t = (r.get("title") or "").strip()
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        s = r.get("llm_sentiment_final")
        score = -(s if isinstance(s, (int, float)) else 0.0) + (0.5 if re.search(NEG, t, re.I) else 0) + (0.3 if r.get("llm_event_type_a") == "earnings" else 0)
        out.append({"title": t, "url": r.get("url") or "", "score": round(score, 2)})
    return sorted(out, key=lambda r: -r["score"])[:12]


def _summarise(sym: str, day: pd.Timestamp, ret: float, heads: list) -> dict:
    """One sentence on the cause from the headlines only, by the local model (LM Studio)."""
    import re
    import requests
    lines = "\n".join(f"- {h['title']}" for h in heads)
    prompt = (f"{sym} closed {ret:.1f}% on {day.date()}. Headlines from the day before and that day:\n{lines}\n\n"
              "Using only facts stated in these headlines, give the most likely cause of the drop. Reply with JSON only: "
              '{"cause_en": "<one sentence, <= 25 words>", "cause_zh": "<the same in Chinese, <= 50 characters>"}')
    url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1").rstrip("/")
    model = os.getenv("ADVICE_LLM", "qwen3.5-9b-mlx")
    try:
        # An empty think block first, or qwen writes minutes of reasoning before the answer
        r = requests.post(f"{url}/chat/completions", timeout=120, json={
            "model": model, "temperature": 0.1, "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "<think>\n\n</think>\n\n"}]})
        txt = re.sub(r"<think>.*?</think>", "", r.json()["choices"][0]["message"]["content"], flags=re.S)
        obj = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        if obj.get("cause_en") and obj.get("cause_zh"):
            return {"cause_en": obj["cause_en"].strip(), "cause_zh": obj["cause_zh"].strip(), "by": model}
    except Exception:  # noqa: BLE001 - no model: the top headline stands in
        pass
    return {"cause_en": heads[0]["title"], "cause_zh": heads[0]["title"], "by": "headline"}


def _news_study() -> dict:
    """The event study's lines the action quotes (news_events.py study)."""
    try:
        rep = json.loads((OUT / "news_study.json").read_text())
    except (OSError, ValueError):
        return {}
    keep = ("guidance cut", "bad earnings news", "upgrade", "fell 5%+ without bad news", "bad news + fell 5%+ that day")
    return {k: {h: rep[k].get(h) for h in ("days", "5d_vs_all", "20d_vs_all", "60d_vs_all")} for k in keep if k in rep}


def _calendar_study() -> dict:
    """The scheduled-event effects the action quotes (calendar_events.py study)."""
    try:
        rep = json.loads((OUT / "calendar_study.json").read_text())
    except (OSError, ValueError):
        return {}
    s, a, k = rep.get("spy_session_return", {}), rep.get("spy_next5_when_ahead", {}), rep.get("stock_excess_by_days_to_report", {})
    return {"fomc_ahead_5d": a.get("FOMC within 5 sessions"), "no_fomc_5d": a.get("no FOMC within 5"),
            "opex_week": (s.get("options-expiration week") or {}).get("vs_other_days"),
            "pre_holiday": (s.get("eve of a holiday") or {}).get("vs_other_days"),
            "day_before_report": k.get("1 days before the report")}


def predict(syms: list) -> dict:
    import joblib
    bundle = joblib.load(OUT / "models.joblib")
    models, thr = bundle["models"], bundle["thresholds"]
    rep = json.loads((OUT / "report.json").read_text())
    spy = _fresh("SPY").Close
    peers = {s: _fresh(s) for s in SEMIS}
    semi = semi_returns(peers)
    _, macro = news_frame()
    try:
        vix = _fresh("^VIX").Close
        hist = vix.rolling(252, min_periods=60).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
        macro = pd.DataFrame({"macro_vix": vix, "macro_vix_pctile_252d": hist})
    except Exception:  # noqa: BLE001
        pass
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trained_through": bundle["trained_through"],
           "direction_reliable": bool(bundle.get("direction_reliable")),
           "model_quality": {str(h): {k: rep["horizons"][str(h)].get(k) for k in ("ic", "ic_t", "q10_breached_pct", "base_up_pct")}
                             for h in HORIZONS},
           "decision_test": rep.get("decision"), "symbols": {},
           "news_study": _news_study(), "calendar_study": _calendar_study()}
    for s in syms:
        d = peers.get(s) if s in peers else _fresh(s)
        past, future = report_days(s)
        f = features(d, spy, semi, past, future, _latest_news(s), macro)
        x = f.iloc[[-1]]
        res = {"as_of": str(d.index[-1].date()), "close": round(float(d.Close.iloc[-1]), 2), "horizons": {}}
        raw = {}
        for hz in HORIZONS:
            r = {"p_up_raw": float(models[f"p_up{hz}"].predict_proba(x)[0, 1]),
                 "q10_raw": float(models[f"q10_{hz}"].predict(x)[0]), "q50": round(float(models[f"q50_{hz}"].predict(x)[0]), 2),
                 "q90_raw": float(models[f"q90_{hz}"].predict(x)[0])}
            raw[hz] = r
            res["horizons"][hz] = _apply_calibration(r, bundle["calibration"][str(hz)])
        h20 = raw[20]
        model_call = advice_from(h20["p_up_raw"], h20["q50"], h20["q10_raw"], raw[5]["p_up_raw"], thr)
        res["model_call"] = model_call
        res["call"] = model_call if bundle.get("direction_reliable") else "hold"
        res["direction_reliable"] = bool(bundle.get("direction_reliable"))
        res["events"] = recent_events(s, d)
        try:                                             # Q.16.7: the barriers, aimed at getting out of a losing holding
            import barrier_model
            res["barrier"] = barrier_model.live(s, d, f)
        except Exception as e:  # noqa: BLE001 - the rest of the advice stands without it
            res["barrier_error"] = f"{type(e).__name__}: {e}"
        out["symbols"][s] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["symbols"], indent=1, default=str))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predict"
    if cmd == "build":
        build()
    elif cmd == "train":
        train()
    elif cmd == "calibrate":                     # recalibrate the saved models from the saved walk-forward
        import joblib
        b = joblib.load(OUT / "models.joblib")
        rep = json.loads((OUT / "report.json").read_text())
        b["calibration"] = rep["calibration"] = calibrate(pd.read_parquet(OUT / "walkforward.parquet"))
        b["direction_reliable"] = rep["direction_reliable"] = _reliable(rep["decision"])
        joblib.dump(b, OUT / "models.joblib")
        (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
        print({h: {k: v for k, v in c.items() if k != "p_grid"} for h, c in rep["calibration"].items()}, "direction_reliable:", rep["direction_reliable"])
    else:
        predict(sys.argv[2:] or ["INTC"])
