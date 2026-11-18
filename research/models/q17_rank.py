"""Q.17.6: cross-sectional ranking on the point-in-time universe, price version (PROJECT_PLAN Q.17.6 /
Q.16.10, 2026-11-18). Also M.4's "does news add to price" baseline once news features exist.

  sectors  sector / industry for the universe tickers names.json and the core file lack (yfinance .info)
  build    every 20 sessions from 2017 (non-overlapping holds), the S&P 500 members of that day that have
           bars (build_universe.py's membership; delisted names without bars are counted, not hidden), with
           price / volume / report features known at that close, and the next 20 sessions from the next
           open: raw return, and the 2b sector residual (r - beta60 x r_fund) demeaned within the day's
           sector (the sector-neutral target)
  score    Colab's walk-forward scores (ridge, then LambdaRank): per-date rank IC against the neutral target
           (Newey-West t), top 20 equal weight after 10 bp a side on turnover vs the day's equal-weight pool,
           and the deflated Sharpe ratio for the two models tried
  Accept: out-of-sample 20-session rank IC > 0.03 (NW t >= 2.5), top 20 after costs beat the pool (t >= 2.0),
  DSR >= 0.95. A pass gives a list of replacement candidates, nothing automatic.

Usage: .venv311/bin/python research/models/q17_rank.py sectors|build|score
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q17_ablation as qa  # noqa: E402

STEP = qa.Q17_DIR / "6"
UNI = Path("/Volumes/Data4T/universe")
BARS = UNI / "bars"
NAMES = HERE.parents[1] / "collectors" / "news" / "gdelt" / "universe_rules" / "names.json"
HOLD, START, FIRST_TEST_YEAR, COST_BP = 20, "2017-01-01", 2020, 10.0
FEATS = ["mom_12_1", "mom_6_1", "mom_3", "rev_1w", "rev_1m", "vol_3m", "dd_1y", "dist_sma50", "dist_sma200",
         "log_dvol", "dvol_trend", "sue", "since_report", "post_report_ret", "beta_1y"]


def _sector_map() -> dict:
    out = {}
    for t, v in json.loads(NAMES.read_text()).items():
        out[t] = {"sector": v.get("sector"), "industry": v.get("industry")}
    out.update(json.loads((qa.STEP2B / "core_sectors.json").read_text()))
    extra = STEP / "extra_sectors.json"
    if extra.exists():
        out.update({k: v for k, v in json.loads(extra.read_text()).items() if v.get("sector")})
    return out


def sectors() -> None:
    import time
    import yfinance as yf
    STEP.mkdir(parents=True, exist_ok=True)
    have = _sector_map()
    todo = sorted(p.name.split(".")[0] for p in BARS.glob("*.daily.parquet") if p.name.split(".")[0] not in have)
    got = {}
    for t in todo:
        try:
            info = yf.Ticker(t).info or {}
        except Exception:  # noqa: BLE001 - a delisted ticker may have no page
            info = {}
        got[t] = {"sector": info.get("sector"), "industry": info.get("industry")}
        time.sleep(0.4)
    (STEP / "extra_sectors.json").write_text(json.dumps(got, indent=1))
    print(f"{sum(bool(v['sector']) for v in got.values())} of {len(todo)} found; none for", [t for t, v in got.items() if not v["sector"]])


def _daily(t: str) -> pd.DataFrame:
    d = pd.read_parquet(BARS / f"{t}.daily.parquet")[["Open", "High", "Low", "Close", "Volume"]]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d[~d.index.duplicated()].sort_index().dropna()


def _reports(t: str) -> pd.DataFrame:
    p = BARS / f"{t}.earnings.json"
    if not p.exists():
        return pd.DataFrame(columns=["known", "surprise"])
    r = pd.DataFrame(json.loads(p.read_text()))
    if r.empty:
        return pd.DataFrame(columns=["known", "surprise"])
    ts = pd.to_datetime(r["time"], utc=True).dt.tz_convert("America/New_York")
    # known from the session it is traded on: before 16:00 that day's close, else the next day's
    known = ts.dt.tz_localize(None).dt.normalize() + pd.to_timedelta((ts.dt.hour >= 16).astype(int), unit="D")
    return pd.DataFrame({"known": known, "surprise": pd.to_numeric(r["surprise"], errors="coerce")}).dropna(subset=["surprise"]).sort_values("known")


def _features(t: str, spy: pd.Series, sec: dict) -> pd.DataFrame | None:
    d = _daily(t)
    if len(d) < 300:
        return None
    c, o, v = d.Close, d.Open, d.Volume
    r = c.pct_change()
    f = pd.DataFrame(index=d.index)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["mom_6_1"] = c.shift(21) / c.shift(126) - 1
    f["mom_3"] = c / c.shift(63) - 1
    f["rev_1w"] = c / c.shift(5) - 1
    f["rev_1m"] = c / c.shift(21) - 1
    f["vol_3m"] = r.rolling(63).std()
    f["dd_1y"] = c / c.rolling(252).max() - 1
    f["dist_sma50"] = c / c.rolling(50).mean() - 1
    f["dist_sma200"] = c / c.rolling(200).mean() - 1
    dv = (c * v).rolling(63).mean()
    f["log_dvol"] = np.log(dv.replace(0, np.nan))
    f["dvol_trend"] = (c * v).rolling(21).mean() / (c * v).rolling(252).mean()
    rs = spy.reindex(d.index).pct_change()
    f["beta_1y"] = r.rolling(252, min_periods=120).cov(rs) / rs.rolling(252, min_periods=120).var()
    rep = _reports(t)
    if len(rep):
        pos = np.searchsorted(rep.known.values, d.index.values, side="right") - 1
        has = pos >= 0
        last_known = pd.Series(np.where(has, rep.known.values[np.maximum(pos, 0)], np.datetime64("NaT")), index=d.index)
        f["sue"] = np.where(has, rep.surprise.values[np.maximum(pos, 0)], np.nan)
        f["since_report"] = (d.index.to_series() - last_known).dt.days
        kpos = np.searchsorted(d.index.values, last_known.values.astype("datetime64[ns]"), side="left") - 1
        ref = np.where(has & (kpos >= 0), c.values[np.maximum(kpos, 0)], np.nan)      # the close before the report
        f["post_report_ret"] = c.values / ref - 1
    else:
        f["sue"], f["since_report"], f["post_report_ret"] = np.nan, np.nan, np.nan
    # next 20 sessions from the next open: raw, and the sector residual
    b, fund = qa._bench(t, d.index, {**sec}) if t in sec else (None, "SPY")
    if b is None:
        import news_events as ne
        b = ne._sessions("SPY").reindex(d.index)
    beta = qa._beta(r, b.Close.pct_change()).fillna(1.0)
    raw = c.shift(-HOLD) / o.shift(-1) - 1
    bm = b.Close.shift(-HOLD) / b.Open.shift(-1) - 1
    f["fwd_raw"] = raw
    f["fwd_resid"] = raw - beta * bm
    f["fund"] = fund
    f["sector"] = (sec.get(t) or {}).get("sector") or "Unknown"
    f["ticker"] = t
    return f


def build() -> Path:
    import news_events as ne
    STEP.mkdir(parents=True, exist_ok=True)
    sec = _sector_map()
    spy = ne._sessions("SPY")
    cal = spy.index[spy.index >= START]
    rebal = cal[::HOLD]
    mem = pd.read_parquet(UNI / "membership.parquet")
    mem["start"], mem["end"] = pd.to_datetime(mem.start), pd.to_datetime(mem.end).fillna(pd.Timestamp("2100-01-01"))
    tickers = sorted(p.name.split(".")[0] for p in BARS.glob("*.daily.parquet"))
    parts = []
    for i, t in enumerate(tickers, 1):
        f = _features(t, spy.Close, sec)
        if f is None:
            continue
        f = f.reindex(rebal).dropna(subset=["fwd_raw"])
        iv = mem[mem.ticker == t]
        if iv.empty:
            continue
        inside = np.zeros(len(f), dtype=bool)
        for _, row in iv.iterrows():
            inside |= (f.index >= row.start) & (f.index < row.end)
        f = f[inside]
        if len(f):
            parts.append(f.reset_index(names="date"))
        if i % 100 == 0:
            print(f"  {i}/{len(tickers)}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["target"] = df.fwd_resid - df.groupby(["date", "sector"]).fwd_resid.transform("mean")
    # cross-sectional ranks of the features per date (to [-0.5, 0.5]); missing -> 0 (the middle)
    for c in FEATS:
        df[f"z_{c}"] = (df.groupby("date")[c].rank(pct=True) - 0.5).fillna(0.0)
    df["year"] = df.date.dt.year
    left = mem[mem.end < pd.Timestamp("2026-11-01")].ticker.unique()
    have = {p.name.split(".")[0] for p in BARS.glob("*.daily.parquet")}
    meta = {"feats": [f"z_{c}" for c in FEATS], "first_test_year": FIRST_TEST_YEAR, "hold": HOLD,
            "members": int(mem.ticker.nunique()), "members_with_bars": int(len(set(mem.ticker) & have)),
            "left_index": int(len(left)), "left_without_bars": int(len(set(left) - have)),
            "rows": int(len(df)), "dates": int(df.date.nunique()), "per_date_median": int(df.groupby("date").size().median()),
            "unknown_sector_share": round(float((df.sector == "Unknown").mean()), 3)}
    out = STEP / "panel.parquet"
    df.to_parquet(out)
    (STEP / "panel_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return out


def _nw(x: np.ndarray, lag: int = 1) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean()), float(x.mean() / math.sqrt(s / n))


def _dsr(sr_obs: float, n: int, skew: float, kurt: float, trials: int, sr_var: float) -> float:
    """Bailey & Lopez de Prado (2014): P(true Sharpe > the best of `trials` under the null), per period."""
    from scipy.stats import norm
    emc = 0.5772156649
    sr0 = math.sqrt(sr_var) * ((1 - emc) * norm.ppf(1 - 1 / trials) + emc * norm.ppf(1 - 1 / (trials * math.e))) if trials > 1 else 0.0
    den = math.sqrt(max(1e-12, 1 - skew * sr_obs + (kurt - 1) / 4 * sr_obs ** 2))
    return float(norm.cdf((sr_obs - sr0) * math.sqrt(n - 1) / den))


def score() -> dict:
    from scipy.stats import kurtosis, skew, spearmanr
    df = pd.read_parquet(STEP / "panel.parquet")
    sc = pd.read_parquet(STEP / "colab" / "scores.parquet")
    df = df.merge(sc, on=["date", "ticker"], how="inner")
    models = [c[len("s_"):] for c in sc.columns if c.startswith("s_")]
    rep = {"rows": int(len(df)), "dates": int(df.date.nunique()), "models": {}}
    excess_by_model = {}
    for m in models:
        ic = df.groupby("date").apply(lambda g: spearmanr(g[f"s_{m}"], g.target).correlation if len(g) > 20 else np.nan).dropna()
        ic_mean, ic_t = _nw(ic.values)
        rows, prev = [], set()
        for d, g in df.groupby("date"):
            top = g.nlargest(20, f"s_{m}")
            names = set(top.ticker)
            turnover = 1.0 if not prev else len(names - prev) / 20
            cost = 2 * turnover * COST_BP / 1e4
            rows.append({"date": d, "top": top.fwd_raw.mean() - cost, "pool": g.fwd_raw.mean()})
            prev = names
        p = pd.DataFrame(rows)
        ex = (p.top - p["pool"]).values
        excess_by_model[m] = ex
        mean, t = _nw(ex)
        rep["models"][m] = {"rank_ic": round(ic_mean, 4), "rank_ic_t_nw": round(ic_t, 2), "periods": int(len(ex)),
                            "top20_excess_per_20d_pct": round(mean * 100, 3), "top20_excess_t_nw": round(t, 2),
                            "top20_ann_sharpe_excess": round(float(ex.mean() / ex.std() * math.sqrt(252 / HOLD)), 3)}
    srs = [v.mean() / v.std() for v in excess_by_model.values()]
    for m, ex in excess_by_model.items():
        sr = ex.mean() / ex.std()
        rep["models"][m]["dsr"] = round(_dsr(sr, len(ex), float(skew(ex)), float(kurtosis(ex, fisher=False)), len(models),
                                             float(np.var(srs)) if len(srs) > 1 else 0.0), 3)
        r = rep["models"][m]
        r["passes"] = bool(r["rank_ic"] > 0.03 and r["rank_ic_t_nw"] >= 2.5 and r["top20_excess_t_nw"] >= 2.0 and r["dsr"] >= 0.95)
    (STEP / "score.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    {"sectors": sectors, "build": build, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
