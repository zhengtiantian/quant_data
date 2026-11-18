"""Q.17.3 stage 1: do analysts' rating and price-target changes predict the next 20 / 60 sessions?
(PROJECT_PLAN Q.17.3, 2026-11-18; Womack 1996)

  fetch   every firm's rating / target change for the 101 core names from Yahoo (yfinance
          upgrades_downgrades, about 2012 on), 1.5 s apart, as quant_ai/markets/analysts.py parses it
          -> Q17_DIR/3/analyst_actions.parquet (not markets.db: the live store keeps only followed names)
  study   two pre-set tests, defined on 2018-2023 and tested on 2024-2026, each against SPY excess and,
          as a check, the 2b sector residual:
            event days   a downgrade / upgrade / target cut / target raise placed on the session;
                         returns from that session's open (news_events' x)
            state        over the last 20 sessions: net downgrades (< 0), net upgrades (> 0), mean target
                         change <= -5% or >= +5%; returns from the next open (news_events' n)
          A change is placed on the first session it could be traded on: a date-only stamp (00:00:00)
          on the next session's open; a timed one (UTC) before 09:30 New York on that session, else the
          next. Target change = pt_current / pt_prior - 1 within a row (immune to splits). t-stats are
          Newey-West on the per-date means (lag = horizon), since the windows overlap.
  Accept: out-of-sample |t| >= 2.5 with the in-sample sign.

Usage: .venv311/bin/python research/models/q17_analysts.py fetch|study
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q17_ablation as qa  # noqa: E402

STEP = qa.Q17_DIR / "3"
IN_SAMPLE, OUT_SAMPLE = ("2018-01-01", "2023-12-31"), ("2024-01-01", "2026-12-31")
HORIZONS = (20, 60)


def fetch() -> Path:
    import yfinance as yf
    STEP.mkdir(parents=True, exist_ok=True)
    rows, failed = [], {}
    for i, s in enumerate(qa._core(), 1):
        try:
            ud = yf.Ticker(s).upgrades_downgrades
        except Exception as e:  # noqa: BLE001 - one name failing must not stop the rest
            failed[s] = f"{type(e).__name__}: {e}"
            time.sleep(5)
            continue
        if ud is not None and len(ud):
            for ts, r in ud.iterrows():
                num = lambda v: float(v) if v is not None and v == v and float(v) > 0 else None  # noqa: E731
                rows.append({"symbol": s, "graded_at": pd.Timestamp(ts), "firm": str(r.get("Firm") or "").strip(),
                             "action": str(r.get("Action") or "").lower() or None,
                             "from_grade": r.get("FromGrade"), "to_grade": r.get("ToGrade"),
                             "pt_current": num(r.get("currentPriceTarget")), "pt_prior": num(r.get("priorPriceTarget"))})
        print(f"  {i:3d} {s}: {0 if ud is None else len(ud)} rows", flush=True)
        time.sleep(1.5)
    df = pd.DataFrame(rows)
    out = STEP / "analyst_actions.parquet"
    df.to_parquet(out)
    (STEP / "fetch_report.json").write_text(json.dumps({"rows": len(df), "names": int(df.symbol.nunique()), "failed": failed,
                                                        "first": str(df.graded_at.min()), "last": str(df.graded_at.max())}, indent=1))
    print(f"{out}: {len(df):,} rows for {df.symbol.nunique()} names; failed {list(failed)}")
    return out


def _place(g: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    """The session each change is first tradable on (NaT past the last bar)."""
    ts = pd.to_datetime(g.graded_at)
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    ny = ts.dt.tz_convert("America/New_York")
    date_only = (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)
    day = ts.dt.tz_localize(None).dt.normalize().where(date_only, ny.dt.tz_localize(None).dt.normalize())
    pre_open = ~date_only & ((ny.dt.hour * 60 + ny.dt.minute) < 9 * 60 + 30)
    v = idx.values
    pos = np.searchsorted(v, day.values.astype("datetime64[ns]"))
    same = (pos < len(idx)) & (v[np.minimum(pos, len(idx) - 1)] == day.values.astype("datetime64[ns]"))
    pos = np.where(same & ~pre_open.values, pos + 1, pos)
    return pd.Series(np.where(pos < len(idx), idx.values[np.minimum(pos, len(idx) - 1)], np.datetime64("NaT")), index=g.index)


def _sessions_frame(acts: pd.DataFrame) -> pd.DataFrame:
    import news_events as ne
    parts = []
    for s, g in acts.groupby("symbol"):
        idx = ne._sessions(s).index
        g = g.assign(session=_place(g, idx)).dropna(subset=["session"])
        g["up"] = (g.action == "up").astype(int)
        g["down"] = (g.action == "down").astype(int)
        pc = (g.pt_current / g.pt_prior - 1).where(g.pt_current.notna() & g.pt_prior.notna())
        g["pt_chg"] = pc.clip(-0.9, 3.0)
        a = g.groupby("session").agg(up=("up", "sum"), down=("down", "sum"), pt_sum=("pt_chg", "sum"), pt_n=("pt_chg", "count"),
                                     pt_cut=("pt_chg", lambda x: int((x < 0).sum())), pt_raise=("pt_chg", lambda x: int((x > 0).sum())))
        a = a.reindex(idx).fillna(0)
        roll = a.rolling(20, min_periods=1).sum()
        f = pd.DataFrame({"symbol": s, "session": idx,
                          "e_down": a.down.values > 0, "e_up": a.up.values > 0, "e_pt_cut": a.pt_cut.values > 0, "e_pt_raise": a.pt_raise.values > 0,
                          "net20": (roll.up - roll.down).values,
                          "pt20": (roll.pt_sum / roll.pt_n.replace(0, np.nan)).values})
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def _nw_t(x: pd.Series, dates: pd.Series, lag: int) -> dict:
    """Mean and Newey-West t of the per-date means (one value per date, in date order)."""
    g = pd.DataFrame({"x": x.values, "d": dates.values}).dropna().groupby("d")["x"].mean().sort_index()
    n = len(g)
    if n < 30:
        return {"n_dates": n}
    e = g.values - g.values.mean()
    lag = min(lag, n - 1)
    s = e @ e / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return {"mean": round(float(g.mean()), 3), "t_nw": round(float(g.mean() / math.sqrt(s / n)), 2), "n_dates": int(n)}


def study() -> dict:
    import news_events as ne
    acts = pd.read_parquet(STEP / "analyst_actions.parquet")
    sf = _sessions_frame(acts)
    targets = {"spy": ne._forward(), "sector": qa._forward_resid()}
    kinds = {
        "event: downgrade": ("x", lambda d: d.e_down), "event: upgrade": ("x", lambda d: d.e_up),
        "event: target cut": ("x", lambda d: d.e_pt_cut), "event: target raise": ("x", lambda d: d.e_pt_raise),
        "state: net downgrades, 20 sessions": ("n", lambda d: d.net20 < 0), "state: net upgrades, 20 sessions": ("n", lambda d: d.net20 > 0),
        "state: mean target change <= -5%, 20 sessions": ("n", lambda d: d.pt20 <= -0.05),
        "state: mean target change >= +5%, 20 sessions": ("n", lambda d: d.pt20 >= 0.05),
    }
    rep = {"actions": int(len(acts)), "names": int(acts.symbol.nunique()), "results": {}}
    for tname, fw in targets.items():
        df = fw.merge(sf, on=["symbol", "session"], how="left")
        for c in ("e_down", "e_up", "e_pt_cut", "e_pt_raise"):
            df[c] = df[c].fillna(False).astype(bool)
        for name, (pref, sel) in kinds.items():
            m = sel(df).fillna(False)
            out = {}
            for per, (a, b) in {"in": IN_SAMPLE, "out": OUT_SAMPLE}.items():
                w = df.session.between(a, b)
                for h in HORIZONS:
                    col = f"{pref}{h}"
                    allm = df.loc[w, col].mean()
                    sub = df[w & m]
                    out[f"{per}_{h}d"] = {"days": int(len(sub)), **_nw_t(sub[col] - allm, sub.session, h)}
            ok = {}
            for h in HORIZONS:
                i, o = out[f"in_{h}d"], out[f"out_{h}d"]
                ok[f"{h}d"] = bool("t_nw" in o and abs(o["t_nw"]) >= 2.5 and "mean" in i and np.sign(o["mean"]) == np.sign(i["mean"]))
            rep["results"].setdefault(name, {})[tname] = {**out, "passes": ok}
    (STEP / "study.json").write_text(json.dumps(rep, indent=1))
    print(f"{rep['actions']:,} changes, {rep['names']} names; excess vs all days, mean % (Newey-West t)")
    for name, v in rep["results"].items():
        for tname, r in v.items():
            cells = " | ".join(f"{p} {h}d: {r[f'{p}_{h}d'].get('mean', '-')} ({r[f'{p}_{h}d'].get('t_nw', '-')}, n {r[f'{p}_{h}d']['days']})"
                               for p in ("in", "out") for h in HORIZONS)
            print(f"  {name:48s} {tname:6s} {cells}  pass {r['passes']}")
    return rep


if __name__ == "__main__":
    {"fetch": fetch, "study": study}[sys.argv[1] if len(sys.argv) > 1 else "study"]()
