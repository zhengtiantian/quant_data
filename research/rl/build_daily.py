#!/usr/bin/env python3
"""Daily decision points for the holdings advisor (PROJECT_PLAN Q.7 phase 2, 2026-10-26).

    python research/rl/build_daily.py        # -> EP_DIR/rl/daily_outcomes.parquet, daily_decisions.parquet, daily_actions.parquet

Every fifth trading day of every stock since 2016 (staggered by stock) is a decision taken after
that day's close, as the 13:00 Dublin report is, before the next open. State: what is known then:
price features computed here (returns, distance to the 20- and 60-day range, moving averages,
ATR, volatility) and the feature store's row for that day (news, LLM sentiment, analyst, macro,
earnings calendar, retail sentiment), without its forward returns and without the 13F holdings
withdrawn in M.7. The path: the next 20 daily bars. A market entry fills at the next open, limits
are set from the close, at the regular-session cost (0.05% a side). Outcomes for every action of
the daily grammar (256).
"""
from __future__ import annotations

import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import earnings_pulse as ep  # noqa: E402
from rl import grammar, simulator  # noqa: E402

OUT = ep.D / "rl"
EVERY = 5
START = "2016-01-01"
DROP = ("future_ret_", "excess_ret_", "inst_holding", "builtAt", "name", "benchmark_symbol", "trade_date", "_id",
        "spy_ret_", "qqq_ret_", "benchmark_ret_")        # spy/qqq/benchmark_ret_Nd are forward-looking in the store


def store_features() -> pd.DataFrame:
    path = OUT / "daily_store.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from dotenv import load_dotenv
    from pymongo import MongoClient
    load_dotenv(ep.ROOT / ".env", override=False)
    col = MongoClient(os.getenv("HOST_LOCAL_MONGO_URI"))["quant_data"]["daily_symbol_features"]
    rows = list(col.find({"date": {"$gte": "2015-06-01"}}, {"_id": 0}))
    df = pd.DataFrame(rows)
    keep = [c for c in df.columns if not any(c.startswith(d) for d in DROP)]
    df = df[keep]
    num = [c for c in df.columns if c not in ("symbol", "date", "sector") and pd.api.types.is_numeric_dtype(df[c])]
    df = df[["symbol", "date", "sector"] + num]
    OUT.mkdir(exist_ok=True)
    df.to_parquet(path)
    return df


def price_features(d: pd.DataFrame) -> pd.DataFrame:
    c, h, l = d["Close"], d["High"], d["Low"]
    r = np.log(c).diff()
    out = pd.DataFrame(index=d.index)
    for n in (1, 5, 20, 60):
        out[f"ret{n}"] = (c / c.shift(n) - 1) * 100
    out["vol20"] = r.rolling(20).std() * 100
    for n in (20, 60):
        out[f"to_hi{n}"] = (c / h.rolling(n).max() - 1) * 100
        out[f"to_lo{n}"] = (c / l.rolling(n).min() - 1) * 100
    out["sma20_gap"] = (c / c.rolling(20).mean() - 1) * 100
    out["sma50_gap"] = (c / c.rolling(50).mean() - 1) * 100
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean() / c * 100
    out["vol_shock"] = d["Volume"] / d["Volume"].rolling(20).mean()
    return out


def build():
    acts = grammar.daily_actions()
    A = {k: acts[k].values for k in ("side_code", "entry_k", "exit_code", "y", "z", "s", "horizon_code")}
    store = store_features()
    syms = sorted(p.name.split(".")[0] for p in ep.D.glob("*.daily.parquet") if not p.name.startswith("SPY."))
    hz_ends = np.array([0, 4, 9, 19], np.int64)
    rows, dec = [], []
    for n, sym in enumerate(syms, 1):
        d = pd.read_parquet(ep.D / f"{sym}.daily.parquet").dropna(subset=["Open", "High", "Low", "Close"])
        if len(d) < 300:
            continue
        pf = price_features(d)
        dates = d.index.strftime("%Y-%m-%d")
        off = zlib.crc32(sym.encode()) % EVERY
        o, h, l, c = d["Open"].values, d["High"].values, d["Low"].values, d["Close"].values
        for t in range(60 + off, len(d) - 20, EVERY):
            if dates[t] < START:
                continue
            sigma = pf["vol20"].iloc[t]
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            g, f, cd = simulator.simulate(o[t + 1:t + 21], h[t + 1:t + 21], l[t + 1:t + 21], c[t + 1:t + 21], float(c[t]), float(o[t + 1]),
                                          float(sigma), 0, hz_ends, A["side_code"], A["entry_k"], A["exit_code"], A["y"], A["z"], A["s"],
                                          A["horizon_code"], simulator.REGULAR_COST)
            eid = f"{sym}|{dates[t]}"
            rows.append(pd.DataFrame({"eid": eid, "action_id": acts["action_id"].values, "gain": g, "filled": f, "code": cd}))
            dec.append({"eid": eid, "symbol": sym, "date": dates[t], "p0": float(c[t]), **pf.iloc[t].to_dict()})
        ep.progress("rl-daily", n, len(syms), f"{sym} decisions={len(dec)}")
    decisions = pd.DataFrame(dec).merge(store, on=["symbol", "date"], how="left")
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT / "daily_outcomes.parquet")
    decisions.to_parquet(OUT / "daily_decisions.parquet")
    acts.to_parquet(OUT / "daily_actions.parquet")
    print(f"DONE rl-daily: {len(decisions)} decisions x {len(acts)} actions = {len(out):,} outcomes; "
          f"{decisions.shape[1]} decision columns", flush=True)


if __name__ == "__main__":
    build()
