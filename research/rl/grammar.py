"""The action grammar (PROJECT_PLAN Q.7, 2026-10-26).

An action is one leg: an entry, an exit and a horizon, with distances in units of the stock's daily
volatility (sigma = vol20, the daily standard deviation of returns in %, so 1 sigma on INTC ~3.5%).
Long only, as the user's broker allows:

  side     reduce  sell shares already held, then buy them back (or not: the time exit)
           add     buy with cash, then sell them (also the template for a stock not held)
  entry    k       0 = at the price now; else a limit k sigma beyond it (above for a sale, below
                   for a purchase), working until the end of the reaction session (d1)
  exit     limit   y sigma back the other way from the entry fill
           trail   a trailing stop z sigma behind the best price since the entry
           bracket a limit y sigma and a stop s sigma, whichever trades first (one cancels the other)
           time    nothing before the horizon
  horizon          whatever is still open closes there: next regular open, the close of the
                   reaction day (d1), or of trading day 3, 5, 10 or 20

Ladders are not actions: they are weighted sets of legs, composed by the policy (Q.5 v4.2).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

SIDES = ("reduce", "add")
ENTRY_K = (0.0, 0.5, 1.0, 2.0)
LIMIT_Y = (0.5, 1.0, 2.0, 3.0)
TRAIL_Z = (0.5, 1.0, 2.0)
BRACKET_YS = ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (3.0, 2.0))
HORIZONS = ("next_open", "d1", "d3", "d5", "d10", "d20")
EXIT_CODES = {"limit": 0, "trail": 1, "bracket": 2, "time": 3}


def exits():
    out = [("limit", y, 0.0, 0.0) for y in LIMIT_Y]
    out += [("trail", 0.0, z, 0.0) for z in TRAIL_Z]
    out += [("bracket", y, 0.0, s) for y, s in BRACKET_YS]
    out += [("time", 0.0, 0.0, 0.0)]
    return out


def actions() -> pd.DataFrame:
    """Every single-leg action, with an id and the numeric parameters the simulator reads."""
    rows = []
    for side, k, (ex, y, z, s), hz in itertools.product(SIDES, ENTRY_K, exits(), HORIZONS):
        rows.append({"side": side, "entry_k": k, "exit": ex, "y": y, "z": z, "s": s, "horizon": hz})
    df = pd.DataFrame(rows)
    df["action_id"] = np.arange(len(df), dtype=np.int32)
    df["side_code"] = (df["side"] == "add").astype(np.int8)
    df["exit_code"] = df["exit"].map(EXIT_CODES).astype(np.int8)
    df["horizon_code"] = df["horizon"].map({h: i for i, h in enumerate(HORIZONS)}).astype(np.int8)
    df["name"] = (df["side"] + " k" + df["entry_k"].map("{:g}".format) + " " + df["exit"]
                  + df.apply(lambda r: {"limit": f" y{r.y:g}", "trail": f" z{r.z:g}", "bracket": f" y{r.y:g}/s{r.s:g}",
                                        "time": ""}[r.exit], axis=1) + " " + df["horizon"])
    return df
