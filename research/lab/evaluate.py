"""One evaluation for every model of the earnings-pulse lab (PROJECT_PLAN Q.6).

A model hands in its out-of-sample step predictions, one row per (report, placement, level,
buy-back) with the realised gain: columns eid, symbol, year, place, level, buyback, gain, pred,
q10 (what research/models/earnings_pulse.py bakeoff writes). A policy turns them into decisions
exactly as the live watch does (earnings_pulse.compose / run_policy: in time order, the first plan
that clears the margin). The verdict:

  - per year 2019-2026, by session (after the close / before the open), per stock
  - selection on 2019-2022, the report on 2023-2026 (a variant is chosen on the first only)
  - the deflated Sharpe ratio (Bailey and Lopez de Prado, 2014): the probability that the per-trade
    Sharpe is above zero once the number of variants tried, and their spread, are accounted for
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import earnings_pulse as ep  # noqa: E402

SELECT_UNTIL = 2022
RISK_LIMITS = {"worst": -10.0, "p10": -2.0, "min_acted": 20}
GRID = {"margin": (0.3, 0.4, 0.5, 0.7), "no_pre": (False, True), "q10_min": (None, -3.0, -2.0, -1.5, -1.0),
        "weight": ("pred", "risk")}


def load_preds(name: str) -> pd.DataFrame:
    """A predictions folder under EP_DIR (bakeoff_preds, cnn_preds) or under EP_DIR/lab_preds (the zoo)."""
    d = ep.D / name if (ep.D / name).is_dir() else ep.D / "lab_preds" / name
    return pd.concat(pd.read_parquet(p) for p in sorted(d.glob("*.parquet")))


def sessions() -> Dict[str, float]:
    pe = pd.read_parquet(ep.D / "pulse_events.parquet")
    return {f"{r.symbol}|{r.date}": r.bmo for r in pe.itertuples()}


def apply_policy(preds: pd.DataFrame, margin: float, no_pre: bool, q10_min: Optional[float], weight: str) -> pd.DataFrame:
    """Per report: acted?, the realised gain of the plan (0 when not acting), placement, steps."""
    k = preds
    if no_pre:
        k = k[k["place"] != "pre"]
    if q10_min is not None:
        k = k[k["q10"] >= q10_min]
    out = []
    for y, g in preds.groupby("year"):
        out.append(ep.run_policy(ep.compose(k[k["year"] == y], margin, weight), g["eid"].unique(), margin).assign(year=y))
    return pd.concat(out, ignore_index=True)


def summary(r: pd.DataFrame) -> Dict[str, float]:
    a = r[r["acted"]]
    if not len(a):
        return {"acted": 0}
    g = a["gain"]
    return {"acted": int(len(a)), "share_acted": len(a) / len(r), "mean": float(g.mean()), "se": float(g.std() / math.sqrt(len(g))),
            "win": float((g > 0).mean()), "p10": float(g.quantile(0.1)), "worst": float(g.min()), "sum": float(g.sum()),
            "sharpe_trade": float(g.mean() / g.std()) if g.std() > 0 else 0.0, "worst_year": float(r.groupby("year")["gain"].mean().min())}


def verdict(r: pd.DataFrame, sess: Dict[str, float]) -> Dict[str, Any]:
    r = r.assign(bmo=r["eid"].map(sess), symbol=r["eid"].str.split("|").str[0])
    out = {"all": summary(r), "select": summary(r[r["year"] <= SELECT_UNTIL]), "report": summary(r[r["year"] > SELECT_UNTIL]),
           "amc": summary(r[r["bmo"] == 0]), "bmo": summary(r[r["bmo"] == 1])}
    out["years"] = {int(y): float(g["gain"].mean()) for y, g in r.groupby("year")}
    out["stocks"] = {s: summary(g) for s, g in r.groupby("symbol") if g["acted"].any()}
    return out


def passes(sel: Dict[str, float]) -> bool:
    return (sel.get("acted", 0) >= RISK_LIMITS["min_acted"] and sel["worst"] > RISK_LIMITS["worst"]
            and sel["p10"] > RISK_LIMITS["p10"])


def variants(grid: Dict[str, Iterable] = GRID) -> List[Dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def deflated_sharpe(returns: np.ndarray, trial_sharpes: List[float]) -> Dict[str, float]:
    """Probability that the true per-trade Sharpe exceeds zero after selecting the best of
    len(trial_sharpes) trials (their variance sets the bar SR0), with the skew and kurtosis of the
    returns. Bailey & Lopez de Prado, 'The Deflated Sharpe Ratio' (2014)."""
    from scipy.stats import kurtosis, norm, skew
    r = np.asarray(returns, float)
    n, T = max(len(trial_sharpes), 1), len(r)
    sr = r.mean() / r.std(ddof=1)
    v = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    e = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - e) * norm.ppf(1 - 1 / n) + e * norm.ppf(1 - 1 / (n * math.e))) if n > 1 else 0.0
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-9))
    z = (sr - sr0) * math.sqrt(T - 1) / denom
    return {"dsr": float(norm.cdf(z)), "sharpe_trade": float(sr), "sr0": float(sr0), "trials": n, "trades": T}
