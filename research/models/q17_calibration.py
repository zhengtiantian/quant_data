"""Round 4's calibration fix for the 5-session barrier model (PROJECT_PLAN Q.17 round 4 addendum, 2026-11-18).

Live, barrier_model.situations() fits two 1-D isotonic curves (lower, upper) on every walk-forward
prediction and live() sets P(none) = max(0, 1 - lower - upper): the calibrator saw the years it is judged
on, and the three probabilities need not sum to one. Here each method is fitted, for every test year,
only on the walk-forward predictions of the years before it (minus the last 7 days, whose 5-session
labels run into the test year) and judged on that year, 2021-2026:

  raw          the model's probabilities
  isotonic     the live method (two 1-D curves, none = the rest, clipped at 0)
  temperature  p_c^(1/T) renormalised, one parameter
  dirichlet    softmax(W log p + b), off-diagonal and intercept L2-penalised (ODIR, Kull et al. 2019)

Scored by multiclass Brier, log loss, per-class 10-bin ECE and how often the probabilities leave the
simplex; then the winner is fitted on every year for live use -> Q17_DIR/calibration/calibrator.json.

Usage: .venv311/bin/python research/models/q17_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q17_ablation as qa  # noqa: E402

STEP = qa.Q17_DIR / "calibration"
WF = Path("/Volumes/Data4T/daily_advice/barrier_wf_5.parquet")
CLS = ("lower", "none", "upper")
EPS = 1e-6
ODIR = 1e-3


def _logp(P):
    return np.log(np.clip(P, EPS, 1))


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(P, y):
    L = _logp(P)
    Y = np.eye(3)[y]

    def nll(t):
        Q = _softmax(L / t[0])
        return -np.mean(np.sum(Y * np.log(np.clip(Q, EPS, 1)), axis=1))
    return {"T": float(minimize(nll, [1.0], bounds=[(0.05, 20)]).x[0])}


def apply_temperature(c, P):
    return _softmax(_logp(P) / c["T"])


def fit_dirichlet(P, y, lam=ODIR):
    L = _logp(P)
    Y = np.eye(3)[y]
    off = ~np.eye(3, dtype=bool)

    def loss(theta):
        W, b = theta[:9].reshape(3, 3), theta[9:]
        Q = _softmax(L @ W.T + b)
        return -np.mean(np.sum(Y * np.log(np.clip(Q, EPS, 1)), axis=1)) + lam * (np.sum(W[off] ** 2) + np.sum(b ** 2))
    x0 = np.r_[np.eye(3).ravel(), np.zeros(3)]
    th = minimize(loss, x0, method="L-BFGS-B").x
    return {"W": th[:9].reshape(3, 3).round(6).tolist(), "b": th[9:].round(6).tolist(), "lambda": lam}


def apply_dirichlet(c, P):
    return _softmax(_logp(P) @ np.array(c["W"]).T + np.array(c["b"]))


def fit_isotonic(P, y):
    from sklearn.isotonic import IsotonicRegression
    grid = np.linspace(0, 1, 101)
    return {c: IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(P[:, i], (y == i).astype(float)).predict(grid).tolist()
            for i, c in ((0, "lower"), (2, "upper"))}


def apply_isotonic(c, P):
    lo = np.array(c["lower"])[np.rint(P[:, 0] * 100).astype(int)]
    hi = np.array(c["upper"])[np.rint(P[:, 2] * 100).astype(int)]
    return np.c_[lo, np.maximum(0.0, 1 - lo - hi), hi]                    # as barrier_model.live() does


METHODS = {"temperature": (fit_temperature, apply_temperature), "dirichlet": (fit_dirichlet, apply_dirichlet),
           "isotonic": (fit_isotonic, apply_isotonic)}


def _ece(y, p, bins=10):
    q = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(abs(y[q == b].mean() - p[q == b].mean()) * (q == b).mean() for b in range(bins) if (q == b).any()))


def _metrics(P, y):
    Y = np.eye(3)[y]
    Pn = P / P.sum(axis=1, keepdims=True)                                  # log loss needs a distribution
    return {"brier": round(float(np.mean(np.sum((P - Y) ** 2, axis=1))), 5),
            "logloss": round(float(-np.mean(np.log(np.clip(Pn[np.arange(len(y)), y], EPS, 1)))), 5),
            "ece_lower": round(_ece((y == 0).astype(float), P[:, 0]), 4), "ece_upper": round(_ece((y == 2).astype(float), P[:, 2]), 4),
            "ece_none": round(_ece((y == 1).astype(float), P[:, 1]), 4),
            "off_simplex_pct": round(float((np.abs(P.sum(axis=1) - 1) > 1e-6).mean() * 100), 2)}


def main() -> dict:
    STEP.mkdir(parents=True, exist_ok=True)
    wf = pd.read_parquet(WF).dropna(subset=["b5"])
    wf["date"] = pd.to_datetime(wf.date)
    P_all, y_all = wf[[f"p_{c}" for c in CLS]].values, wf.b5.astype(int).values
    preds = {m: np.full_like(P_all, np.nan) for m in ["raw", *METHODS]}
    preds["raw"] = P_all.copy()
    test = np.zeros(len(wf), dtype=bool)
    for yr in range(2021, int(wf.year.max()) + 1):
        cal = (wf.date < pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=7)).values
        te = (wf.year == yr).values
        test |= te
        for m, (fit, apply) in METHODS.items():
            c = fit(P_all[cal], y_all[cal])
            preds[m][te] = apply(c, P_all[te])
        print(f"  {yr}: calibrated on {int(cal.sum()):,}, tested on {int(te.sum()):,}", flush=True)
    rep = {"test_days": int(test.sum()), "nested_2021_2026": {m: _metrics(p[test], y_all[test]) for m, p in preds.items()}}
    rep["by_year_logloss"] = {m: {int(yr): _metrics(p[test & (wf.year == yr).values], y_all[test & (wf.year == yr).values])["logloss"]
                                  for yr in range(2021, int(wf.year.max()) + 1)} for m, p in preds.items()}
    best = min(METHODS, key=lambda m: rep["nested_2021_2026"][m]["logloss"])
    rep["chosen"] = best
    final = METHODS[best][0](P_all, y_all)
    (STEP / "calibrator.json").write_text(json.dumps({"method": best, **final, "fitted_on": int(len(wf)),
                                                       "source": str(WF)}, indent=1))
    (STEP / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    main()
