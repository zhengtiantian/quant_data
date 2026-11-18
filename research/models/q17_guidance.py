"""Round 4's new 2a hypothesis: does the guidance in the earnings release (earnings_releases.py's
release_features.jsonl, read by a local LLM from the 8-K) say more about the next 20 / 60 sessions than
the headline rule? (PROJECT_PLAN Q.17 round 4 addendum, 2026-11-18)

Each release is placed on its reaction session (`date` in the file; the SEC acceptance time decides it,
an after-the-close release trades the next session). Kinds, set before looking:
  full-year outlook lowered / raised / maintained / withdrawn
  revenue below / above the company's own prior guidance (units normalised: some rows are in $bn, some
  in $m)
  next-quarter guidance against this quarter, compared with the same ratio four releases earlier (the
  same fiscal quarter a year before, so seasonality cancels): bottom / top fifth, cut points from the
  in-sample years
  tone -1 / +1
Returns minus SPY (and the 2b sector residual), from the reaction session's open and from the next
open (the drift after the market's first answer, what matters for adding after a report), against all
reaction sessions of the same period; Newey-West t on per-date means (lag = horizon).
In sample 2016-2023, out of sample 2024-2026. Accept: out-of-sample |t| >= 2.5 with the in-sample sign.

Usage: .venv311/bin/python research/models/q17_guidance.py
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

STEP = qa.Q17_DIR / "guidance"
FEATS = Path("/Volumes/Data4T/earnings_policy/release_features.jsonl")
IN_SAMPLE, OUT_SAMPLE = (2016, 2023), (2024, 2026)
HORIZONS = (20, 60)


def _ratio(a, b):
    """a / b with the $bn / $m mix undone (a ratio far from 1 by ~1000x is a unit slip)."""
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return np.nan
    r = a / b
    while r > 300:
        r /= 1000
    while r < 1 / 300:
        r *= 1000
    return r


def _nw(x: np.ndarray, lag: int) -> dict:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 15:
        return {"n_dates": n}
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return {"mean": round(float(x.mean()), 3), "t_nw": round(float(x.mean() / math.sqrt(s / n)), 2), "n_dates": n}


def main() -> dict:
    import news_events as ne
    STEP.mkdir(parents=True, exist_ok=True)
    f = pd.DataFrame([json.loads(l) for l in FEATS.read_text().splitlines() if l.strip()])
    f["session"] = pd.to_datetime(f["date"])
    f = f.sort_values(["symbol", "session"]).reset_index(drop=True)
    mid = lambda lo, hi: np.where(pd.notna(lo) & pd.notna(hi), (pd.to_numeric(lo, errors="coerce") + pd.to_numeric(hi, errors="coerce")) / 2,
                                  pd.to_numeric(lo, errors="coerce").fillna(pd.to_numeric(hi, errors="coerce")))
    f["prior_mid"] = mid(f.prior_guide_low, f.prior_guide_high)
    f["next_mid"] = mid(f.next_guide_low, f.next_guide_high)
    rev = pd.to_numeric(f.revenue, errors="coerce")
    f["rev_vs_prior"] = [_ratio(a, b) for a, b in zip(rev, f.prior_mid)]
    f["next_vs_rev"] = [_ratio(a, b) for a, b in zip(f.next_mid, rev)]
    f["next_vs_rev_4"] = f.groupby("symbol").next_vs_rev.shift(4)
    f["guide_seq_change"] = f.next_vs_rev - f.next_vs_rev_4
    f["year"] = f.session.dt.year
    ins = f[f.year.between(*IN_SAMPLE)]
    lo_cut, hi_cut = ins.guide_seq_change.quantile(0.2), ins.guide_seq_change.quantile(0.8)
    fy = f.full_year_outlook.fillna("none")
    kinds = {
        "full-year outlook lowered": fy == "lowered", "full-year outlook raised": fy == "raised",
        "full-year outlook maintained": fy == "maintained", "full-year outlook withdrawn": fy == "withdrawn",
        "revenue below own prior guidance": f.rev_vs_prior < 1.0, "revenue above own prior guidance": f.rev_vs_prior > 1.0,
        "next-quarter guide, bottom fifth vs a year ago": f.guide_seq_change <= lo_cut,
        "next-quarter guide, top fifth vs a year ago": f.guide_seq_change >= hi_cut,
        "tone -1": pd.to_numeric(f.tone, errors="coerce") == -1, "tone +1": pd.to_numeric(f.tone, errors="coerce") == 1,
    }
    fw = {"spy": ne._forward(), "sector": qa._forward_resid()}
    rep = {"releases": int(len(f)), "full_year_outlook": fy.value_counts().to_dict(),
           "with_seq_change": int(f.guide_seq_change.notna().sum()), "seq_cuts": [round(float(lo_cut), 4), round(float(hi_cut), 4)], "results": {}}
    for tname, fr in fw.items():
        x = f.merge(fr, on=["symbol", "session"], how="left")
        for name, m in kinds.items():
            m = m.reindex(x.index).fillna(False).astype(bool) if len(m) == len(x) else m
            out = {}
            for start, pref in (("reaction open", "x"), ("next open", "n")):
                for per, (a, b) in {"in": IN_SAMPLE, "out": OUT_SAMPLE}.items():
                    w = x.year.between(a, b)
                    for h in HORIZONS:
                        col = f"{pref}{h}"
                        base = x.loc[w, col].mean()
                        sub = x[w & m.values]
                        g = (sub[col] - base).groupby(sub.session).mean().sort_index()
                        out[f"{start} {per} {h}d"] = {"events": int(len(sub)), **_nw(g.values, h)}
            ok = {}
            for start in ("reaction open", "next open"):
                for h in HORIZONS:
                    i, o = out[f"{start} in {h}d"], out[f"{start} out {h}d"]
                    ok[f"{start} {h}d"] = bool("t_nw" in o and abs(o["t_nw"]) >= 2.5 and "mean" in i and np.sign(o["mean"]) == np.sign(i["mean"]))
            rep["results"].setdefault(name, {})[tname] = {**out, "passes": ok}
    (STEP / "study.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("releases", "full_year_outlook", "with_seq_change", "seq_cuts")}, default=str))
    print("excess vs all reaction sessions, mean % (Newey-West t, events)")
    for name, v in rep["results"].items():
        for tname, r in v.items():
            cells = " | ".join(f"{k.replace(' open', '')}: {r[k].get('mean', '-')} ({r[k].get('t_nw', '-')}, {r[k]['events']})"
                               for k in r if k != "passes" and k.endswith("20d") or k.endswith("60d") and k != "passes")
            print(f"  {name:46s} {tname:6s} {cells}  pass {[k for k, v in r['passes'].items() if v]}")
    return rep


if __name__ == "__main__":
    main()
