"""Q.17.5: filing changes ("Lazy Prices", Cohen, Malloy & Nguyen 2020), a small test without an LLM
(PROJECT_PLAN Q.17.5, 2026-11-18).

  fetch   every 10-K / 10-Q of the 101 core names accepted from 2017 (foreign issuers file 20-F / 6-K and
          drop out), from EDGAR through edgartools as earnings_releases.py does (same SEC identity):
          the whole primary document as text and, where edgartools finds it, Item 1A (risk factors;
          10-Q: Part II Item 1A). MD&A is left out: edgartools returns it empty for several issuers (INTC).
          One gzipped JSON per filing under Q17_DIR/5/text/, resumable.
  study   each filing against the same form a year earlier (10-K to the previous 10-K, 10-Q to the same
          quarter's 10-Q: periods 330-400 days apart): cosine similarity of word counts (letters only, so
          the changing numbers do not count) for the whole document, and for Item 1A where both have it.
          Returns from the first open after the SEC acceptance time over 60 / 120 sessions: the 2b sector
          residual (round 4's target) and, as a check, minus SPY. Within each calendar quarter the filings are split into
          fifths by similarity; the test is the least-similar fifth's mean minus that quarter's mean,
          Newey-West t over quarters, 2018-2023 in sample and 2024-2026 out of sample.
  Accept: the least-similar fifth trails over 3 / 6 months with t >= 2.5 and the same sign out of sample;
  then an avoid flag and a Q.17.4 feature. Otherwise no factor, and Q.17.1 may show a holding's latest
  filing change as a fact, claiming no edge.

Usage: .venv311/bin/python research/models/q17_filings.py fetch|study
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import sys
from collections import Counter
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q17_ablation as qa  # noqa: E402

STEP = qa.Q17_DIR / "5"
TEXT = STEP / "text"
SINCE = "2017-01-01"
HORIZONS = (60, 120)                     # round 4: 60 / 120 sessions, sector residual first
IN_SAMPLE, OUT_SAMPLE = (2018, 2023), (2024, 2026)
WORD = re.compile(r"[a-z]{2,}")


def _one(f, sym: str) -> str:
    out = TEXT / sym / f"{f.accession_no}.json.gz"
    if out.exists():
        return "cached"
    o = f.obj()
    item1a = None
    try:
        item1a = o["Item 1A"] if f.form == "10-K" else o.get_item_with_part("Part II", "Item 1A")
    except Exception:  # noqa: BLE001 - a section edgartools cannot find is simply missing
        item1a = None
    text = f.text()
    rec = {"symbol": sym, "form": f.form, "accession": f.accession_no, "filing_date": str(f.filing_date),
           "accepted": str(f.acceptance_datetime), "period": str(f.period_of_report), "item1a": item1a or None, "text": text}
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(rec, fh)
    return "ok"


def _symbol(sym: str) -> dict:
    """One name's filings in one worker process (parsing is CPU-bound; threads share one core)."""
    import edgar
    edgar.set_identity(os.getenv("SEC_IDENTITY", "quant-research xiz00188@gmail.com"))
    try:
        fs = edgar.Company(sym).get_filings(form=["10-K", "10-Q"]).filter(date=f"{SINCE}:")
    except Exception as e:  # noqa: BLE001 - a name EDGAR does not know (foreign listing) is skipped
        return {"symbol": sym, "filings": 0, "ok": 0, "errors": [f"{type(e).__name__}"]}
    fs = [f for f in fs if f.form in ("10-K", "10-Q")]             # amendments (10-K/A) left out
    ok, errors = 0, []
    for f in fs:
        try:
            _one(f, sym)
            ok += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{f.accession_no}: {type(e).__name__}: {str(e)[:100]}")
    return {"symbol": sym, "filings": len(fs), "ok": ok, "errors": errors}


def fetch(workers: int = 5) -> None:
    """Five processes: about 3 filings a second, under the SEC's 10 requests a second."""
    from concurrent.futures import ProcessPoolExecutor
    TEXT.mkdir(parents=True, exist_ok=True)
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_symbol, s): s for s in qa._core()}
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            out[r["symbol"]] = r
            print(f"  {i:3d}/{len(futs)} {r['symbol']}: {r['ok']}/{r['filings']}" + (f", errors {len(r['errors'])}" if r["errors"] else ""), flush=True)
    rep = {"filings": sum(r["filings"] for r in out.values()), "ok": sum(r["ok"] for r in out.values()),
           "none_for": sorted(s for s, r in out.items() if not r["filings"]), "per_name": out}
    (STEP / "fetch_report.json").write_text(json.dumps(rep, indent=1))
    print(f"done: {rep['ok']:,} of {rep['filings']:,}; none for {rep['none_for']}")


def _words(t: str | None) -> Counter:
    return Counter(WORD.findall((t or "").lower()))


def _cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return np.nan
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def _place(accepted: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    ts = pd.to_datetime(accepted, utc=True).dt.tz_convert("America/New_York")
    day = ts.dt.tz_localize(None).dt.normalize()
    pre = (ts.dt.hour * 60 + ts.dt.minute) < 9 * 60 + 30
    pos = np.searchsorted(idx.values, day.values.astype("datetime64[ns]"))
    same = (pos < len(idx)) & (idx.values[np.minimum(pos, len(idx) - 1)] == day.values.astype("datetime64[ns]"))
    return np.where(same & ~pre.values, pos + 1, pos)


def _nw(x: np.ndarray, lag: int) -> dict:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return {"n": n}
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return {"mean": round(float(x.mean() * 100), 3), "t_nw": round(float(x.mean() / math.sqrt(s / n)), 2), "n": n}


def study() -> dict:
    import news_events as ne
    recs = []
    for p in TEXT.glob("*/*.json.gz"):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            r = json.load(fh)
        recs.append({k: r[k] for k in ("symbol", "form", "accession", "accepted", "period")} |
                    {"w_text": _words(r["text"]), "w_1a": _words(r.get("item1a")) if r.get("item1a") else None,
                     "len_1a": len(r.get("item1a") or "")})
    df = pd.DataFrame(recs)
    df["period"] = pd.to_datetime(df.period, errors="coerce")
    df = df.dropna(subset=["period"]).sort_values(["symbol", "form", "period"]).reset_index(drop=True)
    sim_t = []
    for (s, f), g in df.groupby(["symbol", "form"], sort=False):
        for i in g.index:
            prev = g[(g.period <= df.at[i, "period"] - pd.Timedelta(days=330)) & (g.period >= df.at[i, "period"] - pd.Timedelta(days=400))]
            if prev.empty:
                sim_t.append((i, np.nan, np.nan, np.nan))
                continue
            j = prev.index[-1]
            sa = _cos(df.at[i, "w_1a"], df.at[j, "w_1a"]) if df.at[i, "w_1a"] and df.at[j, "w_1a"] else np.nan
            sim_t.append((i, _cos(df.at[i, "w_text"], df.at[j, "w_text"]), sa, df.at[j, "len_1a"]))
    s = pd.DataFrame(sim_t, columns=["i", "sim_text", "sim_1a", "len_1a_prev"]).set_index("i")
    df = df.join(s).drop(columns=["w_text", "w_1a"]).dropna(subset=["sim_text"])
    sec = json.loads((qa.STEP2B / "core_sectors.json").read_text())
    spy = ne._sessions("SPY")
    rows = []
    for sym, g in df.groupby("symbol"):
        d = ne._sessions(sym)
        b, _ = qa._bench(sym, d.index, sec)
        beta = qa._beta(d.Close.pct_change(), b.Close.pct_change()).fillna(1.0)
        pos = _place(g.accepted, d.index)
        for (_, r), p in zip(g.iterrows(), pos):
            if p >= len(d):
                continue
            out = {"symbol": sym, "form": r.form, "session": d.index[p], "sim_text": r.sim_text, "sim_1a": r.sim_1a,
                   "len_1a": r.len_1a, "len_1a_prev": r.len_1a_prev}
            for h in HORIZONS:
                e = p + h - 1
                if e >= len(d):
                    out[f"x{h}"] = out[f"s{h}"] = np.nan
                    continue
                rs = d.Close.iloc[e] / d.Open.iloc[p] - 1
                sp = spy.reindex(d.index)
                rm = sp.Close.iloc[e] / sp.Open.iloc[p] - 1
                rb = b.Close.iloc[e] / b.Open.iloc[p] - 1
                out[f"x{h}"] = rs - rm
                out[f"s{h}"] = rs - beta.iloc[p] * rb
            rows.append(out)
    ev = pd.DataFrame(rows)
    ev["quarter"] = ev.session.dt.to_period("Q")
    ev.to_parquet(STEP / "events.parquet")
    rep = {"filings": int(len(ev)), "names": int(ev.symbol.nunique()), "with_item1a": int(ev.sim_1a.notna().sum()),
           "by_form": ev.form.value_counts().to_dict(), "tests": {}}
    for measure in ("sim_text", "sim_1a"):
        e = ev.dropna(subset=[measure]).copy()
        e["fifth"] = e.groupby("quarter")[measure].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) if len(x) >= 10 else np.nan)
        for tgt in ("x", "s"):
            for h in HORIZONS:
                col = f"{tgt}{h}"
                q = e.dropna(subset=[col, "fifth"]).groupby("quarter").apply(
                    lambda g: g.loc[g.fifth == 0, col].mean() - g[col].mean()).dropna()
                yr = np.array([p.year for p in q.index])
                lag = max(1, round(h / 63))
                res = {per: _nw(q.values[(yr >= a) & (yr <= b)], lag) for per, (a, b) in {"in": IN_SAMPLE, "out": OUT_SAMPLE}.items()}
                res["all"] = _nw(q.values, lag)
                i, o = res["in"], res["out"]
                res["passes"] = bool("t_nw" in i and i["t_nw"] <= -2.5 and "mean" in o and o["mean"] < 0) if tgt == "s" else None
                rep["tests"][f"{measure} {'SPY' if tgt == 'x' else 'sector'} {h}d"] = res
    (STEP / "study.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("filings", "names", "with_item1a", "by_form")}, default=str))
    for k, r in rep["tests"].items():
        print(f"  {k:24s} in {r['in']}  out {r['out']}  all {r['all']}  pass {r['passes']}")
    return rep


if __name__ == "__main__":
    {"fetch": fetch, "study": study}[sys.argv[1] if len(sys.argv) > 1 else "study"]()
