"""The live risk-factor flag (PROJECT_PLAN Q.17.5, 2026-11-18): has a company's latest 10-K / 10-Q
rewritten or grown its risk factors more than 80% of filings do?

Q.17.5 (q17_filings.py) found that the fifth of filings whose Item 1A (risk factors) is least similar
to the same form a year earlier trailed their sector afterwards, in sample (2018-2023) and out of
sample (2024-2026): about -3% over 60 sessions and -6% over 120. The effect came from risk text that
grew or was rewritten, not from risk text that shrank.

For the core names, the watchlist and every stock in the trade log (quant_ai's markets.db, read only):
the latest 10-K / 10-Q from EDGAR (edgartools, the same SEC identity as earnings_releases.py), Item 1A
against the same form 330-400 days earlier, cosine similarity of word counts (letters only), as in the
study. Flagged when the similarity is under the study's in-sample 20th percentile for the form and the
text did not shrink (length >= 0.67 x a year ago). Texts are cached under FILING_DIR/text, so a run
fetches only new filings.

Writes FILING_DIR/latest.json, read by quant_ai (markets/filing_changes.py).
Usage: .venv311/bin/python research/models/filing_changes.py [SYMBOLS...]
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q17_filings as qf  # noqa: E402

FILING_DIR = Path(os.getenv("FILING_DIR", "/Volumes/Data4T/filing_changes"))
EP_DIR = Path(os.getenv("EP_DIR", "/Volumes/Data4T/earnings_policy"))
MARKETS_DB = Path(os.getenv("MARKETS_DB", str(Path.home() / "Library/Application Support/quant_ai/markets.db")))
CUTS = {"10-K": 0.9967, "10-Q": 0.8813}       # q17_filings, in-sample 20th percentile of Item 1A similarity
SHRUNK = 0.67
LOOKBACK_DAYS = 430                           # the latest filing and its year-earlier match


def symbols(extra: list[str]) -> list[str]:
    out = {p.name.split(".")[0] for p in EP_DIR.glob("*.daily.parquet")} - {"SPY"}
    try:
        conn = sqlite3.connect(f"file:{MARKETS_DB}?mode=ro", uri=True, timeout=10)
        out |= {r[0].upper() for r in conn.execute("SELECT symbol FROM watchlist")}
        out |= {r[0].upper() for r in conn.execute("SELECT DISTINCT symbol FROM transactions")}
        conn.close()
    except sqlite3.Error:
        pass                                   # no quant_ai store: the core names only
    return sorted(out | {s.upper() for s in extra})


def _load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def check(sym: str) -> dict | None:
    import edgar
    since = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    try:
        fs = [f for f in edgar.Company(sym).get_filings(form=["10-K", "10-Q"]).filter(date=f"{since}:") if f.form in ("10-K", "10-Q")]
    except Exception:  # noqa: BLE001 - a foreign issuer (20-F / 6-K) or a name EDGAR does not know
        return None
    if not fs:
        return None
    qf.TEXT = FILING_DIR / "text"
    recs = []
    for f in fs:
        p = qf.TEXT / sym / f"{f.accession_no}.json.gz"
        if not p.exists():
            qf._one(f, sym)
        recs.append(_load(p))
    recs.sort(key=lambda r: r["period"])
    last = recs[-1]
    per = datetime.strptime(last["period"], "%Y-%m-%d")
    prev = [r for r in recs[:-1] if r["form"] == last["form"]
            and timedelta(days=330) <= per - datetime.strptime(r["period"], "%Y-%m-%d") <= timedelta(days=400)]
    out = {"symbol": sym, "form": last["form"], "accession": last["accession"], "accepted": last["accepted"],
           "period": last["period"], "filing_date": last["filing_date"],
           "url": f"https://www.sec.gov/Archives/edgar/data/{last['accession'].split('-')[0].lstrip('0')}/"
                  f"{last['accession'].replace('-', '')}/{last['accession']}-index.htm",
           "sim_1a": None, "len_ratio": None, "direction": None, "flag": False, "cut": CUTS[last["form"]]}
    if not prev or not last.get("item1a") or not prev[-1].get("item1a"):
        out["reason"] = "no year-earlier filing with risk factors to compare"
        return out
    a, b = last["item1a"], prev[-1]["item1a"]
    sim = qf._cos(qf._words(a), qf._words(b))
    ratio = len(a) / max(1, len(b))
    out.update(sim_1a=round(float(sim), 4), len_ratio=round(ratio, 2), compared_with=prev[-1]["accession"],
               direction="grown" if ratio > 1.5 else "shrunk" if ratio < SHRUNK else "rewritten")
    out["flag"] = bool(sim < CUTS[last["form"]] and ratio >= SHRUNK)
    return out


def main(argv: list[str]) -> dict:
    import edgar
    edgar.set_identity(os.getenv("SEC_IDENTITY", "quant-research xiz00188@gmail.com"))
    FILING_DIR.mkdir(parents=True, exist_ok=True)
    items, failed = {}, {}
    for s in symbols(argv):
        try:
            r = check(s)
        except Exception as e:  # noqa: BLE001 - one name failing must not stop the rest
            failed[s] = f"{type(e).__name__}: {str(e)[:120]}"
            continue
        if r:
            items[s] = r
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cuts": CUTS,
           "evidence": {"in_60": -1.94, "in_120": -3.50, "out_60": -3.24, "out_120": -6.13, "quarters_in": 24, "quarters_out": 11},
           "items": items, "failed": failed}
    tmp = FILING_DIR / "latest.json.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(FILING_DIR / "latest.json")
    flagged = sorted(s for s, r in items.items() if r["flag"])
    print(f"{len(items)} names checked, {len(flagged)} flagged: {flagged}; failed {list(failed)}")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
