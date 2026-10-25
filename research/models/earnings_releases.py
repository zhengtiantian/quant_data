#!/usr/bin/env python3
"""What each earnings release said, read by a local LLM: features for the earnings-day policy (Q.5 v3).

The policy learned from prices, the EPS surprise and news sentiment found no edge at the open
(earnings_policy.py, 2026-10-25): what moves a stock after the open is in the release itself,
the outlook above all. This reads it.

    python research/models/earnings_releases.py fetch     # EDGAR: the release behind each report
    tools/lm_session.py run --model qwen3.5-9b-mlx --workers-env LM_WORKERS -- \
        python research/models/earnings_releases.py extract

fetch: for each report in earnings_policy's events.parquet, the issuer's press release from EDGAR:
an 8-K with item 2.02 (a 6-K for foreign issuers such as TSM), accepted by the SEC before the open
of the reaction day (16:00-16:10 for an after-the-close report), so nothing from after the open is
read. Kept: the opening of the release (headline, quarter, quotes) and its outlook section.
extract: the LLM reads this release next to the previous one's outlook and returns JSON: revenue of
the quarter, the outlook it gave for it last time, the outlook for the next quarter, whether the
full-year outlook was raised / kept / lowered, margin direction, management tone, capital return,
restructuring. Numbers are turned into features in code (revenue against last outlook, next-quarter
outlook against this quarter).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
EP_DIR = Path(os.getenv("EP_DIR", "/Volumes/Data4T/earnings_policy"))
RELEASES = EP_DIR / "releases.jsonl"
EXTRACTED = EP_DIR / "release_features.jsonl"
API_URL = os.getenv("SLM_API_URL", "http://127.0.0.1:1234/v1").rstrip("/")
MODEL = os.getenv("RELEASE_MODEL", "qwen3.5-9b-mlx")
HEAD_CHARS, OUTLOOK_CHARS, PREV_CHARS = 5000, 3500, 2000      # stored
PROMPT_HEAD, PROMPT_OUTLOOK = 2500, 2200                        # sent: prefill dominates the LLM's time
EDGAR_TICKER = {"BRK.B": "BRK-B"}


def progress(stage, done, total, extra=""):
    print(f"PROGRESS {stage} {done} {total} {extra}".rstrip(), flush=True)


def _clean(t: str) -> str:
    return re.sub(r"[ \t ]+", " ", re.sub(r"\n\s*\n+", "\n", t or "")).strip()


GUIDE_WORDS = re.compile(r"plus or minus|approximately \$|in the range of|between \$|to be (?:about |approximately )?\$|"
                         r"expects?(?: [a-z]+){0,4} (?:revenue|sales)|(?:revenue|sales) (?:is|are) expected|outlook for|guidance for|"
                         r"expected to be (?:as follows|approximately|in the range|between)|as follows:",
                         re.I)


def outlook_section(text: str, n: int) -> str:
    """The release's outlook / guidance part. Candidates are "Outlook" / "Guidance" headings and
    sentences that state expected revenue; each is scored on the guidance language that follows
    ("plus or minus", "approximately $", "expects revenue"). The forward-looking statements
    disclaimer ("... update any outlook statements") and quotes full of results numbers lose."""
    best, best_score = "", 0
    cands = [(m.start(), 4) for m in re.finditer(r"\b(?:[A-Z][a-z]+ |Q\d |Fiscal \S+ )?(?:Outlook|OUTLOOK|Guidance|GUIDANCE)\b", text)]
    cands += [(m.start(), 0) for m in GUIDE_WORDS.finditer(text)]
    for pos, bonus in cands:
        w = text[pos:pos + 700]
        if re.search(r"undertakes? no|obligation to (?:update|revise)|safe harbor|securities litigation reform",
                     text[max(0, pos - 250):pos + 250], re.I):
            continue                                     # the disclaimer, not the outlook
        score = 3 * len(GUIDE_WORDS.findall(w)) + min(4, len(re.findall(r"\$\s?\d", w))) + bonus
        if not GUIDE_WORDS.search(w[:350]):
            score -= 6                                   # a heading or sentence with no guidance after it
        if score > best_score:
            best, best_score = text[max(0, pos - 100):max(0, pos - 100) + n], score
    return best


# ---------------------------------------------------------------- fetch

def fetch():
    import edgar
    edgar.set_identity(os.getenv("SEC_IDENTITY", "quant-research xiz00188@gmail.com"))
    ev = pd.read_parquet(EP_DIR / "events.parquet")
    done = set()
    if RELEASES.exists():
        done = {(r["symbol"], r["date"]) for r in map(json.loads, RELEASES.open())}
    syms = sorted(ev["symbol"].unique())
    for n, sym in enumerate(syms, 1):
        todo = ev[(ev["symbol"] == sym) & ~ev.apply(lambda r: (r["symbol"], r["date"]) in done, axis=1)]
        if todo.empty:
            progress("fetch", n, len(syms), f"{sym} done before")
            continue
        found = 0
        try:
            filings = edgar.Company(EDGAR_TICKER.get(sym, sym)).get_filings(form=["8-K", "6-K"]).filter(date="2015-12-01:")
            by_day = {}
            for f in filings:
                by_day.setdefault(str(f.filing_date), []).append(f)
            for r in todo.itertuples():
                rd = datetime.fromisoformat(r.date)
                open_ts = rd.replace(hour=9, minute=30)
                best = None
                for k in range(0, 5):
                    for f in by_day.get((rd - timedelta(days=k)).strftime("%Y-%m-%d"), []):
                        if f.form == "8-K" and "2.02" not in str(getattr(f, "items", "") or ""):
                            continue
                        acc = f.header.acceptance_datetime
                        acc = acc.replace(tzinfo=None) if acc else None
                        if acc is None or acc >= open_ts or acc < open_ts - timedelta(days=5):
                            continue
                        text = ""
                        for a in f.attachments:
                            dt, doc = str(a.document_type or ""), str(a.document or "").lower()
                            if "99" in dt or "ex99" in doc or (f.form == "6-K" and doc.endswith((".htm", ".html"))):
                                try:
                                    t = _clean(a.text())
                                except Exception:  # noqa: BLE001
                                    continue
                                if len(t) > len(text) and re.search(r"quarter|results", t[:3000], re.I):
                                    text = t
                        if text and (best is None or acc > best[0]):
                            best = (acc, f.form, text)
                    if best:
                        break
                row = {"symbol": sym, "date": r.date, "found": bool(best)}
                if best:
                    acc, form, text = best
                    (EP_DIR / "releases").mkdir(exist_ok=True)
                    (EP_DIR / "releases" / f"{sym}_{r.date}.txt").write_text(text)
                    row.update(accepted=acc.isoformat(), form=form, chars=len(text), head=text[:HEAD_CHARS],
                               outlook=outlook_section(text, OUTLOOK_CHARS))
                    found += 1
                with RELEASES.open("a") as fo:
                    fo.write(json.dumps(row) + "\n")
        except Exception as ex:  # noqa: BLE001 - one issuer failing must not stop the rest
            progress("fetch", n, len(syms), f"{sym} failed {type(ex).__name__}: {str(ex)[:80]}")
            continue
        progress("fetch", n, len(syms), f"{sym} {found}/{len(todo)}")


# ---------------------------------------------------------------- extract

PROMPT = """You read a company's quarterly earnings press release. Return one JSON object, nothing else.

{{"revenue": <revenue of the quarter just reported, in millions of the reporting currency, or null>,
 "prior_guide_low": <low end of the revenue outlook the PREVIOUS release gave for this quarter, millions, or null>,
 "prior_guide_high": <high end, or null>,
 "next_guide_low": <low end of the revenue outlook for the next quarter given in THIS release, millions, or null>,
 "next_guide_high": <high end, or null>,
 "full_year_outlook": "raised" | "maintained" | "lowered" | "initiated" | "withdrawn" | "none",
 "margin_outlook": "up" | "flat" | "down" | "none",
 "tone": <management tone from -2 (alarmed) to 2 (very confident)>,
 "capital_return": "new_or_raised" | "cut_or_suspended" | "none",
 "restructuring": true | false}}

A single-number outlook: use it for both low and high. "approximately $X" counts as X. Use null when
the release does not say; do not guess. margin_outlook compares the next quarter's gross (else
operating) margin outlook with the margin of the quarter just reported.

PREVIOUS RELEASE, OUTLOOK SECTION:
{prev}

THIS RELEASE ({symbol}, reported {date}), OPENING:
{head}

THIS RELEASE, OUTLOOK SECTION:
{outlook}
"""


def ask(prompt):
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": 220, "stream": False,
               "messages": [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": "<think>\n\n</think>\n{"}]}   # no reasoning
    for attempt in range(4):
        try:
            r = requests.post(f"{API_URL}/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            text = "{" + r.json()["choices"][0]["message"]["content"].split("</think>")[-1].lstrip().lstrip("{")
            return json.loads(text[:text.rindex("}") + 1])
        except (requests.RequestException, ValueError, KeyError) as ex:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def extract():
    rel = [r for r in map(json.loads, RELEASES.open()) if r.get("found")]
    by_sym = {}
    for r in sorted(rel, key=lambda r: (r["symbol"], r["date"])):
        by_sym.setdefault(r["symbol"], []).append(r)
    done = set()
    if EXTRACTED.exists():
        done = {(r["symbol"], r["date"]) for r in map(json.loads, EXTRACTED.open())}
    jobs = []
    for sym, rs in by_sym.items():
        for i, r in enumerate(rs):
            if (sym, r["date"]) not in done:
                prev = rs[i - 1]["outlook"][:1200] if i else "(not available)"
                jobs.append((r, PROMPT.format(prev=prev or "(none found)", symbol=sym, date=r["date"], head=r["head"][:PROMPT_HEAD],
                                              outlook=r["outlook"][:PROMPT_OUTLOOK] or "(no outlook section found)")))
    only = {x.strip().upper() for x in os.getenv("EXTRACT_SYMBOLS", "").split(",") if x.strip()}
    if only:
        jobs = [j for j in jobs if j[0]["symbol"] in only]
    if os.getenv("EXTRACT_LIMIT"):
        jobs = jobs[:int(os.environ["EXTRACT_LIMIT"])]
    workers = int(os.getenv("LM_WORKERS", "4"))
    progress("extract", 0, len(jobs), f"workers={workers} model={MODEL}")
    ok = bad = 0
    with ThreadPoolExecutor(workers) as ex, EXTRACTED.open("a") as fo:
        futs = {ex.submit(ask, p): r for r, p in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                out = {"symbol": r["symbol"], "date": r["date"], **f.result()}
                ok += 1
            except Exception as e:  # noqa: BLE001
                out = {"symbol": r["symbol"], "date": r["date"], "error": f"{type(e).__name__}: {e}"[:200]}
                bad += 1
            fo.write(json.dumps(out) + "\n")
            fo.flush()
            if i % 10 == 0 or i == len(jobs):
                progress("extract", i, len(jobs), f"ok={ok} failed={bad}")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "fetch":
        fetch()
    elif stage == "extract":
        extract()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
