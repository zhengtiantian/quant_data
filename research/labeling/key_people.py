"""Key-person rule for relevance (2026-11-13): news about a company's leaders is news about the company.

The GD1 relevance student learnt "is this article materially about the company?" and so rejects
follow-ups about a CEO as a person (the Jensen Huang affair: 426 Nvidia articles in one week). The user's
standard: they count, e.g. Huang returning to Nvidia or joining AMD moves Nvidia. Relevant for that company only, when the title
  - has the company next to a leadership word ("Nvidia CEO", "former Nvidia chief", "CEO of Nvidia"), or
  - names a listed person together with a leadership event (resigns, joins, affair, probe, sells shares...), or
  - has the company together with such an event ("Nvidia spy allegations").
Used only to overturn a "not relevant" answer; a quote, a fortune or a biography does not count.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

TABLE = Path(__file__).with_name("key_people.json")
ROLE = r"(?:CEO|CFO|COO|chief executive|chief|boss|founder|co-founder|cofounder|chairman|chairwoman|chair|president|leadership)"


# a title that only names the person must also say something happened to them as a leader
# (a quote, a fortune, a biography is not news about the company)
EVENT = re.compile(r"\b(resign\w*|step(s|ped|ping)? down|quits?|ous?ted|oust\w*|fired|sacked|depart\w*|leaves|leaving|"
                   r"joins|joined|joining|hired|appointed|returns|returning|rejoin\w*|comeback|successor|succession|retires|retiring|"
                   r"affair|scandal|spy|spying|espionage|honeypot|probe|investigat\w*|arrest\w*|charged|indict\w*|sued|lawsuit|"
                   r"misconduct|allegations?|accused|divorce|romantic|romance|health|illness|hospitali[sz]ed|dies|died|death|"
                   r"sells|sold|selling|dumps|offload\w*|stake)\b", re.I)
# with the company name alone (no person) only a scandal or an ouster counts: "Nvidia spy allegations"
SCANDAL = re.compile(r"\b(?!SPY\b)(resign\w*|step(s|ped|ping)? down|ous?ted|oust\w*|fired|sacked|affair|scandal|spy|spying|espionage|"
                     r"honeypot|misconduct|allegations?|accused|arrest\w*|indict\w*|divorce|romantic|romance)\b", re.I)


@lru_cache(maxsize=None)
def _patterns() -> dict[str, tuple[re.Pattern, re.Pattern, re.Pattern]]:
    out = {}
    for sym, entry in json.loads(TABLE.read_text()).items():
        if sym.startswith("_"):
            continue
        alias = "|".join(re.escape(a) for a in entry["aliases"])
        people = "|".join(re.escape(p) for p in entry["people"])
        role = re.compile(
            rf"(?<![\w-])(?:{alias})(?:['’]s)?\s+(?:(?:former|ex|new|interim|then)[\s-]+)?{ROLE}\b"  # Nvidia('s) (former) CEO
            rf"|\b(?:{ROLE})\s+of\s+(?:{alias})\b"                                   # CEO of Nvidia
            rf"|\bex-(?:{alias})\s+{ROLE}\b")                                         # ex-Nvidia chief
        out[sym] = (role, re.compile(rf"\b(?:{people})\b"), re.compile(rf"(?<![\w-])(?:{alias})(?![\w-])"))
    return out


def hit(symbol: str | None, title: str | None) -> str | None:
    """The matched text when the title is about one of the company's leaders, else None."""
    pats = _patterns().get((symbol or "").upper())
    title = title or ""
    if not pats:
        return None
    role, person, alias = pats
    if m := role.search(title):
        return m.group(0)
    if (m := person.search(title)) and EVENT.search(title):
        return m.group(0)
    if (m := alias.search(title)) and (e := SCANDAL.search(title)):   # "Nvidia spy allegations"
        return f"{m.group(0)} + {e.group(0)}"
    return None
