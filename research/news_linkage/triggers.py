"""N.6 step 3 (2026-11-14): when a storyline is hot enough for Claude to judge.

Checked every 30 minutes (after quant_intraday_news) over the storylines of storylines.py. Only articles
marked `progress` count: a repeat is the same story again, filler says nothing new (noise audit of
2026-11-14). A storyline is hot at time t when, over the 24 hours before t,
  burst       it has at least MIN_PROGRESS progress articles from at least MIN_SITES different sites, and at
              least RATIO times its own daily average of the 7 days before. A storyline that runs every day
              (an earnings-season bucket with 2-9 articles a day) is measured against itself and stays quiet;
              a storyline opened in the last day has no history, so the first two conditions alone decide.
  leadership  at least LEAD_PROGRESS progress articles from LEAD_SITES sites name a key person of the company
              together with a leadership event (key_people: resigns, affair, probe, joins a rival...).
After its first firing a storyline fires again only when the story escalates (at least ESCALATE times, and ESCALATE_MIN
articles more than, the 24-hour progress count of its last firing, the Huang affair going from a report to a board probe to a
resignation), at least LINE_GAP after that firing, or when it is hot again more than QUIET_DAYS after
it; otherwise a storyline that stays hot fires at every check. One symbol is judged at most once per
SYMBOL_GAP, and the storylines hot at that check are judged together (the Huang affair ran as five
storylines at once); a storyline that turns hot in between waits for the next judgment of its symbol.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import storylines as sl
from storylines import key_people

MIN_PROGRESS, MIN_SITES, RATIO, BASE_DAYS = 5, 3, 3.0, 7
LEAD_PROGRESS, LEAD_SITES = 2, 2
ESCALATE, ESCALATE_MIN, QUIET_DAYS = 2.0, 5, 3
LINE_GAP, SYMBOL_GAP = timedelta(hours=6), timedelta(hours=2)


@dataclass
class Hot:
    storyline: str
    reason: str            # burst | leadership | new
    n24: int               # progress articles in the last 24 hours
    sites24: int
    base: float            # progress per day over the BASE_DAYS before


@dataclass
class Trigger:
    at: datetime
    symbol: str
    hot: list[Hot] = field(default_factory=list)


def _leadership(a: sl.Article) -> bool:
    return bool(a.person) and bool(key_people.EVENT.search(a.title))


def hot_at(line: sl.Storyline, t: datetime) -> Hot | None:
    """Whether `line` is hot at `t`, using only articles collected before `t`."""
    day = timedelta(days=1)
    prog = [a for a in line.articles if a.mark == "progress" and a.at < t]
    last = [a for a in prog if a.at >= t - day]
    if not last:
        return None
    sites = len({a.domain for a in last})
    lead = [a for a in last if _leadership(a)]
    if len(lead) >= LEAD_PROGRESS and len({a.domain for a in lead}) >= LEAD_SITES:
        return Hot(line.id, "leadership", len(last), sites, 0.0)
    if len(last) < MIN_PROGRESS or sites < MIN_SITES:
        return None
    since = max(t - day - timedelta(days=BASE_DAYS), line.first_at)
    span = (t - day - since).total_seconds() / 86400
    if span < 1:
        return Hot(line.id, "new", len(last), sites, 0.0)
    base = sum(1 for a in prog if since <= a.at < t - day) / span
    if len(last) >= RATIO * max(base, 1.0):
        return Hot(line.id, "burst", len(last), sites, round(base, 1))
    return None


def fire(lines: list[sl.Storyline], t: datetime,
         prev_of: Callable[[sl.Storyline], tuple[datetime, int] | None]) -> list[Hot]:
    """The storylines of one symbol that fire at check `t`. `prev_of(line)` gives the (time, n24) of the
    line's last firing, or None: the replay keeps them in a dict, the live job reads them back from its
    stored judgments. The caller applies SYMBOL_GAP."""
    hot = []
    for l in lines:
        if l.first_at >= t or l.last_at < t - timedelta(days=1):
            continue
        h = hot_at(l, t)
        if not h:
            continue
        prev = prev_of(l)
        escalated = bool(prev) and h.n24 >= ESCALATE * prev[1] and h.n24 - prev[1] >= ESCALATE_MIN
        if prev and not ((escalated and t - prev[0] >= LINE_GAP) or t - prev[0] > timedelta(days=QUIET_DAYS)):
            continue
        if escalated:
            h.reason += "+escalation"
        hot.append(h)
    return hot


def replay(lines: list[sl.Storyline], start: datetime, end: datetime, step: timedelta = timedelta(minutes=30)) -> list[Trigger]:
    """Every check from `start` to `end`: the triggers the live job would have raised."""
    last: dict[str, tuple[datetime, int]] = {}      # storyline -> (time, n24) of its last firing
    judged: dict[str, datetime] = {}                 # symbol -> its last judgment
    out = []
    by_symbol = defaultdict(list)
    for l in lines:
        by_symbol[l.symbol].append(l)
    t = start
    while t <= end:
        for sym, ls in by_symbol.items():
            if sym in judged and t - judged[sym] < SYMBOL_GAP:
                continue
            hot = fire(ls, t, lambda l: last.get(l.id))
            for h in hot:
                last[h.storyline] = (t, h.n24)
            if hot:
                out.append(Trigger(t, sym, hot))
                judged[sym] = t
        t += step
    return out
