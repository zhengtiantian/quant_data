"""Exact simulation of grammar actions on one price path (PROJECT_PLAN Q.7, 2026-10-26).

A path is one array of bars (open, high, low, close) from the decision on: 1-minute bars through
the report evening or pre-market, 5-minute bars for the rest of the reaction day and the two
sessions after, then daily bars to day 20. Horizon ends are bar indices into it.

Fills, conservatively (our orders are too small to move the price):
  - a limit trades when the price goes through it by THROUGH (0.05%), at the limit or at the bar's
    open when the bar opens beyond it (a gap), never better than that
  - a stop (bracket stop, trailing stop) trades at its level or at the open when gapped through,
    always the worse of the two
  - a bar that reaches both a bracket's limit and its stop counts as the stop
  - a trailing stop is checked against the best price before the bar, then the best price moves
  - the exit starts on the bar after the entry fill; anything still open closes at the horizon close
  - COST (0.25%) per side covers spread and slippage outside the regular session
Gain = what the leg made per share in % of the price at the decision, net of costs, against not
doing it (for a reduce: against holding the shares; for an add: against keeping the cash).
"""
from __future__ import annotations

import numpy as np
from numba import njit

COST = 0.25
THROUGH = 0.05


@njit(cache=True)
def _leg(o, h, l, c, p0, sigma, side, k, ex, y, z, s, entry_end, hz_end):
    """One action on one path: (gain %, entry filled 0/1, exit code 0 limit 1 trail 2 bracket-limit
    3 bracket-stop 4 horizon). sigma in % of p0."""
    n = len(o)
    th = THROUGH / 100.0
    # ---- entry
    if k == 0.0:
        fill, i0 = p0, -1                                  # at the price now; the exit starts on bar 0
    else:
        fill, i0 = 0.0, -2
        if side == 0:
            lim = p0 * (1.0 + k * sigma / 100.0)
            for i in range(min(entry_end + 1, n)):
                if h[i] >= lim * (1.0 + th):
                    fill, i0 = max(lim, o[i]), i
                    break
        else:
            lim = p0 * (1.0 - k * sigma / 100.0)
            for i in range(min(entry_end + 1, n)):
                if l[i] <= lim * (1.0 - th):
                    fill, i0 = min(lim, o[i]), i
                    break
        if i0 == -2:
            return 0.0, 0, 4
    # ---- exit
    end = min(hz_end, n - 1)
    best = fill
    exit_px, code = c[end], 4
    for i in range(i0 + 1, end + 1):
        if side == 0:                                      # we sold: buy back
            if ex == 0:
                tgt = fill * (1.0 - y * sigma / 100.0)
                if l[i] <= tgt * (1.0 - th):
                    exit_px, code = min(tgt, o[i]), 0
                    break
            elif ex == 1:
                trig = best * (1.0 + z * sigma / 100.0)    # a rebound z sigma off the low since the sale
                if h[i] >= trig:
                    exit_px, code = max(trig, o[i]), 1
                    break
                best = min(best, l[i])
            elif ex == 2:
                tgt = fill * (1.0 - y * sigma / 100.0)
                stp = fill * (1.0 + s * sigma / 100.0)
                if h[i] >= stp:                            # the stop first when both are reached
                    exit_px, code = max(stp, o[i]), 3
                    break
                if l[i] <= tgt * (1.0 - th):
                    exit_px, code = min(tgt, o[i]), 2
                    break
        else:                                              # we bought: sell
            if ex == 0:
                tgt = fill * (1.0 + y * sigma / 100.0)
                if h[i] >= tgt * (1.0 + th):
                    exit_px, code = max(tgt, o[i]), 0
                    break
            elif ex == 1:
                trig = best * (1.0 - z * sigma / 100.0)
                if l[i] <= trig:
                    exit_px, code = min(trig, o[i]), 1
                    break
                best = max(best, h[i])
            elif ex == 2:
                tgt = fill * (1.0 + y * sigma / 100.0)
                stp = fill * (1.0 - s * sigma / 100.0)
                if l[i] <= stp:
                    exit_px, code = min(stp, o[i]), 3
                    break
                if h[i] >= tgt * (1.0 + th):
                    exit_px, code = max(tgt, o[i]), 2
                    break
    g = (fill - exit_px) if side == 0 else (exit_px - fill)
    return g / p0 * 100.0 - 2.0 * COST, 1, code


@njit(cache=True)
def simulate(o, h, l, c, p0, sigma, entry_end, hz_ends, side, k, ex, y, z, s, hz):
    """All actions on one path. hz_ends: bar index of each horizon; the action arrays are
    parallel. Returns gains (%), filled flags and exit codes."""
    m = len(side)
    gains = np.empty(m, np.float32)
    filled = np.empty(m, np.int8)
    codes = np.empty(m, np.int8)
    for a in range(m):
        e = hz_ends[hz[a]]
        if e < 0:
            gains[a], filled[a], codes[a] = np.nan, 0, 4
            continue
        g, f, cd = _leg(o, h, l, c, p0, sigma, side[a], k[a], ex[a], y[a], z[a], s[a], min(entry_end, e), e)
        gains[a], filled[a], codes[a] = g, f, cd
    return gains, filled, codes
