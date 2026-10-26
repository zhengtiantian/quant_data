#!/usr/bin/env python3
"""A 1D-CNN challenger to the earnings-pulse LightGBM (PROJECT_PLAN Q.5 v4.3, 2026-10-26).

LightGBM sees the evening as a few numbers (the move since the close, its high and low, the
release minute). The CNN reads the last 90 one-minute bars before each placement itself (close,
high and low against the previous close, volume against the last half hour before the close) next
to the same state features, and predicts for all 84 steps (12 levels x 7 buy-back rules) at once
both the gain and its 10th percentile. Its out-of-sample predictions are written in the bake-off's
format, so the live policy (compose / run_policy) is judged on both models the same way.

    python research/models/earnings_pulse_cnn.py sequences   # the minute windows, once
    python research/models/earnings_pulse_cnn.py walkforward # train on earlier years, predict each year
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import earnings_pulse as ep  # noqa: E402

WIN = 90
OUT = ep.D / "cnn"
STEPS = [(lev, bb) for lev in ep.LEVELS for bb in ep.BUYBACK]


def sequences():
    """For every (report, placement): the last WIN one-minute bars before it, as 4 channels."""
    ev = ep.events()
    keys, seqs = [], []
    for n, r in enumerate(ev.itertuples(), 1):
        p = ep.M1 / f"{r.symbol}_{r.day}.parquet"
        if not p.exists():
            continue
        m = pd.read_parquet(p)
        m.index = m.index.tz_convert(ep.NY)
        t = m.index
        pos = lambda ts: int(np.searchsorted(t.values, ts.to_datetime64()))
        close_i = pos(pd.Timestamp(f"{r.day} 16:00", tz=ep.NY))
        if close_i == 0 or close_i >= len(m):
            continue
        c16 = float(m["close"].values[close_i - 1])
        vol = m["volume"].values.astype(float)
        vref = float(np.median(vol[max(0, close_i - 30):close_i])) or 1.0
        _, release_ts = ep._release(m, close_i, r, pos)
        place = {"pre": close_i, "c-5": pos(r.call - pd.Timedelta(minutes=5))}
        for k in (1, 3, 5, 10, 20, 30):
            place[f"r+{k}"] = pos(release_ts + pd.Timedelta(minutes=k))
        ch = np.stack([np.log(m["close"].values / c16), np.log(m["high"].values / c16), np.log(m["low"].values / c16),
                       np.log1p(vol / vref)], axis=1).astype(np.float32)
        for pl, i in place.items():
            if i <= 0 or i > len(m):
                continue
            w = ch[max(0, i - WIN):i]
            x = np.zeros((WIN, 4), np.float32)
            x[WIN - len(w):] = w
            keys.append((f"{r.symbol}|{r.date}", pl))
            seqs.append(x)
        if n % 500 == 0 or n == len(ev):
            ep.progress("sequences", n, len(ev))
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "seq.npy", np.stack(seqs))
    pd.DataFrame(keys, columns=["eid", "place"]).to_parquet(OUT / "seq_keys.parquet")
    print(f"DONE sequences: {len(keys)} windows", flush=True)


def _tables():
    """Static features and the 84 targets per (eid, place), aligned with the sequences."""
    df = ep.leg_frame()
    keys = pd.read_parquet(OUT / "seq_keys.parquet")
    seq = np.load(OUT / "seq.npy")
    keys["row"] = np.arange(len(keys))
    st = df.drop_duplicates(["eid", "place"]).set_index(["eid", "place"])[ep.STATE + ["own_sym", "year", "symbol"]]
    y = df.pivot_table(index=["eid", "place"], columns=["level", "buyback"], values="gain", aggfunc="first")
    y = y.reindex(columns=pd.MultiIndex.from_tuples(STEPS))
    own = df.pivot_table(index=["eid", "place"], columns=["level", "buyback"], values="own_leg", aggfunc="first")
    own = own.reindex(columns=pd.MultiIndex.from_tuples(STEPS))
    k = keys.set_index(["eid", "place"]).join(st, how="inner")
    k = k[k.index.isin(y.index)]
    # a stock's first report has no earlier steps: its own-history row is all empty and pivot_table drops it
    return k, seq[k["row"].values], y.reindex(k.index).values.astype(np.float32), own.reindex(k.index).values.astype(np.float32)


def walkforward():
    import torch
    import torch.nn as nn
    torch.manual_seed(7)
    torch.set_num_threads(int(os.getenv("CNN_THREADS", "6")))
    k, seq, Y, OWN = _tables()
    stat_cols = [c for c in ep.STATE + ["own_sym"]]
    S = k[stat_cols].astype(float).values
    smask = np.isnan(S)
    years = k["year"].values

    class Net(nn.Module):
        def __init__(self, n_stat, n_out):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(4, 32, 5, padding=2), nn.ReLU(), nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.ReLU(),
                                      nn.Conv1d(32, 32, 5, padding=8, dilation=4), nn.ReLU())
            self.stat = nn.Sequential(nn.Linear(n_stat, 32), nn.ReLU())
            self.own = nn.Linear(n_out, 32)
            self.head = nn.Sequential(nn.Linear(32 * 2 + 32 + 32, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n_out * 2))

        def forward(self, x, s, o):
            h = self.conv(x.transpose(1, 2))
            z = torch.cat([h.mean(2), h.amax(2), self.stat(s), self.own(o)], 1)
            out = self.head(z)
            return out[:, :Y.shape[1]], out[:, Y.shape[1]:]

    def prep(idx, mu, sd):
        s = np.where(smask[idx], 0.0, (S[idx] - mu) / sd)
        s = np.concatenate([s, smask[idx].astype(float)], 1)
        o = np.nan_to_num(OWN[idx], nan=0.0)
        return (torch.tensor(seq[idx] * 20.0), torch.tensor(s, dtype=torch.float32), torch.tensor(o))

    def loss_fn(mean, q, y):
        m = ~torch.isnan(y)
        yy = torch.nan_to_num(y)
        hub = nn.functional.huber_loss(mean[m], yy[m], delta=2.0)
        e = yy - q
        pin = torch.maximum(0.1 * e, (0.1 - 1) * e)[m].mean()
        return hub + pin

    (ep.D / "cnn_preds").mkdir(exist_ok=True)
    test_years = [y for y in sorted(set(years)) if y >= 2019]
    for n, yr in enumerate(test_years, 1):
        tr, te = np.where(years < yr)[0], np.where(years == yr)[0]
        mu, sd = np.nanmean(S[tr], 0), np.nanstd(S[tr], 0) + 1e-6
        Xs, Ss, Os = prep(tr, mu, sd)
        Yt = torch.tensor(Y[tr])
        net = Net(Ss.shape[1], Y.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
        for epoch in range(int(os.getenv("CNN_EPOCHS", "12"))):
            net.train()
            perm = torch.randperm(len(tr))
            for b in range(0, len(tr), 256):
                i = perm[b:b + 256]
                mean, q = net(Xs[i], Ss[i], Os[i])
                l = loss_fn(mean, q, Yt[i])
                opt.zero_grad()
                l.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            Xe, Se, Oe = prep(te, mu, sd)
            mean, q = net(Xe, Se, Oe)
        mean, q = mean.numpy(), q.numpy()
        rows = []
        kk = k.iloc[te]
        for j, ((eid, place), row) in enumerate(kk.iterrows()):
            for c, (lev, bb) in enumerate(STEPS):
                g = Y[te[j], c]
                if np.isnan(g):
                    continue
                rows.append((eid, row["symbol"], yr, place, lev, bb, float(g), float(mean[j, c]), float(q[j, c])))
        pd.DataFrame(rows, columns=["eid", "symbol", "year", "place", "level", "buyback", "gain", "pred", "q10"]).to_parquet(
            ep.D / "cnn_preds" / f"{yr}.parquet")
        ep.progress("cnn", n, len(test_years), f"{yr} train={len(tr)} test={len(te)}")
    print("DONE cnn walk-forward", flush=True)


if __name__ == "__main__":
    {"sequences": sequences, "walkforward": walkforward}[sys.argv[1]]()
