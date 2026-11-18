"""The barrier model's Dirichlet calibration: a distribution out, and better than raw on overconfident input."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "models"))

import dirichlet as bm  # noqa: E402 - no data files at import (CI)


def test_dirichlet_keeps_a_distribution_and_fixes_overconfidence():
    rng = np.random.default_rng(7)
    n = 6000
    true = rng.dirichlet([2, 3, 4], n)
    y = np.array([rng.choice(3, p=p) for p in true])
    over = true ** 3 / (true ** 3).sum(axis=1, keepdims=True)          # sharper than the truth
    c = bm.dirichlet_fit(over, y)
    q = bm.dirichlet_apply(c, over)
    assert np.allclose(q.sum(axis=1), 1.0)
    assert (q >= 0).all()
    nll = lambda p: -np.mean(np.log(p[np.arange(n), y]))               # noqa: E731
    assert nll(q) < nll(over)
