"""Dirichlet calibration (Kull et al. 2019) for multiclass probabilities: softmax(W log p + b), with the
off-diagonal weights and the intercepts L2-penalised (ODIR). Used by barrier_model (situations / live);
a module of its own so it imports without the data files the model's other imports read.
"""
from __future__ import annotations

import numpy as np


def dirichlet_fit(P: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> dict:
    """Dirichlet calibration (Kull et al. 2019): softmax(W log p + b), off-diagonal and intercept L2-penalised."""
    from scipy.optimize import minimize
    L = np.log(np.clip(P, 1e-6, 1))
    Y = np.eye(P.shape[1])[y]
    off = ~np.eye(P.shape[1], dtype=bool)
    k = P.shape[1]

    def loss(th):
        W, b = th[:k * k].reshape(k, k), th[k * k:]
        Z = L @ W.T + b
        Z -= Z.max(axis=1, keepdims=True)
        Q = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
        return -np.mean(np.sum(Y * np.log(np.clip(Q, 1e-6, 1)), axis=1)) + lam * (np.sum(W[off] ** 2) + np.sum(b ** 2))
    th = minimize(loss, np.r_[np.eye(k).ravel(), np.zeros(k)], method="L-BFGS-B").x
    return {"W": th[:k * k].reshape(k, k).round(6).tolist(), "b": th[k * k:].round(6).tolist(), "lambda": lam}


def dirichlet_apply(c: dict, P: np.ndarray) -> np.ndarray:
    Z = np.log(np.clip(P, 1e-6, 1)) @ np.asarray(c["W"]).T + np.asarray(c["b"])
    Z -= Z.max(axis=1, keepdims=True)
    return np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
