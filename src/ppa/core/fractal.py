"""Box-counting fractal dimension of a path in the plane.

Cover the path with a grid of boxes of size eps and count the boxes it touches, N(eps).
For a fractal, N(eps) ~ eps^(-D): a straight line has D = 1, a filled area D = 2, the Koch
snowflake curve D = log 4 / log 3 = 1.262. D is the slope of log N against log(1/eps).
Local slopes (one per pair of neighbouring scales) show whether D holds at every zoom
level: steady local slopes mean the picture looks alike at every size.

Calibration (koch_curve, a line) is part of the walk skill: the counter must read known
dimensions correctly before its readings on the zeta walks are trusted. Random walks read
below their true D = 2 at these sizes (finite length), so nulls are compared at the same
number of steps and the same scales, never against the textbook value.
"""
from __future__ import annotations

import numpy as np


def densify(P: np.ndarray, h: float) -> np.ndarray:
    """Points every <= h along the polyline P (complex numbers), so no box is skipped."""
    seg = np.diff(P)
    k = np.maximum(1, np.ceil(np.abs(seg) / h)).astype(np.int64)
    idx = np.repeat(np.arange(len(seg)), k)
    frac = np.concatenate([np.arange(m) / m for m in k])
    return np.concatenate([P[:-1][idx] + seg[idx] * frac, [P[-1]]])


def box_counts(P: np.ndarray, lo: float = 1 / 512, hi: float = 1 / 8, k: int = 10):
    ext = max(np.ptp(P.real), np.ptp(P.imag))
    eps = ext * np.geomspace(hi, lo, k)
    Q = densify(P, eps[-1] / 4)
    x, y = Q.real - Q.real.min(), Q.imag - Q.imag.min()
    cnt = np.array([len(np.unique(np.floor(x / e).astype(np.int64) * 10 ** 9
                                  + np.floor(y / e).astype(np.int64))) for e in eps])
    return eps, cnt


def box_dimension(P: np.ndarray, **kw) -> float:
    eps, cnt = box_counts(P, **kw)
    return float(np.polyfit(np.log(1 / eps), np.log(cnt), 1)[0])


def local_slopes(P: np.ndarray, **kw) -> list[float]:
    eps, cnt = box_counts(P, **kw)
    return [round(float(v), 3) for v in np.diff(np.log(cnt)) / np.diff(np.log(1 / eps))]


def koch_curve(level: int = 8) -> np.ndarray:
    P = np.array([0, 1], dtype=complex)
    r = np.exp(1j * np.pi / 3)
    for _ in range(level):
        a, b = P[:-1], P[1:]
        d = (b - a) / 3
        P = np.concatenate([np.stack([a, a + d, a + d + d * r, a + 2 * d], 1).ravel(), [P[-1]]])
    return P
