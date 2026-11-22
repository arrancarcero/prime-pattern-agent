"""Null models: what a pattern must beat.

cramer_wheel: random 'primes' that keep the wheel (no multiples of 2, 3, 5, 7) but are
otherwise random with density 1/log n. Anything real primes do beyond this model is
structure the wheel alone does not explain.
"""
from __future__ import annotations

import numpy as np

from ppa.core.projections import wheel


def cramer_wheel(lo: int, hi: int, W: int = 210, seed: int = 0,
                 block: int = 10_000_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.array(wheel(W), dtype=np.int64)
    phi = len(r)
    block = max(W, (block // W) * W)      # blocks must line up with the wheel
    out = []
    for a in range(lo - lo % W, hi, block):
        bases = np.arange(a, a + block, W, dtype=np.int64)
        cand = (bases[:, None] + r[None, :]).ravel()   # sorted: r < W, bases step W
        cand = cand[(cand >= max(lo, 11)) & (cand < hi)]
        prob = W / (phi * np.log(cand))
        out.append(cand[rng.random(len(cand)) < prob])
    return np.concatenate(out)
