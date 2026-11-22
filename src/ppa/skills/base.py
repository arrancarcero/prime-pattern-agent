"""What every skill looks like, and the shared run context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Context:
    cfg: dict
    train: tuple[int, int]
    test: tuple[int, int]
    P_train: np.ndarray
    P_test: np.ndarray
    cache: dict = field(default_factory=dict)

    def null_primes(self, which: str) -> np.ndarray:
        """Cramer-wheel model primes for the same range (built once, then cached)."""
        from ppa.core.nulls import cramer_wheel
        key = f"null_{which}"
        if key not in self.cache:
            lo, hi = self.train if which == "train" else self.test
            nm = self.cfg["null_model"]
            seed = nm["seed"] + (0 if which == "train" else 1)
            self.cache[key] = cramer_wheel(lo, hi, nm["wheel"], seed)
        return self.cache[key]


@dataclass
class Skill:
    name: str
    null_model: str          # stated before any data is looked at (contract rule 2)
    description: str
    run: Callable            # run(ctx, **params) -> list[Finding]
    needs_primes: bool = True
