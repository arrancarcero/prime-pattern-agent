"""
randomness_tuple.py
===================
A frozen tuple that outlines what TRUE RANDOMNESS must look like, and a
sieve that checks any +1/-1 sequence against it.

Key idea: no finite test can PROVE a sequence is random. The tuple lists
properties every truly random sequence has, with the spread chance allows.
A sequence that breaks any row is NOT random. A sequence that passes every
row is "indistinguishable from random by these tests", which is the same
standard cryptography uses (NIST SP 800-22 works this way).

Each row: (name, statistic, expected, sd_rule, why)
  statistic is turned into a z-score; |z| > 4 means "not random".

Sequences tested in the self-run:
  1. os.urandom bits                -> the control, should pass
  2. Liouville signs lambda(n)      -> a sign flip set by prime factors:
                                       +1 if n has an even number of prime
                                       factors, -1 if odd
  3. prime mod-3 class signs        -> +1 if p = 1 mod 3, -1 if p = 2 mod 3
                                       (your digital-root buckets {1,4,7}
                                       versus {2,5,8})
Run:  python randomness_tuple.py 10000000
"""
from __future__ import annotations

import math
import os
import sys
from collections import namedtuple

import numpy as np

Row = namedtuple("Row", "name expected why")

RANDOMNESS_TUPLE: tuple[Row, ...] = (
    Row("balance", 0.0,
        "A fair coin gives as many +1 as -1. Sum / sqrt(n) stays near 0."),
    Row("sign_flips", 0.0,
        "Each step flips sign with chance 1/2: (n-1)/2 flips, spread sqrt(n-1)/2."),
    Row("lag1_memory", 0.0,
        "Each value tells you nothing about the next: lag-1 correlation near 0."),
    Row("lag2_memory", 0.0,
        "Nor about the one after that: lag-2 correlation near 0."),
    Row("spectral_peak", 0.0,
        "No hidden period: the biggest FFT peak is no bigger than chance allows."),
    Row("walk_scale", 0.0,
        "As a random walk, the running sum wanders about sqrt(n), not more, not less."),
)


def sieve_of_randomness(x: np.ndarray) -> dict:
    """Return a z-score for each row. |z| <= 4 passes."""
    x = x.astype(np.float64)
    n = len(x)
    z = {}
    z["balance"] = x.sum() / math.sqrt(n)
    flips = np.count_nonzero(x[1:] != x[:-1])
    z["sign_flips"] = (flips - (n - 1) / 2) / (math.sqrt(n - 1) / 2)
    z["lag1_memory"] = float(np.mean(x[1:] * x[:-1])) * math.sqrt(n - 1)
    z["lag2_memory"] = float(np.mean(x[2:] * x[:-2])) * math.sqrt(n - 2)
    # Spectral: periodogram values of white noise are exponential with mean 1;
    # the max of m of them is about log(m) with sd about 1.28.
    m = min(n, 2 ** 22)
    S = np.abs(np.fft.rfft(x[:m] - x[:m].mean())) ** 2 / m
    S = S[1:]
    z["spectral_peak"] = (S.max() - math.log(len(S))) / 1.28
    # Walk: for a random walk, max|S_k|/sqrt(n) has mean about 1.25, sd about 0.6
    walk = np.cumsum(x)
    z["walk_scale"] = (np.abs(walk).max() / math.sqrt(n) - 1.2533) / 0.6
    return {k: round(float(v), 2) for k, v in z.items()}


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------
def true_random_signs(n: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(os.urandom((n + 7) // 8), dtype=np.uint8))[:n]
    return bits.astype(np.int8) * 2 - 1


def liouville_signs(N: int) -> np.ndarray:
    """lambda(n) = (-1)^Omega(n) for n = 1..N  (Omega counts prime factors with repeats)."""
    omega = np.zeros(N + 1, dtype=np.int8)
    s = np.ones(N + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    for p in np.nonzero(s)[0]:
        q = int(p)
        while q <= N:
            omega[q::q] += 1
            q *= int(p)  # next power of p
    lam = np.where(omega % 2 == 0, 1, -1).astype(np.int8)
    return lam[1:]


def prime_class_signs(N: int) -> np.ndarray:
    s = np.ones(N + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    P = np.nonzero(s)[0]
    P = P[P > 3]
    return np.where(P % 3 == 1, 1, -1).astype(np.int8)


if __name__ == "__main__":
    N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10 ** 7
    seqs = {
        "true random (os.urandom)": true_random_signs(N),
        "Liouville sign flips": liouville_signs(N),
        "prime mod-3 bucket signs": prime_class_signs(N * 10),
    }
    names = [r.name for r in RANDOMNESS_TUPLE]
    print(f"{'sequence':28}" + "".join(f"{k:>14}" for k in names) + "   verdict")
    for label, x in seqs.items():
        z = sieve_of_randomness(x)
        bad = [k for k in names if abs(z[k]) > 4]
        verdict = "indistinguishable from random" if not bad else "NOT random: " + ", ".join(bad)
        print(f"{label:28}" + "".join(f"{z[k]:>14}" for k in names) + f"   {verdict}  (n={len(x):,})")
