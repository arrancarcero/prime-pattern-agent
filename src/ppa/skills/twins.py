"""Twin primes: how they follow each other, tested against a Hardy-Littlewood-style null.

Measurements on the lower twins p (p, p + 2 both prime), p > 17:
  1. count vs Hardy-Littlewood: pi_2(x) ~ 2 C2 * integral dt / log^2 t
  2. digital-root transitions: the lower twin's dr is 2, 5 or 8; how often does the next
     twin repeat the same dr? (no structure: 1/3)
  3. gaps between consecutive twin pairs, scaled by the local average gap: share of tiny
     (< 0.25) and huge (> 3) gaps (random scattering: 22.1% and 5.0%)
  4. twins per block of 30030 numbers: Fano factor variance / mean (random scattering: 1)

Two nulls, like the transitions skill's stage A and stage B.
Stage A, the twin wheel. A twin pair can only start where p is not 0 or -2
mod every prime q <= 17 (22,275 of every 510,510 numbers: the 'admissible' positions that
Hardy-Littlewood and Maynard build on). The null keeps exactly those positions and picks each
one at random with the probability that matches the real twins' density in that stretch.
So it knows everything the small primes force (gaps are multiples of 6, where twins may
land) and nothing else.
Stage B, the sieved null: the same, but positions are struck out for every prime q <= Q
(default sqrt(x) / 30), sieved segment by segment. Q stays far below sqrt(x): at Q = sqrt(x) the null
would simply BE the twin primes.
A pattern that survives A but not B is explained by primes between 17 and Q (a rediscovery);
only a pattern beyond stage B by many standard deviations is a candidate.
Measured: at 1e8 every effect is explained by primes <= 97. At 1e9 the sharper data push
three effects past a sieve to 97 (|z| ~ 5 to 9), and a sieve to 1000 explains them again
(|z| < 5); the Fano factor still sits at z = -4.2 there (0.778 real vs 0.803), the gap
shrinking as Q grows. Rule learned: as x grows, Q must grow too.
"""
import math

import numpy as np

from ppa.core.ledger import Finding
from ppa.core.sieve import primes_between
from ppa.skills.base import Skill

C2 = 0.6601618158468696          # twin prime constant
WHEEL_PRIMES = (3, 5, 7, 11, 13, 17)
W = 2 * math.prod(WHEEL_PRIMES)  # 510510
BLOCK = 30030
DENSITY_STRETCH = 10 ** 7


def lower_twins(max_x: int) -> np.ndarray:
    P = primes_between(5, max_x)
    T = P[:-1][np.diff(P) == 2]
    return T[T > 17]


def admissible_residues() -> np.ndarray:
    r = np.arange(1, W, 2)
    ok = np.ones(len(r), dtype=bool)
    for q in WHEEL_PRIMES:
        ok &= (r % q != 0) & (r % q != q - 2)
    return r[ok]


def null_twins(real: np.ndarray, max_x: int, seed: int) -> np.ndarray:
    """Admissible positions, each kept with the probability matching the real density."""
    rng = np.random.default_rng(seed)
    res = admissible_residues()
    out = []
    for lo in range(0, max_x, DENSITY_STRETCH):
        hi = min(lo + DENSITY_STRETCH, max_x)
        k0, k1 = lo // W, hi // W + 1
        cand = (np.arange(k0, k1, dtype=np.int64)[:, None] * W + res[None, :]).ravel()
        cand = cand[(cand >= max(lo, 18)) & (cand < hi - 2)]
        n_real = int(np.sum((real >= lo) & (real < hi)))
        if len(cand) and n_real:
            out.append(cand[rng.random(len(cand)) < n_real / len(cand)])
    return np.sort(np.concatenate(out))


def sieved_null(real: np.ndarray, max_x: int, Q: int, seed: int) -> np.ndarray:
    """Stage B: strike p = 0 and p = -2 mod every prime q <= Q, then pick at the real density."""
    rng = np.random.default_rng(seed)
    qs = [int(q) for q in primes_between(3, Q + 1)]
    out = []
    for lo in range(0, max_x, DENSITY_STRETCH):
        hi = min(lo + DENSITY_STRETCH, max_x)
        ok = np.zeros(hi - lo, dtype=bool)
        ok[(1 - lo) % 2::2] = True
        for q in qs:
            ok[(-lo) % q::q] = False
            ok[(q - 2 - lo) % q::q] = False
        cand = np.nonzero(ok)[0].astype(np.int64) + lo
        cand = cand[(cand > 17) & (cand < max_x - 2)]
        n_real = int(np.sum((real >= lo) & (real < hi)))
        if len(cand) and n_real:
            out.append(cand[rng.random(len(cand)) < n_real / len(cand)])
    return np.sort(np.concatenate(out))


def metrics(T: np.ndarray) -> dict:
    dr = 1 + (T - 1) % 9
    cls = (dr - 2) // 3                                  # 2 -> 0, 5 -> 1, 8 -> 2
    repeat = float(np.mean(cls[1:] == cls[:-1]))
    step = float(np.mean((cls[1:] - cls[:-1]) % 3 == 1))  # 2 -> 5 -> 8 -> 2
    g = np.diff(T).astype(float)
    stretch = (T[1:] // DENSITY_STRETCH).astype(np.int64)
    mean_gap = np.bincount(stretch, g) / np.maximum(np.bincount(stretch), 1)
    s = g / mean_gap[stretch]
    counts = np.bincount((T // BLOCK).astype(np.int64))[:-1].astype(float)
    # Fano within each 1e7 stretch, so the slow thinning of twins (1/log^2) is not counted
    per = DENSITY_STRETCH // BLOCK
    fanos = [c.var() / c.mean() for i in range(1, len(counts) // per)
             if (c := counts[i * per:(i + 1) * per]).mean() > 0]
    return {"repeat": repeat, "step_forward": step,
            "tiny": float(np.mean(s < 0.25)), "huge": float(np.mean(s > 3)),
            "fano": float(np.mean(fanos))}


LABELS = {"repeat": "next twin repeats the digital root",
          "step_forward": "next twin steps forward 2 -> 5 -> 8",
          "tiny": "tiny gaps (< 0.25 x average)",
          "huge": "huge gaps (> 3 x average)",
          "fano": "Fano factor of twins per 30030-block"}
RANDOM = {"repeat": 1 / 3, "step_forward": 1 / 3, "tiny": 1 - math.exp(-0.25),
          "huge": math.exp(-3), "fano": 1.0}


def run(ctx, max_x=None, seeds=6, Q=None, **_):
    tcfg = ctx.cfg.get("twins", {})
    if max_x is None:
        max_x = int(min(tcfg.get("max_x", 10 ** 8), 2 * 10 ** 9))
    # More data makes the test sharper, so the sieve null must know more primes as x grows:
    # default Q = sqrt(x) / 30 (333 at 1e8, 1054 at 1e9), far below sqrt(x)
    Q = int(Q or tcfg.get("Q", max(97, math.isqrt(max_x) // 30)))
    Q = min(Q, int(math.isqrt(max_x) // 10))            # keep the null far from 'is the twins'

    from scipy.integrate import quad
    T = lower_twins(max_x)
    out = []

    # 1. count vs Hardy-Littlewood
    n_all = len(T) + 4                                   # (3,5), (5,7), (11,13), (17,19)
    li2 = quad(lambda t: 1 / math.log(t) ** 2, 2, max_x, limit=500)[0]
    pred = 2 * C2 * li2
    out.append(Finding(
        "twins", f"twin prime count to {max_x:.0e} vs Hardy-Littlewood", "established",
        f"{n_all:,} twin pairs; Hardy-Littlewood predicts {pred:,.0f} (ratio {n_all / pred:.5f})",
        survived=abs(n_all / pred - 1) < 0.01))

    # 2-4. structure between twins, real vs stage A (wheel to 17) and stage B (sieve to Q)
    real = metrics(T)
    A = [metrics(null_twins(T, max_x, s)) for s in range(seeds)]
    B = [metrics(sieved_null(T, max_x, Q, 100 + s)) for s in range(seeds)]

    def z(k, ms):
        v = np.array([m[k] for m in ms])
        mu, sd = float(v.mean()), float(v.std(ddof=1)) or 1e-12
        return mu, sd, (real[k] - mu) / sd

    for k in LABELS:
        ma, sa, za = z(k, A)
        mb, sb, zb = z(k, B)
        not_random = abs(real[k] - RANDOM[k]) > 0.01 * max(RANDOM[k], 0.05)
        if abs(zb) > 5:
            verdict, label = f"CANDIDATE: beyond the sieve to {Q}", "candidate"
        elif abs(za) > 5:
            verdict, label = f"rediscovery: explained by primes 19 to {Q}", "established"
        elif not_random:
            verdict, label = "rediscovery: explained by the wheel (primes up to 17)", "established"
        else:
            verdict, label = "no effect", "not-supported"
        out.append(Finding(
            "twins", f"twins: {LABELS[k]}", label,
            f"real {real[k]:.4f}; random {RANDOM[k]:.4f}; wheel to 17: {ma:.4f} (z = {za:+.1f}); "
            f"sieve to {Q}: {mb:.4f} (z = {zb:+.1f}). {verdict}",
            survived=abs(zb) > 5,
            details={"real": real[k], "random": RANDOM[k], "stageA": [ma, sa, za],
                     "stageB": [mb, sb, zb], "Q": Q}))
    return out


SKILL = Skill("twins", "twin-wheel null: admissible positions picked at random at the real density",
              "Twin primes: count vs Hardy-Littlewood, dr transitions, gaps, per-cycle Fano",
              run, needs_primes=False)
