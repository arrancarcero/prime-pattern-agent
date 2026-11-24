"""The walk home: zeta zeros as walks of arrows that end at the origin.

One-way walk (eta): the alternating zeta function
    eta(s) = 1 - 2^-s + 3^-s - 4^-s + ...,   s = 1/2 + i t
adds arrows of length 1/sqrt(n), each turned by the angle t log n, signs alternating.
eta(s) = (1 - 2^(1-s)) zeta(s), so at a zeta zero the walk ends exactly at the origin:
the zero "walks home". But it walks toward infinity: plain partial sums wobble by about
1/(2 sqrt(N)), so getting within r of home takes about 1/(4 r^2) arrows. Averaging each
pair of neighbouring stops (the midpoint of the last arrow) settles far faster.

Two-way walk (Riemann-Siegel, "counting back from infinity"): the infinite tail folds
back as a mirror image. With M = floor(sqrt(t / 2 pi)) arrows forward,
    zeta(1/2 + it) ~ F + e^(-2i theta(t)) * conj(F),   F = sum_{n <= M} n^(-1/2 - it)
the mirror half is the forward half reflected, coming in from infinity. At a zero the two
halves cancel, so the walk ends at the origin after only 2M arrows (plus a small
correction). Its distance from home is |Z(t)|, Hardy's Z, which spacing.py also uses.

Null model (stated first): at heights between zeros the walks do NOT end at the origin.
"""
import math

import numpy as np

from ppa.core.ledger import Finding
from ppa.skills.base import Skill
from ppa.skills.candles import ZEROS
from ppa.skills.spacing import hardy_z, theta, zeros_up_to


def eta_stops(t: float, N: int) -> np.ndarray:
    """Every stop of the one-way walk: partial sums of (-1)^(n+1) n^(-1/2 - it), n = 1..N."""
    n = np.arange(1, N + 1, dtype=np.float64)
    steps = np.where(n % 2 == 1, 1.0, -1.0) * n ** -0.5 * np.exp(-1j * t * np.log(n))
    return np.cumsum(steps)


def eta_averaged(stops: np.ndarray) -> np.ndarray:
    """Midpoint of each arrow: (S_N + S_(N+1)) / 2, which settles much faster."""
    return (stops[:-1] + stops[1:]) / 2


def two_way(t: float):
    """(forward stops, full walk stops) of the Riemann-Siegel two-way walk, rotated by
    e^(i theta) so the end lands on the real line at Z(t) (main sum, no correction)."""
    M = int(math.floor(math.sqrt(t / (2 * math.pi))))
    n = np.arange(1, M + 1, dtype=np.float64)
    fwd = np.exp(1j * theta(t)) * n ** -0.5 * np.exp(-1j * t * np.log(n))
    back = np.conj(fwd[::-1])                       # the mirror walk, from the far end inward
    path = np.concatenate([[0], np.cumsum(np.concatenate([fwd, back]))])
    return M, path


def steps_needed(seq: np.ndarray, r: float) -> int | None:
    """First N after which the walk stays within r of the origin (None: not within len)."""
    far = np.nonzero(np.abs(seq) > r)[0]
    if len(far) == 0:
        return 1
    last = int(far[-1]) + 2
    return last if last <= len(seq) else None


def run(ctx, N=400_000, **_):
    out = []
    mids = [(a + b) / 2 for a, b in zip(ZEROS, ZEROS[1:])]

    # 1. Both walks come home at zeros, and not between them (null)
    eta_zero = [abs(eta_averaged(eta_stops(g, N))[-1]) for g in ZEROS]
    eta_mid = [abs(eta_averaged(eta_stops(t, N))[-1]) for t in mids]
    rs_zero = [abs(hardy_z(g)[0]) for g in ZEROS]
    rs_mid = [abs(hardy_z(t)[0]) for t in mids]
    home = max(eta_zero) < 1e-4 and max(rs_zero) < 0.05 and min(eta_mid) > 0.05 and min(rs_mid) > 0.1
    out.append(Finding(
        "walk", "walk home: both walks end at the origin at zeta zeros", "established",
        f"first {len(ZEROS)} zeros: one-way walk ({N:,} arrows, averaged) ends within "
        f"{max(eta_zero):.1e} of home; two-way walk within {max(rs_zero):.1e}. Between zeros "
        f"(null): one-way at least {min(eta_mid):.2f} away, two-way at least {min(rs_mid):.2f}",
        survived=home))

    # 2. How many arrows each method needs for the same accuracy
    rows = {}
    for g in ZEROS:
        M, _ = two_way(g)
        r = max(abs(hardy_z(g)[0]), 1e-3)            # the two-way walk's own accuracy
        S = eta_stops(g, N)
        rows[g] = {"two_way_arrows": 2 * M, "target": round(r, 5),
                   "one_way_averaged": steps_needed(eta_averaged(S), r),
                   "one_way_plain": steps_needed(S, r) or f">{N:,} (about {int(1 / (4 * r * r)):,})"}
    g0 = ZEROS[0]
    out.append(Finding(
        "walk", "counting back from infinity: arrows needed for the same accuracy", "established",
        f"first zero (t = {g0:.2f}): two-way walk {rows[g0]['two_way_arrows']} arrows; one-way walk "
        f"{rows[g0]['one_way_averaged']} arrows averaged, {rows[g0]['one_way_plain']} plain",
        survived=all(isinstance(v["one_way_averaged"], int) and v["one_way_averaged"] > v["two_way_arrows"]
                     for v in rows.values()),
        details={f"{g:.2f}": v for g, v in rows.items()}))

    # 3. Fractal dimension of the walk paths (the spiral picture)
    out.append(fractal_finding())

    # 4. Cross-check with the spacing skill's zero finder
    found = zeros_up_to(max(ZEROS) + 1)
    err = max(abs(a - b) for a, b in zip(found, ZEROS))
    out.append(Finding(
        "walk", "cross-check: walk home heights = spacing skill's zeros", "established",
        f"{len(found)} zeros found below {max(ZEROS) + 1:.0f} (expected {len(ZEROS)}), "
        f"largest height difference {err:.1e}",
        survived=len(found) == len(ZEROS) and err < 0.01))
    return out


def fractal_finding(N=20_000, seed=0):
    """Box-counting dimension of the one-way walk at zeros, between zeros, and of a null
    walk with the same step lengths 1/sqrt(n) but random directions. Calibrated first on
    a straight line (D = 1) and the Koch curve (D = 1.262)."""
    from ppa.core.fractal import box_dimension, koch_curve, local_slopes
    line = box_dimension(np.array([0, 1 + 0.3j]))
    koch = box_dimension(koch_curve(8))
    calibrated = abs(line - 1) < 0.05 and abs(koch - math.log(4) / math.log(3)) < 0.05
    rng = np.random.default_rng(seed)
    n = np.arange(1, N + 1, dtype=np.float64)
    path = lambda steps: np.concatenate([[0], np.cumsum(steps)])
    mids = [(a + b) / 2 for a, b in zip(ZEROS, ZEROS[1:])]
    d_zero = [box_dimension(path(np.diff(np.concatenate([[0], eta_stops(g, N)])))) for g in ZEROS]
    d_mid = [box_dimension(path(np.diff(np.concatenate([[0], eta_stops(t, N)])))) for t in mids]
    d_null = [box_dimension(path(n ** -0.5 * np.exp(2j * np.pi * rng.random(N)))) for _ in range(10)]
    m = lambda v: (float(np.mean(v)), float(np.std(v)))
    (z, zs), (b, bs), (u, us) = m(d_zero), m(d_mid), m(d_null)
    raw = np.concatenate([[0], eta_stops(ZEROS[0], 200_000)])
    slopes = local_slopes(raw, lo=1 / 1024, hi=1 / 4, k=9)
    smooth = local_slopes(eta_averaged(raw), lo=1 / 1024, hi=1 / 4, k=9)
    structured = u - z > 5 * max(zs, us)
    same_shape = abs(z - b) < 2 * max(zs, bs) + 0.02
    return Finding(
        "walk", "fractal dimension of the walk spirals", "established",
        f"calibration: line {line:.2f} (1), Koch {koch:.3f} (1.262). One-way walk ({N:,} arrows): "
        f"at zeros D = {z:.3f} +- {zs:.3f}, between zeros D = {b:.3f} +- {bs:.3f}; null (same steps, "
        f"random directions) D = {u:.3f} +- {us:.3f}. Local slopes at the first zero, large to small "
        f"scales: zig-zag walk {slopes}; smoothed (midpoint) spiral {smooth}. The walks are more ordered "
        f"than chance; the steady ~1.2 comes from the zig-zag of the alternating signs, while the "
        f"spiral itself flattens toward D = 1 (a smooth curve) up close. Being a zero changes where "
        f"the walk ends, not its dimension",
        survived=(structured and same_shape) if calibrated else None,
        details={"D_zero": round(z, 4), "D_between": round(b, 4), "D_null": round(u, 4),
                 "koch": round(koch, 4), "local_slopes": slopes, "smoothed_local_slopes": smooth})


SKILL = Skill("walk", "between zeros the walks do not end at the origin",
              "Eta walk and the Riemann-Siegel two-way walk: zeros walk home",
              run, needs_primes=False)
