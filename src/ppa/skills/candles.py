"""Standard candles: redshift and brightness of the zeta zeros in the primes.

The explicit formula writes the prime error as waves, one per zeta zero rho = beta + i gamma:
    (psi(x) - x) / sqrt(x)  ~  - sum over zeros of 2 x^(beta - 1/2) cos(gamma log x - phase) / |rho|
Read it like an astronomer reads a galaxy:
  * colour (redshift): in ordinary numbers n, a zero's wave has period ~ 2 pi x / gamma, so it
    stretches as x grows. In u = log x it is a fixed spectral line at frequency gamma.
  * brightness: after dividing by sqrt(x), a zero ON the line (beta = 1/2) shines equally at
    every distance. A zero OFF the line brightens by x^(beta - 1/2), like a supernova that
    does not dim as predicted (how dark energy was found in 1998).
So:  measured brightness far / near = e^((beta - 1/2) * du)   ->   implied beta.

Null model (stated first): every zero has beta = 1/2, so the far/near brightness ratio is 1.
Calibration (the tool-tuple rule): a fake zero with beta = 0.6 is planted at gamma = 35, where
no real zero is; the skill must recover beta ~ 0.6, or its own verdicts are not trusted.
Limit: this sees only zeros whose waves the data can resolve (gamma up to ~50 at x = 10^8).
It cannot prove RH: an off-line zero far higher up would be invisible at any reachable x.
"""
import math

import numpy as np

from ppa.core.ledger import Finding
from ppa.core.sieve import primes_between
from ppa.skills.base import Skill

# First ten zeta zero heights (Odlyzko's tables)
ZEROS = (14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
         37.586178, 40.918719, 43.327073, 48.005151, 49.773832)
FAKE = (0.6, 35.0)          # planted off-line zero for calibration (beta, gamma)
EMPTY = 18.0                # a frequency with no zero: should stay dark


def psi_curve(max_x: int, points: int = 20_000):
    """u = log x on an even grid, and f(u) = (psi(x) - x)/sqrt(x) with its mean removed."""
    P = primes_between(2, max_x).astype(np.float64)
    pts, w = [P], [np.log(P)]
    k = 2
    while 2.0 ** k <= max_x:
        Q = P[P ** k <= max_x]
        pts.append(Q ** k)
        w.append(np.log(Q))
        k += 1
    pts, w = np.concatenate(pts), np.concatenate(w)
    o = np.argsort(pts)
    pts, cum = pts[o], np.cumsum(w[o])
    u = np.linspace(math.log(100), math.log(max_x), points)
    x = np.exp(u)
    f = (cum[np.searchsorted(pts, x, side="right") - 1] - x) / np.sqrt(x)
    return u, f - f.mean()


def brightness(f, u, gamma) -> float:
    """Hann-windowed amplitude of the wave at frequency gamma (in log x)."""
    w = np.hanning(len(u))
    return float(abs(np.sum(w * f * np.exp(-1j * gamma * u))) / np.sum(w))


def peak_near(f, u, gamma, half_width=0.6) -> float:
    gs = np.linspace(gamma - half_width, gamma + half_width, 241)
    return float(gs[int(np.argmax([brightness(f, u, g) for g in gs]))])


def candle(f, u, gamma):
    """(near brightness, far brightness, implied beta)."""
    h = len(u) // 2
    a, b = brightness(f[:h], u[:h], gamma), brightness(f[h:], u[h:], gamma)
    du = u[h:].mean() - u[:h].mean()
    return a, b, 0.5 + math.log(b / a) / du


def run(ctx, max_x=None, tol=0.03, **_):
    if max_x is None:
        max_x = int(min(ctx.cfg.get("candles", {}).get("max_x", 10 ** 8), 10 ** 9))
    u, f = psi_curve(max_x)
    out = []

    # 1. Calibration: can the skill see an off-line zero at all?
    beta_f, gamma_f = FAKE
    fake = f - 2 * np.exp((beta_f - 0.5) * u) * np.cos(gamma_f * u) / abs(complex(beta_f, gamma_f))
    _, _, b_fake = candle(fake, u, gamma_f)
    dark = brightness(f, u, EMPTY) / brightness(f, u, ZEROS[0])
    calibrated = abs(b_fake - beta_f) < tol and dark < 0.1
    out.append(Finding(
        "candles", "calibration: planted off-line zero is detected", "established",
        f"planted beta = {beta_f} at gamma = {gamma_f}: recovered beta = {b_fake:.3f}; "
        f"empty frequency {EMPTY} is {dark:.1%} as bright as the first zero",
        survived=calibrated, details={"recovered_beta": round(b_fake, 4)}))

    # 2. Redshift: do the spectral lines sit on the known zero heights?
    peaks = [peak_near(f, u, g) for g in ZEROS]
    worst = max(abs(p - g) for p, g in zip(peaks, ZEROS))
    out.append(Finding(
        "candles", "redshift: prime spectral lines = zeta zero heights", "established",
        f"{len(ZEROS)} lines found within {worst:.3f} of the known heights (primes to {max_x:.0e}); "
        f"wave period in n = 2 pi x / gamma, e.g. {2 * math.pi * max_x / ZEROS[0]:.3g} at x = {max_x:.0e}",
        survived=worst < 0.1,
        details={"peaks": [round(p, 3) for p in peaks]}))

    # 3. Brightness: does every candle hold steady (beta = 1/2)?
    rows = {g: candle(f, u, g) for g in ZEROS}
    betas = [r[2] for r in rows.values()]
    drift = max(abs(b - 0.5) for b in betas)
    out.append(Finding(
        "candles", "standard candles: implied real part of each zero", "established",
        f"implied beta for the first {len(ZEROS)} zeros: {min(betas):.3f} to {max(betas):.3f} "
        f"(null: 0.5, tolerance {tol}); largest drift {drift:.3f}"
        + ("" if calibrated else "  [calibration failed: verdict not trusted]"),
        survived=(drift < tol) if calibrated else None,
        details={f"{g:.2f}": {"near": round(a, 4), "far": round(b, 4), "beta": round(be, 4)}
                 for g, (a, b, be) in rows.items()}))
    return out


SKILL = Skill("candles", "every zero has real part 1/2: brightness far/near = 1 after dividing by sqrt(x)",
              "Redshift and standard-candle test of the zeta zeros inside the primes",
              run, needs_primes=False)
