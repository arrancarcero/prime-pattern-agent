"""The spiral picture: the smoothed one-way walk at the first zeta zero, zoomed in three
times, and the local dimension at each zoom level for the zig-zag walk, the smoothed spiral
and a null walk (same step lengths, random directions).

    python examples/walk_fractal.py      -> walk_fractal.png
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppa.skills.candles import ZEROS  # noqa: E402
from ppa.core.fractal import local_slopes  # noqa: E402
from ppa.skills.walk import eta_averaged, eta_stops  # noqa: E402

N = 200_000
raw = np.concatenate([[0], eta_stops(ZEROS[0], N)])
P = eta_averaged(raw)          # midpoints: the spiral without the zig-zag
end = P[-1]
fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))
for a, (w, lab) in zip(ax[:3], [(1.0, "whole walk"), (0.05, "zoom x20"), (0.0025, "zoom x400")]):
    m = (np.abs(P.real - end.real) < w) & (np.abs(P.imag - end.imag) < w)
    a.plot(P.real[m], P.imag[m], lw=0.6, color="#2b6cb0")
    a.plot(0, 0, "o", color="#c05621", ms=5)
    a.set_xlim(end.real - w, end.real + w)
    a.set_ylim(end.imag - w, end.imag + w)
    a.set_aspect("equal")
    a.set_title(f"smoothed walk home, t = {ZEROS[0]:.3f}: {lab}")
    a.set_xticks([])
    a.set_yticks([])
rng = np.random.default_rng(0)
n = np.arange(1, N + 1)
null = np.concatenate([[0], np.cumsum(n ** -0.5 * np.exp(2j * np.pi * rng.random(N)))])
scales = np.geomspace(4, 1024, 9)
mid = np.sqrt(scales[:-1] * scales[1:])
for path, lab, c in [(raw, "zig-zag walk (alternating signs)", "#2b6cb0"),
                     (P, "smoothed spiral (midpoints)", "#2f855a"),
                     (null, "null: random directions", "#a0aec0")]:
    ax[3].semilogx(mid, local_slopes(path, lo=1 / 1024, hi=1 / 4, k=9), "o-", color=c, label=lab)
ax[3].axhline(1, color="k", lw=0.8, ls=":")
ax[3].axhline(np.log(4) / np.log(3), color="#c05621", lw=0.8, ls=":")
ax[3].text(5, 1.275, "Koch curve 1.26", fontsize=8, color="#c05621")
ax[3].text(5, 1.01, "smooth curve 1", fontsize=8)
ax[3].set_xlabel("zoom (1 / box size, relative to the walk's size)")
ax[3].set_ylabel("local dimension")
ax[3].set_title("dimension at each zoom level")
ax[3].legend(fontsize=8)
fig.tight_layout()
fig.savefig("walk_fractal.png", dpi=120)
