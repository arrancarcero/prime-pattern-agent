"""Pictures behind the moire and zone_plate tools in the tool tuple.

    python examples/moire_zone_plates.py        -> moire_zone_plates.png

Panel 1: grids with spacings p and q overlap; the beat repeats every N/(q - p).
Panel 2: the same grids inverted (dark <-> light): the beat is unchanged.
Panel 3: two zone plates side by side make straight fringes (the classic lens moire):
         cos(k r1^2) - cos(k r2^2) beat as cos(k (r1^2 - r2^2)), and r1^2 - r2^2 is a
         straight line in x when the centres are shifted along x.
Panel 4: a zone plate cos(pi x^2 / N) sampled every s-th point. If s divides N, the samples
         repeat far sooner (aliasing): the factor shows up as a shorter cycle.
"""
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

p, q = 29, 31
N = p * q
d = q - p

fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

# 1. Moire of two grids
x = np.linspace(0, 2 * N / d, 6000)
g1 = (np.cos(2 * np.pi * x / p) > 0).astype(float)
g2 = (np.cos(2 * np.pi * x / q) > 0).astype(float)
overlay = g1 * g2                                   # light only where both are light
win = int(len(x) / (2 * N / d) * p)                 # blur over one grid spacing
blur = np.convolve(overlay, np.ones(win) / win, mode="same")
ax[0, 0].plot(x, overlay * 0.4 + 0.6, color="#cbd5e0", lw=0.6)
ax[0, 0].plot(x, blur, color="#2b6cb0", lw=2, label="seen from far away (blurred)")
for k in range(3):
    ax[0, 0].axvline(k * N / d, color="#c05621", ls=":")
ax[0, 0].set_title(f"Grids spaced {p} and {q} overlap: beat every N/(q-p) = {N}/{d} = {N / d:g}")
ax[0, 0].set_xlabel("position")
ax[0, 0].legend(loc="lower right")

# 2. Inverted grids: same beat
inv = (1 - g1) * (1 - g2)
blur_i = np.convolve(inv, np.ones(win) / win, mode="same")
ax[0, 1].plot(x, blur, color="#2b6cb0", lw=2, label="original")
ax[0, 1].plot(x, blur_i, color="#c05621", lw=2, ls="--", label="inverted (dark <-> light)")
for k in range(3):
    ax[0, 1].axvline(k * N / d, color="#718096", ls=":")
ax[0, 1].set_title("Inverting the grids moves the brightness, not the beat")
ax[0, 1].set_xlabel("position")
ax[0, 1].legend(loc="lower right")

# 3. Two zone plates: straight fringes
n = 600
yy, xx = np.mgrid[-1:1:n * 1j, -1.5:1.5:int(1.5 * n) * 1j]
k, shift = 60.0, 0.12
z1 = np.cos(k * ((xx - shift) ** 2 + yy ** 2)) > 0
z2 = np.cos(k * ((xx + shift) ** 2 + yy ** 2)) > 0
ax[1, 0].imshow(z1 ^ z2, cmap="gray", extent=(-1.5, 1.5, -1, 1))
ax[1, 0].set_title("Two zone plates, centres shifted: straight moire fringes")
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])

# 4. Zone plate sampled with stride s: factor strides alias
Nz = 7 * 11
m = np.arange(0, 2 * Nz)
for s, c in ((7, "#c05621"), (5, "#2b6cb0")):
    vals = np.cos(np.pi * (s * s * m * m % (2 * Nz)) / Nz)
    cyc = Nz // math.gcd(s * s, Nz)
    lab = f"stride {s}: {'shares a factor with' if cyc < Nz else 'coprime to'} {Nz}, repeats every {cyc}"
    ax[1, 1].plot(m, vals + (1.2 if s == 7 else -1.2), color=c, lw=1.2, marker=".", ms=3, label=lab)
ax[1, 1].set_title(f"Zone plate cos(pi x^2 / {Nz}) sampled every s-th point")
ax[1, 1].set_xlabel("sample number")
ax[1, 1].set_yticks([])
ax[1, 1].legend(loc="upper right", fontsize=8)

fig.tight_layout()
fig.savefig("moire_zone_plates.png", dpi=120)
print(f"beat period {N}/{d} = {N / d:g};  p = (sqrt(d^2 + 4N) - d)/2 = {(math.isqrt(d * d + 4 * N) - d) // 2}")
