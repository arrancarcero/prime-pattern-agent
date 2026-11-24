"""Shor's algorithm on the simulated quantum computer: what the measurement sees for N = 21.

    python examples/quantum_shor.py      -> quantum_shor.png
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppa.skills.quantum import order_finding_state, shor  # noqa: E402

probs, t, L = order_finding_state(21, 2)
Q = 1 << t
r = shor(21)
plt.figure(figsize=(9, 4.5))
plt.bar(np.arange(Q), probs, width=1.0, color="#2b6cb0")
for s in range(r["r"]):
    plt.axvline(s * Q / r["r"], color="#c05621", ls=":", lw=1)
plt.title(f"Shor on 21 ({t + L} simulated qubits): peaks every {Q}/{r['r']} -> period {r['r']} -> "
          f"factors {r['factors']}")
plt.xlabel("measured value k")
plt.ylabel("probability")
plt.tight_layout()
plt.savefig("quantum_shor.png", dpi=120)
