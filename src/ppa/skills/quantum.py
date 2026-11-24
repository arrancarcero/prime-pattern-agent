"""A small quantum computer, simulated: Shor's algorithm and the memory wall.

The simulator keeps the full state vector: n qubits = 2^n complex amplitudes. Every gate is
applied to all of them. That is exactly what a classical machine must pay, and why each extra
qubit doubles the memory.

Shor's order finding (the quantum part of factoring N):
  1. counting register of t qubits (2^t >= N^2) in an even superposition of all j,
     work register of L qubits (2^L >= N) set to 1
  2. controlled modular multiplication: |j>|y> -> |j>|a^j * y mod N>   (a permutation, so unitary)
  3. inverse quantum Fourier transform on the counting register (an FFT along that axis)
  4. measure: the result k lands near a multiple of 2^t / r, where r is the order of a mod N
  5. continued fractions turn k / 2^t into s / r; then gcd(a^(r/2) +- 1, N) gives the factors
Step 3 is the same move as the sign_fft tool: hide the factors in a period, then Fourier
transform. A quantum computer holds the 2^t-long register in t qubits; the simulator has to
write all 2^t numbers down.

Uses CuPy (GPU) when installed and enabled in agent.toml ([quantum] gpu = true), else NumPy.
"""
import math
import time
from fractions import Fraction

import numpy as np

from ppa.core.ledger import Finding
from ppa.skills.base import Skill


def backend(gpu: bool):
    if gpu:
        try:
            import cupy as cp
            cp.zeros(1)
            return cp, "GPU (CuPy)"
        except Exception:  # noqa: BLE001
            pass
    return np, "CPU (NumPy)"


def order_finding_state(N: int, a: int, xp=np, dtype=np.complex128):
    """Run steps 1-3; return (probabilities of each counting value k, t, L)."""
    L = max(1, (N - 1).bit_length())
    t = 2 * L
    while (1 << t) < N * N:
        t += 1
    Q, W = 1 << t, 1 << L
    # 1. superposition over j, work register = 1
    state = xp.zeros((Q, W), dtype=dtype)
    state[:, 1] = 1 / math.sqrt(Q)
    # 2. controlled multiplication by a^j mod N, applied as a permutation of basis states
    aj = xp.asarray([pow(a, j, N) for j in range(Q)], dtype=xp.int64)
    y = xp.arange(W, dtype=xp.int64)[None, :]
    target = xp.where(y < N, (aj[:, None] * y) % N, y)
    new = xp.zeros_like(state)
    rows = xp.broadcast_to(xp.arange(Q)[:, None], (Q, W))
    new[rows, target] = state
    # 3. inverse QFT on the counting register = FFT along axis 0
    new = xp.fft.fft(new, axis=0) / math.sqrt(Q)
    probs = xp.sum(xp.abs(new) ** 2, axis=1)
    if xp is not np:
        probs = xp.asnumpy(probs)
    return probs, t, L


def shor(N: int, xp=np, seed: int = 1, max_shots: int = 40):
    """Factor N with simulated order finding. Returns a dict of what happened."""
    rng = np.random.default_rng(seed)
    for a in range(2, N):
        if math.gcd(a, N) != 1:
            continue          # a lucky gcd is not quantum; skip it to test the quantum part
        probs, t, L = order_finding_state(N, a, xp)
        probs = probs / probs.sum()
        Q = 1 << t
        for shot in range(1, max_shots + 1):
            k = int(rng.choice(Q, p=probs))
            r = Fraction(k, Q).limit_denominator(N - 1).denominator
            if pow(a, r, N) != 1 or r % 2:
                continue
            h = pow(a, r // 2, N)
            if h == N - 1:
                break         # this a is unlucky; try the next a
            f = math.gcd(h - 1, N)
            if 1 < f < N:
                peaks = int(np.sum(probs > 0.5 / r))
                return {"N": N, "a": a, "r": r, "factors": (f, N // f), "qubits": t + L,
                        "state_bytes": (1 << (t + L)) * 16, "shots": shot, "peaks": peaks}
    return {"N": N, "factors": None, "qubits": None}


def threads_across(N: int, a: int):
    """Entanglement across two cuts of Shor's state (after the multiplication step):
    the 'mirror' between counting and work registers, and a cut through the middle of
    the counting register after the QFT. Threads = Schmidt rank (non-zero singular values).
    A classical splitter (circuit cutting, tensor networks) pays for every thread.
    Across the mirror the count is exactly the period r: to fill in one side from the
    other you would already need the answer."""
    L = max(1, (N - 1).bit_length())
    t = 2 * L
    while (1 << t) < N * N:
        t += 1
    Q, W = 1 << t, 1 << L
    st = np.zeros((Q, W))
    j = np.arange(Q)
    st[j, [pow(a, int(x), N) for x in j]] = 1 / math.sqrt(Q)

    def rank_entropy(M):
        sv = np.linalg.svd(M, compute_uv=False)
        p = sv ** 2
        p = p[p > 1e-9]
        return int(len(p)), float(-(p * np.log2(p)).sum())

    mirror, ent = rank_entropy(st)
    after = np.fft.fft(st, axis=0) / math.sqrt(Q)
    mid, _ = rank_entropy(after.reshape(1 << (t // 2), -1))
    r = next(k for k in range(1, N) if pow(a, k, N) == 1)
    return {"N": N, "r": r, "qubits": t + L, "mirror_threads": mirror,
            "entanglement_bits": round(ent, 3), "mid_counting_threads": mid}


def memory_wall(xp, sizes, dtype=np.complex64):
    """Allocate a 2^n state and apply a QFT to all n qubits; time it for each n."""
    rows = []
    for n in sizes:
        t0 = time.perf_counter()
        v = xp.ones(1 << n, dtype=dtype)
        v = xp.fft.fft(v)
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
        rows.append((n, (1 << n) * np.dtype(dtype).itemsize, time.perf_counter() - t0))
        del v
    return rows


def run(ctx, numbers=(15, 21, 35, 143), max_qubits=None, **_):
    qcfg = ctx.cfg.get("quantum", {})
    xp, where = backend(bool(qcfg.get("gpu", False)))
    max_qubits = int(max_qubits or qcfg.get("max_qubits", 26))
    out = []

    # Calibration: the simulator's QFT equals the textbook DFT matrix on 3 qubits
    Q = 8
    v = np.random.default_rng(0).normal(size=Q) + 1j * np.random.default_rng(1).normal(size=Q)
    v /= np.linalg.norm(v)
    F = np.exp(-2j * np.pi * np.outer(np.arange(Q), np.arange(Q)) / Q) / math.sqrt(Q)
    qft_ok = np.allclose(np.fft.fft(v) / math.sqrt(Q), F @ v) and abs(np.linalg.norm(F @ v) - 1) < 1e-12
    out.append(Finding("quantum", "calibration: simulated QFT is unitary and matches the DFT matrix",
                       "established", f"3-qubit check on {where}: {'matches' if qft_ok else 'MISMATCH'}",
                       survived=qft_ok))

    # Shor's algorithm on small numbers
    rows = []
    for N in numbers:
        if (N - 1).bit_length() * 3 > max_qubits:
            continue
        rows.append(shor(N, xp))
    ok = all(r["factors"] and r["factors"][0] * r["factors"][1] == r["N"] for r in rows)
    out.append(Finding(
        "quantum", "Shor's algorithm on a simulated quantum computer", "established",
        "; ".join(f"{r['N']} = {r['factors'][0]} x {r['factors'][1]} (a = {r['a']}, period r = {r['r']}, "
                  f"{r['qubits']} qubits, {r['state_bytes'] / 2**20:.3g} MB, {r['shots']} shot(s))"
                  for r in rows if r["factors"]),
        survived=ok and bool(rows), details={"runs": rows, "backend": where}))

    # Threads across the mirror: what a split-and-stitch simulator would pay
    th = [threads_across(r["N"], r["a"]) for r in rows if r["factors"]]
    out.append(Finding(
        "quantum", "threads across the mirror: entanglement = period r", "established",
        "; ".join(f"{x['N']}: {x['mirror_threads']} threads (r = {x['r']}), mid-register cut "
                  f"{x['mid_counting_threads']}" for x in th)
        + ". Splitting the qubits costs one stored piece per thread, and across the mirror that "
          "is the period itself (~2^2048 for RSA-2048)",
        survived=bool(th) and all(x["mirror_threads"] == x["r"] for x in th), details={"cuts": th}))

    # The memory wall: each qubit doubles the state
    sizes = list(range(14, max_qubits + 1, 2))
    wall = memory_wall(xp, sizes)
    growth = [b2 / b1 for (_, b1, _), (_, b2, _) in zip(wall, wall[1:])]
    t_ratio = wall[-1][2] / wall[-2][2] if len(wall) > 1 else float("nan")
    rsa_qubits = 2 * 2048 + 3
    out.append(Finding(
        "quantum", "memory wall: every qubit doubles the classical cost", "established",
        f"{where}: {wall[0][0]} qubits {wall[0][1] / 2**20:.3g} MB -> {wall[-1][0]} qubits "
        f"{wall[-1][1] / 2**30:.3g} GB (x{growth[0]:.0f} per 2 qubits); full QFT time x{t_ratio:.1f} per 2 qubits. "
        f"Shor on RSA-2048 needs ~{rsa_qubits} qubits: about 10^{(rsa_qubits * math.log10(2) + math.log10(8)):.0f} "
        f"bytes as a state vector (the universe has ~10^80 atoms)",
        survived=all(abs(g - 4) < 1e-9 for g in growth),
        details={"rows": [{"qubits": n, "bytes": b, "seconds": round(s, 4)} for n, b, s in wall]}))
    return out


SKILL = Skill("quantum", "a classical simulator must store all 2^n amplitudes (no shortcut)",
              "Simulated quantum computer: Shor's algorithm on small N and the 2^n memory wall",
              run, needs_primes=False)
