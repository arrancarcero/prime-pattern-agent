"""Factoring toolbox skill: calibrate every tool, then run them all on numbers built
with different weaknesses, and report which pattern each tool catches."""
import random

from ppa.core.factoring import is_prime, next_prime, smooth_minus_one_prime, hardened_prime
from ppa.core.ledger import Finding
from ppa.skills.base import Skill
from ppa.truth.tool_tuple import TOOL_TUPLE, fingerprint, verify_tools


def _cases(rng):
    p = next_prime(rng.getrandbits(26) | (1 << 25))
    yield "small factor", 1009, next_prime(10 ** 12 + 39)
    yield "factors close together", p, next_prime(p + 2000)
    yield "p - 1 is smooth", smooth_minus_one_prime(rng, 26), next_prime(rng.getrandbits(26) | (1 << 25))
    yield "random 26-bit factors", next_prime(rng.getrandbits(26) | (1 << 25)), next_prime(rng.getrandbits(26) | (1 << 25))
    yield "hardened 34-bit factors", hardened_prime(rng, 34), hardened_prime(rng, 34)


def run(ctx, **_):
    fp = fingerprint()
    out = []
    checks = verify_tools()
    for name, ok, null_ok in checks:
        out.append(Finding("factoring", f"calibrate tool: {name}", "established" if ok and null_ok else "check",
                           f"known answer {'passes' if ok else 'FAILS'}; null prime {'passes' if null_ok else 'FAILS'}",
                           survived=ok and null_ok, details={"tool_fingerprint": fp}))
    if not all(ok and null_ok for _, ok, null_ok in checks):
        return out                                    # never run untrusted tools
    rng = random.Random(ctx.cfg["null_model"]["seed"])
    for label, a, b in _cases(rng):
        N = a * b
        caught = []
        for t in TOOL_TUPLE:
            f, steps = t.fn(N, t.budget)
            if f is not None and 1 < f < N and N % f == 0:
                caught.append(f"{t.name} ({steps:,} steps)")
        out.append(Finding("factoring", f"toolbox on: {label}", "established",
                           f"N = {N:,}; caught by: {', '.join(caught) or 'none within budget'}",
                           survived=bool(caught), details={"N": N, "caught_by": caught}))
    return out


SKILL = Skill("factoring", "known answers + a prime that every tool must fail on",
              "Which factoring sieve catches which kind of hidden structure?", run, needs_primes=False)
