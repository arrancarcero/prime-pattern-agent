"""The tool tuple: the factoring toolbox frozen like the truth tuple.

A tuple can hold functions. Frozen, no tool can be swapped during a run. The fingerprint
hashes each tool's source code, so an edited or stubbed tool changes it between runs.
Every tool must reproduce a known answer and must FAIL on a prime before it is trusted.
"""
import hashlib
import inspect
from collections import namedtuple

from ppa.core.factoring import ecm, fermat, moire, p_minus_1, rho, trial, zone_plate

Tool = namedtuple("Tool", "name fn catches budget")

TOOL_TUPLE: tuple = (
    Tool("trial",     trial,     "a small factor",                    20_000),
    Tool("fermat",    fermat,    "two factors close together",        20_000),
    Tool("rho",       rho,       "any factor up to ~1e12 (birthday)", 20_000),
    Tool("p_minus_1", p_minus_1, "p - 1 built from small primes",      2_000),
    Tool("ecm",       ecm,       "some curve size p+1-a_p is smooth",  4_000),
    Tool("moire",     moire,     "small beat gap q - p (Fermat in costume)",   20_000),
    Tool("zone_plate", zone_plate, "a small factor, seen as aliasing (trial in costume)", 20_000),
)

CALIBRATION: tuple = (
    ("trial",     1009 * 1000000000039, 1009),
    ("fermat",    10007 * 10009,        10007),
    ("rho",       1000003 * 1000033,    None),   # None: any proper factor is accepted
    ("p_minus_1", 2311 * 1000000007,    None),   # 2310 = 2*3*5*7*11
    ("ecm",       1000003 * 1000033,    None),
    ("moire",     10007 * 10009,        10007),
    ("zone_plate", 1009 * 1000000000039, 1009),
)

# A new tool that claims to be an old method in disguise must give the same answers.
TWINS: tuple = (("moire", "fermat"), ("zone_plate", "trial"))
NULL_PRIME = 1_000_000_000_000_037


def fingerprint(tools=TOOL_TUPLE) -> str:
    h = hashlib.sha256()
    for t in tools:
        h.update(t.name.encode())
        h.update(inspect.getsource(t.fn).encode())
    return h.hexdigest()[:16]


def verify_tools(tools=TOOL_TUPLE):
    """Return (name, known_answer_ok, null_prime_ok) for every tool."""
    out = []
    for name, N, expect in CALIBRATION:
        t = next(x for x in tools if x.name == name)
        f, _ = t.fn(N, t.budget)
        ok = f is not None and 1 < f < N and N % f == 0 and (expect is None or f in (expect, N // expect))
        null_f, _ = t.fn(NULL_PRIME, t.budget)
        out.append((name, ok, null_f is None))
    return out


def twin_numbers(count=200, seed=11):
    """Test numbers both twins can reach: close-factor and small-factor semiprimes."""
    import random
    from ppa.core.factoring import PRIMES, next_prime
    rng = random.Random(seed)
    out = []
    for _ in range(count // 2):
        p = next_prime(rng.randrange(10 ** 8, 10 ** 12))
        out.append(p * next_prime(p + 2 * rng.randrange(1, 15_000)))       # beat gap < 30,000
        out.append(PRIMES[rng.randrange(1, 15_000)] * next_prime(rng.randrange(10 ** 11, 10 ** 13)))
    return out


def verify_twins(tools=TOOL_TUPLE, numbers=None):
    """Return (new, old, agreements, tried): both must find the same factor, or both fail."""
    numbers = numbers or twin_numbers()
    by = {t.name: t for t in tools}
    out = []
    for new, old in TWINS:
        agree = 0
        for N in numbers:
            a, _ = by[new].fn(N, by[new].budget)
            b, _ = by[old].fn(N, by[old].budget)
            same = (a is None and b is None) or (a and b and {a, N // a} == {b, N // b})
            agree += bool(same)
        out.append((new, old, agree, len(numbers)))
    return out
