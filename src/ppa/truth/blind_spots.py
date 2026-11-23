"""
blind_spots.py
==============
Where the digital root goes blind, and how much of that mod 11 sees.

Why the digital root is blind: on the 9-position clock, 9 is a full lap,
so 9 (and 0) do nothing. Adding, dropping, or trading a 9 for a 0 leaves the
digital root unchanged. Adding also ignores order, so swaps are invisible.

Why mod 11 sees most of it: 10 = -1 on the 11-position clock, so digit
positions alternate +1, -1, +1, ... (the alternating digit sum).

  0 <-> 9 at one spot    value changes by 9 * 10^k = +-9 mod 11  -> never 0  -> always caught
  neighbours swapped     changes by (a-b) * 9 * 10^k = +-9(a-b)  -> never 0  -> always caught
  9 or 0 dropped/added   every digit to its left shifts one place and flips
                         sign, so the change is roughly random mod 11
                         -> missed about 1 time in 11 (caught ~91%)
  swap two places apart  both spots have the same sign -> invisible to mod 11 too
  (any same-parity swap)

Every one of these is invisible to the digital root, so whatever mod 11
catches here is pure gain from the second view.
"""
from __future__ import annotations

import random

BLIND_SPOTS = (
    # (name, what happens, mod 11 guarantee)
    ("drop9", "a 9 goes missing", "caught about 10 times in 11"),
    ("add9", "an extra 9 appears", "caught about 10 times in 11"),
    ("drop0", "a 0 goes missing", "caught about 10 times in 11"),
    ("add0", "an extra 0 appears", "caught about 10 times in 11"),
    ("zero_nine", "a 0 written as 9 or a 9 as 0", "always caught"),
    ("swap_next", "two neighbouring digits swapped", "always caught"),
    ("swap_far", "two digits swapped anywhere", "caught when the gap is odd"),
)


def mutate(s: str, kind: str, rng: random.Random) -> str | None:
    """Make one blind-spot mistake in the digit string s (None if impossible)."""
    if kind in ("drop9", "drop0"):
        d = kind[-1]
        idx = [i for i, c in enumerate(s) if c == d and not (i == 0 and len(s) > 1 and s[1] == "0")]
        if not idx:
            return None
        i = rng.choice(idx)
        return s[:i] + s[i + 1:]
    if kind in ("add9", "add0"):
        i = rng.randrange(1, len(s) + 1)
        return s[:i] + kind[-1] + s[i:]
    if kind == "zero_nine":
        idx = [i for i, c in enumerate(s) if c in "09" and i > 0]
        if not idx:
            return None
        i = rng.choice(idx)
        return s[:i] + ("9" if s[i] == "0" else "0") + s[i + 1:]
    if kind == "swap_next":
        i = rng.randrange(len(s) - 1)
        if s[i] == s[i + 1] or (i == 0 and s[1] == "0"):
            return None
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if kind == "swap_far":
        i, j = sorted(rng.sample(range(len(s)), 2))
        if s[i] == s[j] or (i == 0 and s[j] == "0"):
            return None
        t = list(s)
        t[i], t[j] = t[j], t[i]
        return "".join(t)
    raise ValueError(kind)


def alt_sum(n: int) -> int:
    """Alternating digit sum from the right (+ - + - ...), reduced mod 11."""
    return sum(int(d) * (1 if i % 2 == 0 else -1) for i, d in enumerate(reversed(str(n)))) % 11


EXAMPLE = 3952806
EXAMPLE_MISTAKES = (
    ("the 9 dropped", 352806),
    ("an extra 0", 39528006),
    ("0 written as 9", 3952896),
    ("9 and 5 swapped (neighbours)", 3592806),
    ("3 and 5 swapped (two apart)", 5932806),
)


def walkthrough(n: int = EXAMPLE, mistakes=EXAMPLE_MISTAKES):
    from ppa.truth.truth_tuple import digital_root
    print(f"{'number':>10}  dr  mod11   {'':28} verdict")
    print(f"{n:>10}  {digital_root(n):>2}  {alt_sum(n):>5}   {'original':28}")
    for label, w in mistakes:
        d_ok, m_ok = digital_root(w) == digital_root(n), alt_sum(w) == alt_sum(n)
        verdict = "both blind" if d_ok and m_ok else "mod 11 catches it" if d_ok else "dr catches it"
        print(f"{w:>10}  {digital_root(w):>2}  {alt_sum(w):>5}   {label:28} {verdict}")


def measure(trials: int = 20_000, seed: int = 20261123):
    """Rows of (name, description, dr_catch_%, mod11_catch_%)."""
    rng = random.Random(seed)
    rows = []
    for name, desc, _ in BLIND_SPOTS:
        n = dr = m11 = 0
        while n < trials:
            v = rng.randrange(10 ** 5, 10 ** 8)
            w = mutate(str(v), name, rng)
            if w is None or int(w) == v:
                continue
            diff = int(w) - v
            n += 1
            dr += diff % 9 != 0
            m11 += diff % 11 != 0
        rows.append((name, desc, 100 * dr / n, 100 * m11 / n))
    return rows


if __name__ == "__main__":
    walkthrough()
    print()
    for name, desc, dr, m in measure():
        print(f"{desc:34} dr {dr:5.1f}%   mod11 {m:5.1f}%")
