"""
overflow_trail.py
=================
The user's idea: when adding digits for a digital root, the steps that go past 9
(like 14 -> 5 and 12 -> 3) leave extra information behind. Keep it, and you can
recover far more than the root, up to the original number in its original order.

The running root lives on a 9-position clock. Each step adds one digit:
    r_new = r + d,  and if that passes 9, take 9 off  -> one "lap" (overflow)
Keep the trail of (digit position, running root, overflow?) for every step.

Why the trail rebuilds the number:
    d = r_new - r + 9 * overflow
Each digit is the difference between neighbouring roots, plus 9 if a lap happened.
The overflow flag fixes the digital root's blind spot: once the running root is at
least 1, adding 0 never laps, but adding 9 always does. So 0 and 9 come apart.

The information ladder (how much each piece pins down):
    root only        -> 1 of 9 groups         (1,000,000 seven-digit numbers each)
    root + laps      -> the exact digit sum   (root + 9 * laps)
    flags + root     -> where the laps happened, which narrows order
    every root       -> only 0-vs-9 left open at each step where the root stood still
    full trail       -> exactly one number
The full trail is as long as the number itself, so it's another way of writing
the number, not a shortcut. The order lives in where the laps happen.
"""
from __future__ import annotations

from collections import namedtuple
from functools import lru_cache

Step = namedtuple("Step", "digit total root overflow")


def trail(n: int, right_to_left: bool = True) -> list[Step]:
    """Running digital root, one digit at a time. Default reads from the right,
    the way the user worked 3952806 (6, 0, 8, 2, 5, 9, 3)."""
    digits = [int(c) for c in str(abs(int(n)))]
    if right_to_left:
        digits.reverse()
    r, out = 0, []
    for d in digits:
        total = r + d
        lap = total > 9
        r = total - 9 if lap else total
        out.append(Step(d, total, r, lap))
    return out


def rebuild(steps, right_to_left: bool = True) -> int:
    """Recover the number from the roots and overflow flags alone (digits unused)."""
    r, digits = 0, []
    for s in steps:
        digits.append(s.root - r + 9 * s.overflow)
        r = s.root
    if right_to_left:
        digits.reverse()
    return int("".join(map(str, digits)))


def laps(n: int) -> int:
    return sum(s.overflow for s in trail(n))


def digit_sum_from_laps(n: int) -> int:
    """Final root + 9 per lap = the true digit sum (e.g. 6 + 3*9 = 33)."""
    t = trail(n)
    return t[-1].root + 9 * sum(s.overflow for s in t)


# ---------------------------------------------------------------------------
# Counting: how many k-digit numbers share each level of information
# ---------------------------------------------------------------------------
def _first_digit_ok(digits_right_to_left_count: int, pos: int, d: int) -> bool:
    return not (pos == digits_right_to_left_count - 1 and d == 0)  # no leading zero


def count_same_root(k: int, root: int) -> int:
    @lru_cache(None)
    def go(pos, r):
        if pos == k:
            return int(r == root)
        return sum(go(pos + 1, (r + d - 1) % 9 + 1 if r + d else 0)
                   for d in range(10) if _first_digit_ok(k, pos, d))
    return go(0, 0)


def count_same_sum(k: int, digit_sum: int) -> int:
    @lru_cache(None)
    def go(pos, s):
        if s > digit_sum:
            return 0
        if pos == k:
            return int(s == digit_sum)
        return sum(go(pos + 1, s + d) for d in range(10) if _first_digit_ok(k, pos, d))
    return go(0, 0)


def count_same_flags(k: int, flags: tuple[bool, ...], root: int) -> int:
    """k-digit numbers whose trail has the same overflow pattern and final root."""
    @lru_cache(None)
    def go(pos, r):
        if pos == k:
            return int(r == root)
        c = 0
        for d in range(10):
            if not _first_digit_ok(k, pos, d):
                continue
            total = r + d
            lap = total > 9
            if lap == flags[pos]:
                c += go(pos + 1, total - 9 if lap else total)
        return c
    return go(0, 0)


def count_same_roots(k: int, roots: tuple[int, ...]) -> int:
    """k-digit numbers with the same running roots at every step, lap flags unknown.
    Only 0 vs 9 is ambiguous (both leave the root where it was)."""
    c, r = 1, 0
    for pos, nr in enumerate(roots):
        c *= sum(1 for d in range(10) if _first_digit_ok(k, pos, d)
                 and ((r + d - 9) if r + d > 9 else r + d) == nr)
        r = nr
    return c


def ladder(n: int):
    """Rows of (what you keep, how many numbers of the same length match)."""
    t = trail(n)
    k = len(t)
    root = t[-1].root
    return (
        ("digital root only", count_same_root(k, root)),
        ("root + lap count (= digit sum)", count_same_sum(k, digit_sum_from_laps(n))),
        ("root + where the laps happened", count_same_flags(k, tuple(s.overflow for s in t), root)),
        ("every running root, no lap flags", count_same_roots(k, tuple(s.root for s in t))),
        ("full trail (roots + laps)", 1),
    )


def show(n: int):
    t = trail(n)
    print(f"Trail for {n} (read right to left):")
    print(f"  {'add':>4} {'total':>6} {'root':>5}  lap?")
    for s in t:
        print(f"  {s.digit:>4} {s.total:>6} {s.root:>5}  {'yes (one lap of 9)' if s.overflow else ''}")
    L = sum(s.overflow for s in t)
    print(f"Laps: {L}   digit sum = {t[-1].root} + {L} x 9 = {digit_sum_from_laps(n)}")
    print(f"Rebuilt from roots + lap flags alone: {rebuild(t)}")
    print(f"\nHow many {len(t)}-digit numbers match each level of information:")
    for what, c in ladder(n):
        print(f"  {what:34} {c:>10,}")


if __name__ == "__main__":
    import sys
    show(int(sys.argv[1]) if len(sys.argv) > 1 else 3952806)
