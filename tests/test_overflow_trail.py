import random

from ppa.truth.overflow_trail import (count_same_root, count_same_roots, count_same_sum,
                                      digit_sum_from_laps, ladder, laps, rebuild, trail)


def test_users_number():
    t = trail(3952806)
    assert [s.total for s in t] == [6, 6, 14, 7, 12, 12, 6]
    assert laps(3952806) == 3 and digit_sum_from_laps(3952806) == 33
    assert [c for _, c in ladder(3952806)] == [1_000_000, 464_653, 41_057, 4, 1]


def test_rebuild_round_trip():
    rng = random.Random(5)
    for _ in range(5000):
        n = rng.randrange(1, 10**12)
        t = trail(n)
        assert rebuild(t) == n
        assert digit_sum_from_laps(n) == sum(map(int, str(n)))


def test_nine_always_laps_zero_never_does():
    for r_start in range(1, 10):
        n = int(f"9{r_start}")      # read right to left: r_start first, then 9
        assert trail(n)[1].overflow
        n = int(f"10{r_start}")     # then 0, then 1
        assert not trail(n)[1].overflow


def test_counts_match_brute_force():
    k = 4
    nums = range(10**(k - 1), 10**k)
    assert count_same_root(k, 6) == sum(1 for n in nums if (n - 1) % 9 + 1 == 6)
    assert count_same_sum(k, 20) == sum(1 for n in nums if sum(map(int, str(n))) == 20)
    roots = tuple(s.root for s in trail(3906))
    assert count_same_roots(k, roots) == sum(
        1 for n in nums if tuple(s.root for s in trail(n)) == roots)
