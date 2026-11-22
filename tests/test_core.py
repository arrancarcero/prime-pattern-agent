import numpy as np

from ppa.core.nulls import cramer_wheel
from ppa.core.projections import GATES_24, dr, gate_bits, mirror, wheel
from ppa.core.sieve import primes_between


def test_sieve_matches_known_counts():
    assert len(primes_between(0, 10 ** 6)) == 78498
    assert len(primes_between(0, 1000)) + len(primes_between(1000, 10 ** 5)) == 9592


def test_digital_root_and_mirror():
    assert dr(123456789) == 9 and dr(10) == 1
    assert all(dr((9 - a) * b) in (9, 9 - dr(a * b)) for a in range(1, 9) for b in range(1, 10))
    assert mirror(2) == 7 and mirror(3, 24) == 21


def test_gates_are_three_bits():
    assert GATES_24 == wheel(24)
    for a in GATES_24:
        for b in GATES_24:
            assert gate_bits(a * b % 24) == gate_bits(a) ^ gate_bits(b)


def test_null_model_is_sorted_and_on_the_wheel():
    N = cramer_wheel(10 ** 7, 3 * 10 ** 7, 210, seed=1)
    assert np.all(np.diff(N) > 0), "null model must be sorted with no repeats"
    assert np.all(np.gcd(N, 210) == 1), "null model must stay on the 210-wheel"
    P = primes_between(10 ** 7, 3 * 10 ** 7)
    assert abs(len(N) / len(P) - 1) < 0.01, "null model density must match primes"


def test_curve_point_counts_and_hasse_bound():
    from ppa.skills.curves import a_p
    # y^2 = x^3 + x + 1 over F_7 has 4 affine points + infinity = 5, so a_7 = 7 + 1 - 5 = 3
    assert a_p(1, 1, 7) == 3
    for p in (101, 1009, 10007):
        assert abs(a_p(1, 1, p)) <= 2 * p ** 0.5
    # complex multiplication: y^2 = x^3 - x has a_p = 0 when p = 3 mod 4
    assert all(a_p(-1, 0, p) == 0 for p in (7, 11, 19, 23, 31, 43))
