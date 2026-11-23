import random

from ppa.truth.truth_tuple import digital_root, dr_pairs, precheck, verify_calc


def test_users_example():
    assert digital_root(3952806) == 6
    assert (1, 5) in dr_pairs(6) and (7, 8) in dr_pairs(6) and len(dr_pairs(6)) == 9


def test_correct_answers_pass():
    rng = random.Random(1)
    for _ in range(2000):
        a, b = rng.randrange(10**4, 10**7), rng.randrange(10**4, 10**7)
        for op, v in (("*", a * b), ("+", a + b), ("-", a - b)):
            assert precheck(op, a, b, v).passed
            assert verify_calc(op, a, b, v).ok


def test_one_wrong_digit_always_caught():
    rng = random.Random(2)
    for _ in range(2000):
        a, b = rng.randrange(10**4, 10**7), rng.randrange(10**4, 10**7)
        s = list(str(a * b))
        i = rng.randrange(len(s))
        s[i] = rng.choice([d for d in "0123456789" if d != s[i] and not (i == 0 and d == "0")])
        assert not precheck("*", a, b, int("".join(s))).passed


def test_swap_missed_by_dr_but_exact_still_catches():
    a, b = 1234, 5679            # 7007886 -> swap digits two apart (same parity spot)
    t = str(a * b)
    wrong = int(t[0] + t[3] + t[2] + t[1] + t[4:])   # swap positions 1 and 3
    pc = precheck("*", a, b, wrong)
    assert "digital_root" not in pc.failed_by      # dr is blind to order
    assert verify_calc("*", a, b, wrong).ok is False


def test_huge_power_precheck():
    v = verify_calc("**", 7, 500, 7**500 + 9)        # off by 9: dr blind, mod 11 sees it
    assert v.ok is False and "mod11" in v.note
