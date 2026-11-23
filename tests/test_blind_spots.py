import random

from ppa.truth.blind_spots import BLIND_SPOTS, alt_sum, measure, mutate
from ppa.truth.truth_tuple import digital_root


def test_alt_sum_is_mod_11():
    rng = random.Random(3)
    for _ in range(5000):
        n = rng.randrange(1, 10**12)
        assert alt_sum(n) == n % 11


def test_digital_root_blind_to_every_spot():
    rng = random.Random(4)
    for name, _, _ in BLIND_SPOTS:
        for _ in range(500):
            v = rng.randrange(10**5, 10**8)
            w = mutate(str(v), name, rng)
            if w:
                assert digital_root(int(w)) == digital_root(v)


def test_mod11_guarantees():
    rows = {r[0]: r for r in measure(trials=3000)}
    assert rows["zero_nine"][3] == 100.0
    assert rows["swap_next"][3] == 100.0
    for k in ("drop9", "add9", "drop0", "add0"):
        assert 85 < rows[k][3] < 97          # ~10 in 11
    assert 40 < rows["swap_far"][3] < 70     # odd gaps only
