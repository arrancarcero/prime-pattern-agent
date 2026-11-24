import numpy as np

from ppa.skills.base import Context
from ppa.skills.twins import SKILL, admissible_residues, lower_twins, sieved_null


def test_wheel_and_counts():
    assert len(admissible_residues()) == 22275            # prod (q - 2), q = 3..17
    T = lower_twins(10 ** 6)
    assert len(T) + 4 == 8169
    assert np.all(T % 6 == 5)                              # (6k - 1, 6k + 1)


def test_sieved_null_matches_density_and_runs():
    T = lower_twins(10 ** 7)
    N = sieved_null(T, 10 ** 7, 97, 0)
    assert abs(len(N) / len(T) - 1) < 0.02
    out = SKILL.run(Context({}, (0, 0), (0, 0), None, None), max_x=10 ** 7, seeds=3)
    assert out[0].survived and len(out) == 6
