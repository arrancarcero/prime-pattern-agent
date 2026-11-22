from ppa.truth import truth_tuple
from ppa.truth.randomness_tuple import liouville_signs, sieve_of_randomness, true_random_signs


def test_truth_tuple_self_test_to_1e7():
    assert all(v.ok is not False for v in truth_tuple.self_test(7))


def test_truth_tuple_is_frozen():
    assert isinstance(truth_tuple.TRUTH, tuple)
    assert truth_tuple.verify("pi", 10 ** 8, 5761455).ok
    assert truth_tuple.verify("pi", 10 ** 8, 5761456).ok is False


def test_randomness_tuple_controls():
    assert all(abs(z) < 5 for z in sieve_of_randomness(true_random_signs(10 ** 6)).values())
    assert all(abs(z) < 5 for z in sieve_of_randomness(liouville_signs(10 ** 6)).values())
