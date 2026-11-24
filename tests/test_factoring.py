from ppa.truth.tool_tuple import TOOL_TUPLE, Tool, fingerprint, verify_tools


def test_every_tool_calibrates_and_fails_on_a_prime():
    assert all(ok and null_ok for _, ok, null_ok in verify_tools())


def test_a_stubbed_tool_is_caught():
    def stub(N, budget, seed=1):
        return 3, 1          # always claims success, like a stubbed compiler
    stubbed = tuple(Tool(t.name, stub if t.name == "rho" else t.fn, t.catches, t.budget) for t in TOOL_TUPLE)
    assert fingerprint(stubbed) != fingerprint()
    rho_check = next(c for c in verify_tools(stubbed) if c[0] == "rho")
    assert rho_check == ("rho", False, False)


def test_moire_and_zone_plate_match_their_twins():
    from ppa.core.factoring import moire, zone_plate
    from ppa.truth.tool_tuple import verify_twins
    assert moire(29 * 31, 10)[0] == 29                 # beat gap 2 -> found at once
    assert zone_plate(7 * 1000003, 10)[0] == 7
    for new, old, agree, n in verify_twins(numbers=None):
        assert agree == n, f"{new} disagrees with {old}"


def test_sign_fft_dark_lines():
    from ppa.core.factoring import jacobi_all, sign_fft
    import numpy as np
    J = jacobi_all(77)
    assert J[2] == -1 and J[7] == 0 and J[4] == 1          # (2/77) = (2/7)(2/11) = 1 * -1
    mag = np.abs(np.fft.fft(J.astype(float)))
    assert np.allclose(mag[[1, 2, 3]], 77 ** 0.5)          # flat glow
    assert mag[7] < 1e-9 and mag[11] < 1e-9                # dark at the factors
    assert sign_fft(1009 * 1013, 2_000_000)[0] == 1009
    assert sign_fft(1_000_003, 2_000_000)[0] is None       # a prime has no dark lines


def test_quadratic_sieve_reaches_past_rho():
    from ppa.core.factoring import _sqrt_mod, quadratic_sieve
    assert _sqrt_mod(10, 13) ** 2 % 13 == 10
    p, q = 759329574457, 1000000000039                  # 40-bit factors, 24-digit N
    f, _ = quadratic_sieve(p * q, 5_000_000)
    assert f in (p, q)
    assert quadratic_sieve(1_000_000_000_000_037, 100_000)[0] is None   # a prime: no factor
