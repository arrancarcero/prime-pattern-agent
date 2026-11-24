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
