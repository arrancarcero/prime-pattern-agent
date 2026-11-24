from ppa.skills.base import Context
from ppa.skills.candles import SKILL


def test_candles_small_range():
    ctx = Context({}, (0, 0), (0, 0), None, None)
    cal, red, bright = SKILL.run(ctx, max_x=10**7, tol=0.05)
    assert cal.survived, cal.summary          # planted beta=0.6 zero is recovered
    assert red.survived, red.summary          # lines sit on the zero heights
    assert bright.survived, bright.summary    # every real zero reads as beta ~ 1/2
