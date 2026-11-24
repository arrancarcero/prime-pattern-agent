from ppa.skills.base import Context
from ppa.skills.spacing import SKILL, hardy_z, zeros_up_to


def test_first_zero_and_sign_change():
    g = zeros_up_to(60)
    assert abs(g[0] - 14.1347) < 0.01 and len(g) == 13     # 13 zeros below height 60
    assert hardy_z(14.0)[0] * hardy_z(14.3)[0] < 0


def test_spacing_prefers_gue():
    cal, law = SKILL.run(Context({}, (0, 0), (0, 0), None, None), T=2000)
    assert cal.survived, cal.summary
    assert law.survived and law.details["ks"]["GUE"] < law.details["ks"]["GOE"], law.summary
