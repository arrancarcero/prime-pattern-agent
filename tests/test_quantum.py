from ppa.skills.base import Context
from ppa.skills.quantum import SKILL, order_finding_state, shor


def test_shor_small():
    r = shor(15)
    assert sorted(r["factors"]) == [3, 5] and r["r"] == 4 and r["qubits"] == 12
    probs, t, L = order_finding_state(21, 2)
    Q = 1 << t
    assert abs(probs.sum() - 1) < 1e-9                       # unitary: probabilities add to 1
    top = sorted(range(Q), key=lambda k: -probs[k])[:6]
    assert all(min(abs(k - s * Q / 6) for s in range(7)) < 1 for k in top)   # peaks at Q/6 multiples


def test_quantum_skill():
    cal, sh, wall = SKILL.run(Context({}, (0, 0), (0, 0), None, None), numbers=(15, 21), max_qubits=20)
    assert cal.survived and sh.survived and wall.survived
