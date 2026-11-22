import numpy as np

from ppa.core.ledger import Finding
from ppa.skills.base import Skill
from ppa.truth.randomness_tuple import RANDOMNESS_TUPLE, liouville_signs, sieve_of_randomness


def _finding(name, x, limit):
    z = sieve_of_randomness(x)
    bad = [r.name for r in RANDOMNESS_TUPLE if abs(z[r.name]) > limit]
    return Finding("randomness", name, "established",
                   "indistinguishable from random" if not bad else "not random: " + ", ".join(bad),
                   survived=not bad, details={"z": z, "n": int(len(x))})


def run(ctx, **_):
    lim = ctx.cfg["thresholds"]["randomness_z"]
    P = ctx.P_test[ctx.P_test > 3]
    return [
        _finding("randomness: prime mod-3 bucket signs", np.where(P % 3 == 1, 1, -1), lim),
        _finding("randomness: prime mod-4 signs", np.where(P % 4 == 1, 1, -1), lim),
        _finding("randomness: Liouville sign flips",
                 liouville_signs(min(ctx.test[1], 10 ** 7)), lim),
    ]


SKILL = Skill("randomness", "the randomness tuple (six rows, |z| <= 4)",
              "Which prime-derived sign sequences look like fair coin flips?", run)
