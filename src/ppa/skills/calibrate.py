from ppa.core.ledger import Finding
from ppa.core.sieve import primes_between
from ppa.skills.base import Skill
from ppa.truth import truth_tuple


def run(ctx, **_):
    max_n = ctx.test[1]
    total = len(primes_between(0, ctx.train[0])) + len(ctx.P_train) + len(ctx.P_test)
    v = truth_tuple.verify("pi", max_n, total)
    label = "established" if v.ok else "check"
    note = "no known value for this max_n" if v.ok is None else ("matches" if v.ok else v.note)
    return [Finding("calibrate", f"pi({max_n:.0e}) = {total:,}", label,
                    f"truth tuple {truth_tuple.FINGERPRINT}: {note}",
                    survived=v.ok, details={"fingerprint": truth_tuple.FINGERPRINT})]


SKILL = Skill("calibrate", "truth tuple (exact prime counts)",
              "Sieve must reproduce pi(max_n) before anything else runs.", run)
