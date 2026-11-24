"""The agent loop: calibrate -> fixed skills -> explore (plan, test, learn) -> ledger."""
from __future__ import annotations

import time
from collections import Counter

from ppa.agent.planner import Planner
from ppa.core.ledger import Ledger
from ppa.core.sieve import primes_between
from ppa.skills import REGISTRY
from ppa.skills.base import Context

FIXED = ("dr_gap", "fano", "spectrum", "gates", "randomness", "nature", "curves", "factoring", "candles", "spacing", "walk", "quantum")


def build_context(cfg: dict) -> Context:
    max_n = cfg["range"]["max_n"]
    train = (max_n // 10, max_n // 2)
    test = (max_n // 2, max_n)
    t = time.time()
    P_train, P_test = primes_between(*train), primes_between(*test)
    print(f"sieved {len(P_train) + len(P_test):,} primes in {time.time() - t:.1f}s "
          f"(train {train[0]:.0e}-{train[1]:.0e}, test {test[0]:.0e}-{test[1]:.0e})")
    return Context(cfg, train, test, P_train, P_test)


def run_agent(cfg: dict, intent: str = "all") -> Ledger:
    ledger = Ledger(cfg["output"]["runs_dir"], {"config": cfg, "intent": intent,
                                                "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    needs_primes = intent not in ("nature", "curves", "factoring", "candles", "spacing", "walk", "quantum")
    ctx = build_context(cfg) if needs_primes else Context(cfg, (0, 0), (0, 0), None, None)

    if needs_primes:  # contract rule 1
        print("calibrate")
        cal = ledger.add(REGISTRY["calibrate"].run(ctx)[0])
        if cal.survived is False:
            ledger.save()
            raise SystemExit("Sieve disagrees with the truth tuple. Stopping.")

    if intent == "all":
        fixed = FIXED
    elif intent in REGISTRY and intent not in ("transitions", "calibrate"):
        fixed = (intent,)
    else:
        fixed = ()
    for name in fixed:
        print(name)
        for f in REGISTRY[name].run(ctx):
            ledger.add(f)

    if intent in ("all", "explore", "transitions"):
        print("explore")
        planner = Planner(cfg)
        for _ in range(cfg["planner"]["max_steps"]):
            h = planner.next()
            if h is None:
                break
            f = REGISTRY[h.skill].run(ctx, **h.params)[0]
            f.details["planner_reason"] = h.reason
            ledger.add(f)
            planner.learn(h, f)

    counts = Counter(f.label for f in ledger.findings)
    print("summary:", dict(counts))
    cands = [f.name for f in ledger.findings if f.label == "candidate"]
    print("candidates for a human to check:", cands or "none")
    print("ledger written to", ledger.save())
    return ledger
