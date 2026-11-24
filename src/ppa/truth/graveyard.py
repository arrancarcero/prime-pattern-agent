"""The graveyard: a separate layer that remembers what was WRONG, so it is not tested again.

Two layers:
  GRAVEYARD (frozen tuple, fingerprinted like the truth tuple)
      Ideas we tested and buried by hand, each with the claim, the test that killed it,
      the evidence (numbers), what was tried, and the rule that lets it be dug up again.
  runs/graveyard.json (learned layer, written by the agent)
      Every planner hypothesis that failed its tests, with the range it was tried at.

Before the planner runs a hypothesis it asks check(). A buried idea is skipped, unless
something has changed that could change the answer: a bigger range, a method not yet
tried, or new data. "Wrong" can mean "not yet tested far enough", so every grave says how
to reopen it. Ideas proven impossible (by counting or by proof) have no reopen rule.

The graveyard is not a list of shame: half of these are the user's own ideas, and each one
taught something (the dead end is often where the real result was found).
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import namedtuple
from pathlib import Path

Grave = namedtuple("Grave", "key claim killed_by evidence tried reopen_if")
Check = namedtuple("Check", "status key reason")      # status: "buried" | "reopened" | "clear"

NEVER = {}    # reopen_if for ideas ruled out by proof or counting

GRAVEYARD: tuple[Grave, ...] = (
    Grave("transitions:beyond_gaps",
          "consecutive primes' residues (mod up to 72, lags 1-3) hold a pattern beyond what prime gaps explain",
          "transitions skill, stage B (gap-rule null)",
          "primes to 1e10: every pattern up to mod 72, lag 3 explained by gaps; no candidates",
          {"max_n": 1e10, "max_modulus": 72, "max_lag": 3},
          {"max_n_above": 1e10, "max_modulus_above": 72, "max_lag_above": 3}),
    Grave("transitions:fibonacci_24",
          "a window of exactly 24 primes correlates through the Fibonacci 24-cycle",
          "the researcher, 2026-11-11",
          "mod 24 is not special: its gates are a 3-bit XOR group like any (Z/24)*",
          {}, NEVER),
    Grave("digital_root:predict_positions",
          "digital roots alone predict where the next prime is",
          "dr_gap skill and the researcher",
          "replaced by: what is true of a prime's neighbours, given its position",
          {}, NEVER),
    Grave("digital_root:recover_number",
          "a 'which pairs lead to which' digital-root table recovers the original number",
          "counting; overflow trail (ppa trail)",
          "1,000,000 seven-digit numbers share each digital root; recovering the number needs the "
          "full trail of roots and laps, which is as long as the number itself",
          {}, NEVER),
    Grave("zeta:negative_primes_one_third_line",
          "a 'negative primes' zeta, with a sign flip, has its zeros on the 1/3 line",
          "zeta walks, Layer 9 (2026-11-22)",
          "the sign-flip walks did not come home on the 1/3 line; single-gate (Hurwitz-type) "
          "functions have zeros that wander off any one line (Davenport-Heilbronn)",
          {"method": ("eta walk", "single-gate walk")},
          {"method_not_in": ("eta walk", "single-gate walk")}),
    Grave("abundance:prime_Z",
          "elements with a prime number of protons are more abundant in the universe",
          "examples/element_abundance_primes.py (Lodders 2010 solar system table)",
          "x4.9 (p = 0.04) only because of N (next to spallation-depleted B) and F (burned in stars); "
          "Z >= 11: p = 0.16; past the iron peak: p = 0.62",
          {"data": ("Lodders 2010 solar system",)},
          {"data_not_in": ("Lodders 2010 solar system",)}),
    Grave("factoring:moire_new_reach",
          "the moire beat of two grids factors numbers that Fermat's method cannot",
          "tool tuple twin check",
          "same answer as Fermat on 200/200 numbers: the beat gap d = q - p is Fermat's 2b; "
          "inverting the grids leaves the beat unchanged",
          {}, NEVER),
    Grave("factoring:zone_plate_new_reach",
          "zone-plate aliasing factors numbers that trial division cannot",
          "tool tuple twin check",
          "same answer as trial division on 200/200 numbers: a stride that aliases is a divisor",
          {}, NEVER),
    Grave("factoring:sign_fft_beats_trial",
          "sign flips + FFT factor faster than trial division",
          "sign_fft tool timing",
          "it works (77 -> 7, 1,022,117 -> 1009) but the pattern is N long: 15,001,999 took 37 s "
          "against ~3,000 trial divisions; only a quantum computer does this in ~log N qubits",
          {"hardware": ("classical",)},
          {"hardware_not_in": ("classical",)}),
    Grave("quantum:mirror_split_beats_wall",
          "splitting the qubits and filling in a gradient across the mirror avoids the memory wall",
          "quantum skill, threads across the mirror",
          "threads across the cut = the period r exactly (15: 4, 21: 6, 35: 12, 143: 60); "
          "for RSA-2048, r ~ 2^2048",
          {}, NEVER),
    Grave("quantum:time_encoding_beats_wall",
          "encoding part of the qubit state in processing time avoids the memory wall",
          "compute-on-demand test (N = 143)",
          "memory 897 MB -> 1 MB, but one probability took 0.42 s, so all 65,536 take ~7.6 hours: "
          "the cost moves from memory into time; Holevo: n qubits give at most n bits when read",
          {}, NEVER),
    Grave("quantum:mirrored_memory_vm",
          "a PC with a virtual machine and 3-way mirrored memory can do what a quantum computer does",
          "the memory wall (quantum skill)",
          "mirroring stores the same data 3 times (capacity / 3); a VM shares the same hardware; "
          "the state needs 2^n numbers (30 qubits = 16 GB, RSA-2048 ~ 10^1235 bytes)",
          {}, NEVER),
    Grave("crypto:precompute_rsa",
          "precomputing now breaks RSA keys later, like Logjam broke shared Diffie-Hellman primes",
          "reasoning (Logjam, Adrian et al. 2015)",
          "precomputation pays only for shared parameters; every RSA key has its own N",
          {}, NEVER),
    Grave("randomness:tuple_means_secure",
          "a generator that passes the randomness tuple is safe for keys and salts",
          "examples/pss_demo.py",
          "Mersenne Twister passes every row but is predictable: the tuple is necessary, not sufficient",
          {}, NEVER),
)

FINGERPRINT = hashlib.sha256(repr(GRAVEYARD).encode()).hexdigest()[:16]
_BY_KEY = {g.key: g for g in GRAVEYARD}


def _reopens(rule: dict, ctx: dict) -> str | None:
    """Return the reason an idea may be dug up, or None if it stays buried."""
    for k, v in rule.items():
        if k.endswith("_above"):
            field = k[: -len("_above")]
            if ctx.get(field) is not None and ctx[field] > v:
                return f"{field} = {ctx[field]:g} is beyond {v:g}"
        elif k.endswith("_not_in"):
            field = k[: -len("_not_in")]
            if ctx.get(field) is not None and ctx[field] not in v:
                return f"new {field}: {ctx[field]}"
    return None


def hypothesis_key(skill: str, params: dict) -> str:
    return f"{skill}:" + ",".join(f"{k}={params[k]}" for k in sorted(params))


class Graveyard:
    """Frozen graves + the learned layer in runs/graveyard.json."""

    def __init__(self, runs_dir: str | Path = "runs"):
        self.path = Path(runs_dir) / "graveyard.json"
        self.learned = json.loads(self.path.read_text()) if self.path.exists() else []

    def check(self, key: str, **ctx) -> Check:
        # 1. a frozen grave for this exact idea
        g = _BY_KEY.get(key)
        if g is not None:
            why = _reopens(g.reopen_if, ctx)
            return Check("reopened", key, why) if why else Check("buried", key, g.evidence)
        # 2. planner transitions fall under the frozen 'beyond gaps' grave
        if key.startswith("transitions:M="):
            p = dict(kv.split("=") for kv in key.split(":", 1)[1].split(","))
            gg = _BY_KEY["transitions:beyond_gaps"]
            covered = int(p["M"]) <= gg.tried["max_modulus"] and int(p["lag"]) <= gg.tried["max_lag"]
            if covered:
                why = _reopens({"max_n_above": gg.tried["max_n"]}, ctx)
                learned_fail = any(e["key"] == key and ctx.get("max_n", 0) <= e["max_n"] for e in self.learned)
                if not why:
                    return Check("buried", key, gg.evidence)
                if not learned_fail:
                    return Check("reopened", key, why)
        # 3. the learned layer: same hypothesis failed before at this range or larger
        for e in self.learned:
            if e["key"] == key:
                if ctx.get("max_n", 0) > e["max_n"]:
                    return Check("reopened", key, f"max_n = {ctx['max_n']:g} beyond the {e['max_n']:g} it failed at")
                return Check("buried", key, f"failed at max_n = {e['max_n']:g}: {e['summary']}")
        return Check("clear", key, "")

    def bury(self, key: str, max_n: float, summary: str) -> None:
        self.learned = [e for e in self.learned if e["key"] != key]
        self.learned.append({"key": key, "max_n": max_n, "summary": summary[:300],
                             "buried": time.strftime("%Y-%m-%d %H:%M")})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.learned, indent=1))
