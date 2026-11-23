# Prime-Pattern Agent

A research agent that looks for patterns in primes and in natural measurements, and only keeps a pattern if it:

1. beats a named null model,
2. survives on a held-out range it was never fitted on,
3. passes the mirror check (reversed and reflected), and
4. is not already in the known-results tuple.

Companion document: **The Prime-Pattern Map** (every idea, why it works, and how it connects).

## Install (Windows PowerShell, conda base)

```powershell
cd prime-pattern-agent
python -m pip install -e ".[nature,dev]"
```

`numpy` is the only required package. `[nature]` adds statsmodels, scipy and pooch for the
natural-measurement skill. `[dev]` adds pytest.

## Run

```powershell
ppa truth                         # recheck the truth tuple (about 20 s at --exp 9)
ppa skills                        # list skills and the null model each one must beat
ppa contract                      # print the rules every run follows
ppa tools                         # fingerprint and calibrate the factoring tool tuple
ppa check "1234*5679=7007886"     # fast digital-root + mod 11 check, then exact
ppa blindspots                    # where the digital root is blind and what mod 11 sees
ppa run                           # everything, using agent.toml
ppa run --intent explore          # only the self-directed exploration loop
ppa run --intent nature           # only natural measurements
ppa run --intent curves           # Hasse-Weil bound and Sato-Tate angles on elliptic curves
ppa run --intent factoring        # the factoring toolbox, calibrated first
ppa run --max 1e10 --steps 60     # bigger range, longer exploration
python -m pytest                  # 11 tests: core, truth, agent, factoring
```

If `ppa` is not found, use `python -m ppa run ...` instead.
Each run writes `runs/ledger-YYYYMMDD-HHMMSS.json`.

## Folder structure

```
prime-pattern-agent/
├── agent.toml               settings: range, thresholds, planner budget, null model
├── CONTRACT.md              the 10 rules every run obeys
├── pyproject.toml           package + `ppa` command
├── runs/                    ledgers (one JSON file per run)
├── tests/                   test_core, test_truth, test_agent
└── src/ppa/
    ├── cli.py               ppa run | skills | truth | contract
    ├── config.py            reads agent.toml
    ├── core/                the instruments
    │   ├── projections.py   dr, mirror, wheel, gate_bits, joint_state (Layers 1, 2, 6)
    │   ├── sieve.py         segmented sieve
    │   ├── nulls.py         Cramer-wheel random-prime model
    │   ├── stats.py         transition tables, residual z, holdout score, Fano curve, constant B
    │   ├── factoring.py     trial, Fermat, Pollard rho, Pollard p-1, elliptic-curve method
    │   └── ledger.py        Finding + labels, writes runs/*.json
    ├── truth/               frozen tuples (read-only during a run)
    │   ├── truth_tuple.py       known values: pi(10^k), twins, structure, constants
    │   ├── randomness_tuple.py  the six-row signature of true randomness
    │   ├── nature_truth.py      known natural periods (sunspots, CO2, El Nino)
    │   ├── known_results.py     published results + retired ideas
    │   └── tool_tuple.py        the factoring tools, frozen, fingerprinted, calibrated
    ├── skills/              one detector per file, each names its null model
    │   ├── calibrate.py     sieve vs truth tuple (always first; stops the run on failure)
    │   ├── transitions.py   mod-M, lag-k transitions: stage A vs random, stage B vs gap rule
    │   ├── dr_gap.py        digital-root transitions vs the gap rule
    │   ├── fano.py          Fano factor vs window size
    │   ├── spectrum.py      FFT peaks vs wheel periods
    │   ├── gates.py         gate occupancy + 3-bit XOR rule
    │   ├── randomness.py    prime-derived sign sequences through the randomness tuple
    │   ├── nature.py        natural measurements: known periods + randomness tuple
    │   ├── curves.py        elliptic curves: |a_p| <= 2 sqrt(p) and Sato-Tate angles
    │   └── factoring.py     each factoring tool against numbers with different weaknesses
    └── agent/
        ├── planner.py       proposes hypotheses: seed, deepen, combine, refine
        └── loop.py          calibrate -> fixed skills -> explore -> ledger
```

## How the agent finds further concepts

The planner starts with every modulus in `start_moduli`, then learns from each result:

| Rule | Trigger | New hypothesis |
| --- | --- | --- |
| deepen | a pattern beats the random model | same modulus, one lag further |
| combine | two moduli both show patterns | their lcm (the Chinese Remainder / moire idea) |
| refine | a true candidate appears | 2M and 3M, ahead of everything else |

Transitions are tested in two stages. Stage A compares real primes with random wheel numbers.
Real primes usually win, and that is the known Lemke Oliver-Soundararajan bias. Stage B compares
them with the gap rule (next class = this class + gap, gaps taken from a coarser class). Only
stage B can produce a real **candidate**: structure that gap sizes do not explain.

## The tool tuple (check the tools, not just their results)

The factoring tools live in a frozen tuple. Before any of them runs, each one must reproduce a
known factorization and must fail on a prime, and the tuple's fingerprint (a hash of every tool's
source code) is printed. A stubbed tool that always claims success fails both checks and changes
the fingerprint, so it is caught three ways.

| Tool | Catches |
| --- | --- |
| trial | a small factor |
| fermat | two factors close together (N = a^2 - b^2) |
| rho | any factor up to about 1e12 (a walk mod p that loops) |
| p_minus_1 | p - 1 built from small primes |
| ecm | some elliptic curve with a smooth size p + 1 - a_p (inside the Hasse range) |

## Fast pre-check: casting out nines + mod 11

`truth_tuple.precheck(op, a, b, got)` checks an answer without redoing the calculation.
The digital root (mod 9) the answer *must* have comes from the inputs' digital roots;
if the answer's digital root differs, it is wrong. The digital root cannot see digit
order, so a second view, the alternating digit sum (mod 11), is checked too. Together
they work as one mod 99 check. `verify_calc` runs the pre-check first and only does the
exact calculation if it passes. `dr_pairs(6)` lists the 9 digit pairs that lead to 6.

Measured on 200,000 wrong multiplication answers:

| Mistake | Digital root | Mod 11 | Both |
| --- | --- | --- | --- |
| One wrong digit | 97.9% | 100% | 100% |
| Two digits swapped | 0% | 54% | 54% |
| Random wrong answer | 89.0% | 90.9% | 99.0% |

A failed pre-check proves the answer is wrong; a pass only means "probably right".
Going backwards does not work: 1,000,000 seven-digit numbers share each digital root.

### Blind spots (`ppa blindspots`, `truth/blind_spots.py`)

On the 9-position clock, 9 is a full lap, so 9 and 0 do nothing. That is why a 9 can be
skipped when taking a digital root, and also why the digital root cannot see a 9 or 0
added, dropped or traded, or any change in digit order. On the 11-position clock,
10 = -1, so digit places alternate + and -. That second view sees most of the gap:

| Mistake (all invisible to the digital root) | Mod 11 catches | Why |
| --- | --- | --- |
| 0 written as 9, or 9 as 0 | 100% | changes the value by 9 x 10^k = +-9 mod 11, never 0 |
| Neighbouring digits swapped | 100% | changes it by +-9(a - b), never 0 |
| A 9 or 0 dropped or added | ~92% | every digit to its left flips sign; misses ~1 in 11 |
| Any two digits swapped | ~57% | same-sign places (even gap) are invisible to mod 11 too |

## Example: RSA-PSS signatures (`examples/pss_demo.py`)

RSA-PSS built by hand from RFC 8017 (EMSA-PSS, MGF1 with SHA-256, 32-byte salt) and tested three ways:

1. **Checked by the real library:** every signature is verified by `cryptography` (OpenSSL). The same
   message gives a different signature each time, and a tampered message is rejected by both.
2. **Salts through the randomness tuple:** `os.urandom` passes; a reused salt fails (and makes PSS
   deterministic); Python's Mersenne Twister *passes* but is predictable from its seed. Passing the
   tuple is necessary, not sufficient.
3. **Keys through the tool tuple:** a badly generated 1024-bit key (primes close together) is factored
   by Fermat in one step, and a forged signature is accepted by OpenSSL. A properly generated key
   survives every tool.

```powershell
python -m pip install -e ".[crypto]"
python examples/pss_demo.py
```

## Labels

`established`, `conjecture`, `hypothesis`, `speculation`, `retired`, `rediscovery`,
`candidate`, `not-supported`, `check`. The agent can never assign anything higher than
`candidate`. Promotion needs a proof or a published result.

## Extend it

- **New skill:** copy any file in `skills/`, set `SKILL = Skill(name, null_model, description, run)`,
  add the module name to `MODULES` in `skills/__init__.py`.
- **New projection:** add a function to `core/projections.py`, use it inside a skill.
- **Your own measurements:** add a series in `load()` in `skills/nature.py`
  (for example spinning-fluid speeds or a tire runout trace).
- **New known result:** add a `Known(...)` row to `truth/known_results.py`, so the agent
  labels it as a rediscovery instead of new.

## First results (2026-11-21, primes to 10^9)

- Calibration: pi(10^9) = 50,847,534, matching the truth tuple.
- Transitions mod 3 and mod 5 beat random wheel numbers (z = 11 to 69). All of it is
  explained by the real gap distribution, so they are logged as rediscoveries.
- No candidates. Nothing beyond the gap rule survived at any modulus tested up to 72 and lag 3.
- Fano factor below 1 and falling with window size, as the Montgomery-Soundararajan conjecture predicts.
- The second-order constant B = 1 - Euler's constant - log(2 pi) = -1.4151, measured from prime
  counts in windows: -1.443 +/- 0.070 at 10^9, -1.415 +/- 0.017 at 10^10 (middle windows H = 10^3, 10^4).
- Natural measurements: the instrument found the 11-year sunspot cycle, yearly CO2 cycle and
  yearly El Nino cycle unaided.
