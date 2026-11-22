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
ppa run                           # everything, using agent.toml
ppa run --intent explore          # only the self-directed exploration loop
ppa run --intent nature           # only natural measurements
ppa run --max 1e10 --steps 60     # bigger range, longer exploration
python -m pytest                  # 8 tests: core, truth, agent
```

If `ppa` is not found, use `python -m ppa run ...` instead.
Each run writes `runs/ledger-YYYYMMDD-HHMMSS.json`.

## Folder structure

```
prime-pattern-agent/
├── agent.toml               settings: range, thresholds, planner budget, null model
├── CONTRACT.md              the 9 rules every run obeys
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
    │   ├── stats.py         transition tables, residual z, holdout score, Fano curve
    │   └── ledger.py        Finding + labels, writes runs/*.json
    ├── truth/               frozen tuples (read-only during a run)
    │   ├── truth_tuple.py       known values: pi(10^k), twins, structure, constants
    │   ├── randomness_tuple.py  the six-row signature of true randomness
    │   ├── nature_truth.py      known natural periods (sunspots, CO2, El Nino)
    │   └── known_results.py     published results + retired ideas
    ├── skills/              one detector per file, each names its null model
    │   ├── calibrate.py     sieve vs truth tuple (always first; stops the run on failure)
    │   ├── transitions.py   mod-M, lag-k transitions: stage A vs random, stage B vs gap rule
    │   ├── dr_gap.py        digital-root transitions vs the gap rule
    │   ├── fano.py          Fano factor vs window size
    │   ├── spectrum.py      FFT peaks vs wheel periods
    │   ├── gates.py         gate occupancy + 3-bit XOR rule
    │   ├── randomness.py    prime-derived sign sequences through the randomness tuple
    │   └── nature.py        natural measurements: known periods + randomness tuple
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
