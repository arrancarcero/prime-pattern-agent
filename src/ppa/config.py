"""Load agent.toml. Uses tomllib (Python 3.11+) with a small fallback parser."""
from __future__ import annotations

import ast
from pathlib import Path

DEFAULTS = {
    "range": {"max_n": 1e8},
    "thresholds": {"z_survive": 4.0, "corr_survive": 0.5, "randomness_z": 4.0},
    "planner": {"max_steps": 24, "start_moduli": [3, 4, 5, 8, 9, 10, 12, 24],
                "start_lags": [1], "max_modulus": 72, "max_lag": 3},
    "null_model": {"wheel": 210, "seed": 20261121},
    "output": {"runs_dir": "runs"},
}


def _fallback_parse(text: str) -> dict:
    out: dict = {}
    section = out
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = out.setdefault(line[1:-1].strip(), {})
            continue
        key, val = (s.strip() for s in line.split("=", 1))
        section[key] = ast.literal_eval(val)
    return out


def load_config(path: str | Path = "agent.toml") -> dict:
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    p = Path(path)
    if p.exists():
        text = p.read_text(encoding="utf-8")
        try:
            import tomllib
            user = tomllib.loads(text)
        except ModuleNotFoundError:
            user = _fallback_parse(text)
        for sec, vals in user.items():
            cfg.setdefault(sec, {}).update(vals)
    cfg["range"]["max_n"] = int(float(cfg["range"]["max_n"]))
    return cfg
