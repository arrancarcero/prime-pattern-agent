"""Skills registry. Each skill is one detector with a named null model.
Add a skill: write a module with a `SKILL = Skill(...)`, then list it in MODULES."""
from importlib import import_module

MODULES = ("calibrate", "transitions", "dr_gap", "fano", "spectrum",
           "gates", "randomness", "nature", "curves", "factoring", "candles", "spacing")

REGISTRY = {}
for _m in MODULES:
    _s = import_module(f"ppa.skills.{_m}").SKILL
    REGISTRY[_s.name] = _s
