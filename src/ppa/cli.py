"""Command line:  ppa run [--intent ...] [--max N]   |  ppa skills  |  ppa truth  |  ppa contract
Without installing:  python -m ppa run --intent explore"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ppa", description="Prime-Pattern Agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the agent")
    r.add_argument("--intent", default="all",
                   help="all | explore | dr_gap | fano | spectrum | gates | randomness | nature | curves | factoring")
    r.add_argument("--max", type=float, help="override range.max_n")
    r.add_argument("--config", default="agent.toml")
    r.add_argument("--steps", type=int, help="override planner.max_steps")
    sub.add_parser("skills", help="list skills and their null models")
    t = sub.add_parser("truth", help="self-test the truth tuple")
    t.add_argument("--exp", type=int, default=8)
    sub.add_parser("contract", help="print the agent contract")
    sub.add_parser("tools", help="fingerprint and calibrate the factoring tool tuple")
    a = ap.parse_args(argv)

    if a.cmd == "skills":
        from ppa.skills import REGISTRY
        for s in REGISTRY.values():
            print(f"{s.name:12} null: {s.null_model}\n{'':12} {s.description}")
    elif a.cmd == "truth":
        from ppa.truth import truth_tuple
        print("fingerprint", truth_tuple.FINGERPRINT)
        bad = 0
        for v in truth_tuple.self_test(a.exp):
            bad += v.ok is False
            print("PASS" if v.ok else "FAIL", v.key, v.args, v.got, v.note)
        print("ALL PASS" if not bad else f"{bad} FAILED")
    elif a.cmd == "tools":
        from ppa.truth.tool_tuple import TOOL_TUPLE, fingerprint, verify_tools
        print("tool tuple fingerprint", fingerprint())
        for (name, ok, null_ok), t in zip(verify_tools(), TOOL_TUPLE):
            print(f"{name:10} known answer {'PASS' if ok else 'FAIL'}  null prime {'PASS' if null_ok else 'FAIL'}  catches: {t.catches}")
    elif a.cmd == "contract":
        p = Path(__file__).resolve().parents[2] / "CONTRACT.md"
        print(p.read_text(encoding="utf-8") if p.exists() else "CONTRACT.md not found")
    else:
        from ppa.agent.loop import run_agent
        from ppa.config import load_config
        cfg = load_config(a.config)
        if a.max:
            cfg["range"]["max_n"] = int(a.max)
        if a.steps:
            cfg["planner"]["max_steps"] = a.steps
        run_agent(cfg, a.intent)


if __name__ == "__main__":
    main()
