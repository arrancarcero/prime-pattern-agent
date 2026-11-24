from ppa.agent.loop import run_agent
from ppa.config import load_config


def test_agent_runs_and_labels_everything(tmp_path):
    cfg = load_config("agent.toml")
    cfg["range"]["max_n"] = 10 ** 7
    cfg["planner"]["max_steps"] = 6
    cfg["output"]["runs_dir"] = str(tmp_path)
    cfg["graveyard"] = {"enforce": False}      # test the planner itself, not the graveyard
    ledger = run_agent(cfg, "explore")
    assert ledger.findings[0].skill == "calibrate" and ledger.findings[0].survived
    assert len(ledger.findings) >= 4
    assert list(tmp_path.glob("ledger-*.json"))


def test_graveyard_skips_buried_and_reopens(tmp_path):
    from ppa.truth.graveyard import Graveyard
    cfg = load_config("agent.toml")
    cfg["range"]["max_n"] = 10 ** 7
    cfg["planner"]["max_steps"] = 3
    cfg["output"]["runs_dir"] = str(tmp_path)
    ledger = run_agent(cfg, "explore")          # every seed lies under the 1e10 'beyond gaps' grave
    assert ledger.meta.get("graveyard_skipped")
    assert all(f.skill == "calibrate" for f in ledger.findings)
    g = Graveyard(tmp_path)
    assert g.check("transitions:M=9,lag=1", max_n=1e8).status == "buried"
    assert g.check("transitions:M=9,lag=1", max_n=1e11).status == "reopened"
    assert g.check("abundance:prime_Z", data="Gaia stellar survey").status == "reopened"
    assert g.check("quantum:mirrored_memory_vm", max_n=1e30).status == "buried"   # no reopen rule
    g.bury("transitions:M=97,lag=1", 1e7, "faded")
    assert Graveyard(tmp_path).check("transitions:M=97,lag=1", max_n=1e7).status == "buried"
    assert Graveyard(tmp_path).check("transitions:M=97,lag=1", max_n=1e9).status == "reopened"
