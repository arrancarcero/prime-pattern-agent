from ppa.agent.loop import run_agent
from ppa.config import load_config


def test_agent_runs_and_labels_everything(tmp_path):
    cfg = load_config("agent.toml")
    cfg["range"]["max_n"] = 10 ** 7
    cfg["planner"]["max_steps"] = 6
    cfg["output"]["runs_dir"] = str(tmp_path)
    ledger = run_agent(cfg, "explore")
    assert ledger.findings[0].skill == "calibrate" and ledger.findings[0].survived
    assert len(ledger.findings) >= 4
    assert list(tmp_path.glob("ledger-*.json"))
