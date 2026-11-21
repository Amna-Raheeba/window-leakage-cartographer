import json
import os

from window_leakage.cli import main


def test_cli_no_command_returns_2(capsys):
    assert main([]) == 2


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Leaked test windows" in out
    assert "seed 7" in out


def test_cli_evaluate_writes_files(tmp_path):
    out_dir = str(tmp_path / "results")
    assert main(["evaluate", "--out", out_dir]) == 0
    for name in (
        "evaluation.json",
        "evaluation.csv",
        "evaluation_summary.md",
        "HEADLINE.md",
    ):
        assert os.path.isfile(os.path.join(out_dir, name)), name
    with open(os.path.join(out_dir, "evaluation.json")) as fh:
        payload = json.load(fh)
    assert "headline" in payload and "per_config" in payload
    assert payload["n_runs"] == 20
