"""dm-bench CLI: build -> verify -> solve -> eval round trip (fast methods)."""
import json

from src.bench.__main__ import main


def test_cli_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DM_BENCH_TEST_SECRET", raising=False)
    rel, sols = str(tmp_path / "rel"), str(tmp_path / "sol")
    assert main(["build", "--out", rel, "--tiers", "easy", "--splits", "val", "test",
                 "--limit-docs", "1"]) == 0
    assert "skipping split 'test'" in capsys.readouterr().err
    assert main(["verify", rel]) == 0

    assert main(["solve", "--release", rel, "--out", sols + "/oracle", "--method", "oracle",
                 "--tiers", "easy", "--splits", "val"]) == 0
    assert main(["solve", "--release", rel, "--out", sols + "/random", "--method", "random",
                 "--tiers", "easy", "--splits", "val"]) == 0
    assert main(["solve", "--release", rel, "--out", sols + "/x", "--method", "oracle",
                 "--tiers", "easy", "--splits", "test"]) == 2          # oracle refused on test

    for method, expected in (("oracle", 1.0), ("random", 0.0)):
        out = str(tmp_path / f"{method}.json")
        assert main(["eval", "--release", rel, "--solutions", f"{sols}/{method}",
                     "--tiers", "easy", "--splits", "val", "--out", out]) == 0
        with open(out) as fh:
            res = json.load(fh)
        assert res["eval_version"] == "1.0"
        assert res["benchmark"] == "dm-bench@0.1.0"
        assert res["solvers"] == [f"{method}@0.1"]
        assert len(res["release_sha256"]) == 64
        assert res["scores"]["easy"]["val"]["direct_acc"]["mean"] == expected


def test_solve_fails_on_missing_release(tmp_path):
    assert main(["solve", "--release", str(tmp_path / "nope"), "--out", str(tmp_path / "s")]) == 2
