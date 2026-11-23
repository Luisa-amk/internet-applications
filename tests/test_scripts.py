"""The run/plot helper scripts. Nothing is actually trained."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load_script("run")
plot_results = load_script("plot_results")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("20", "20"), ("0.01", "0.01"), ("true", "true"), ("dirichlet", '"dirichlet"')],
)
def test_toml_value_quotes_only_strings(raw, expected):
    assert run.toml_value(raw) == expected


def test_parse_overrides_rejects_unknown_setting():
    with pytest.raises(SystemExit, match="unknown setting"):
        run.parse_overrides(["learning_rate=0.1"], run.app_config())


def test_build_command_passes_results_dir_with_spaces():
    overrides = run.parse_overrides(["partitioner=dirichlet"], run.app_config())
    cmd = run.build_command(5, overrides, "my run", Path("/a b/results"))

    assert cmd[1:5] == ["run", ".", "local", "--stream"]
    run_config = cmd[cmd.index("--run-config") + 1]
    assert 'partitioner="dirichlet"' in run_config
    assert 'results-dir="/a b/results"' in run_config
    assert 'run-name="my run"' in run_config
    assert cmd[-1] == "num-supernodes=5"


def test_run_config_string_is_parsed_correctly_by_flower():
    # Feed our string through Flower's own parser to be sure quoting survives
    from flwr.common.config import parse_config_args

    overrides = run.parse_overrides(
        ["partitioner=dirichlet", "dirichlet-alpha=0.1", "seed=7"], run.app_config()
    )
    cmd = run.build_command(10, overrides, None, Path("/x y/Trinity Documents/results"))
    parsed = parse_config_args([cmd[cmd.index("--run-config") + 1]])

    assert parsed == {
        "partitioner": "dirichlet",
        "dirichlet-alpha": 0.1,
        "seed": 7,
        "results-dir": "/x y/Trinity Documents/results",
    }


def test_plot_results_end_to_end(tmp_path, capsys):
    run_file = tmp_path / "r.json"
    run_file.write_text(
        json.dumps(
            {
                "name": "iid-c2-r2-s42",
                "num_clients": 2,
                "finished": True,
                "total_time_s": 20.0,
                "rounds": [
                    {"round": 0, "accuracy": 0.1, "loss": 2.3, "round_time_s": None},
                    {"round": 1, "accuracy": 0.3, "loss": 1.9, "round_time_s": 9.0},
                    {"round": 2, "accuracy": 0.4, "loss": 1.7, "round_time_s": 11.0},
                ],
            }
        )
    )
    out = tmp_path / "fig.png"
    plot_results.main([str(run_file), "-o", str(out), "--x", "time"])

    assert out.stat().st_size > 0
    printed = capsys.readouterr().out
    assert "iid-c2-r2-s42" in printed and "40.00%" in printed and "10.0" in printed
