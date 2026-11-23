"""Run one experiment and save its results JSON to ./results.

Wraps `flwr run` so that results always land in this repo's results/ folder, string
values don't need TOML quoting, and setting names are checked before a long run starts.

Examples:
    python scripts/run.py -n 10 num-server-rounds=20 learning-rate=0.01
    python scripts/run.py -n 10 --name task3-alpha0.5 partitioner=dirichlet dirichlet-alpha=0.5
    python scripts/run.py --dry-run -n 2      # print the flwr command without running it
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def app_config() -> dict:
    """The default run settings from pyproject.toml."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return pyproject["tool"]["flwr"]["app"]["config"]


def toml_value(raw: str) -> str:
    """Leave numbers/bools/quoted strings alone; quote anything else as a TOML string."""
    try:
        tomllib.loads(f"x = {raw}")
        return raw
    except tomllib.TOMLDecodeError:
        return json.dumps(raw)  # a JSON string is a valid TOML basic string


def parse_overrides(pairs: list[str], known: dict) -> dict[str, str]:
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {pair!r}")
        if key not in known:
            raise SystemExit(f"unknown setting {key!r}. Known settings: {', '.join(known)}")
        overrides[key] = toml_value(value)
    return overrides


def build_command(num_clients: int, overrides: dict[str, str], name: str | None, results_dir: Path):
    run_config = dict(overrides)
    run_config["results-dir"] = json.dumps(str(results_dir))
    if name:
        run_config["run-name"] = json.dumps(name)
    flwr = Path(sys.executable).with_name("flwr")
    return [
        str(flwr) if flwr.exists() else "flwr",
        "run",
        ".",
        "local",
        "--stream",
        "--run-config",
        " ".join(f"{k}={v}" for k, v in run_config.items()),
        "--federation-config",
        f"num-supernodes={num_clients}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-n", "--clients", type=int, default=10, help="number of clients")
    parser.add_argument("--name", help="results file name (default: generated from settings)")
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results", help="default: ./results"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the command only")
    parser.add_argument("settings", nargs="*", metavar="key=value", help="override pyproject")
    args = parser.parse_args(argv)

    if args.clients < 1:
        parser.error("--clients must be at least 1")

    overrides = parse_overrides(args.settings, app_config())
    cmd = build_command(args.clients, overrides, args.name, args.results_dir.resolve())
    print("$", shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION", "1")
    # flwr starts `flower-superlink` via PATH, so make this venv's bin/ visible even
    # when the venv isn't activated (e.g. `~/.venvs/ia-a2/bin/python scripts/run.py`)
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = os.pathsep.join([venv_bin, env.get("PATH", "")])
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
