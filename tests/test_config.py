"""Check pyproject.toml against the code that reads it.

These tests need only the standard library (no torch / flwr), so CI can run them
in a few seconds before the heavier unit tests.
"""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "pytorchexample"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
FLWR = PYPROJECT["tool"]["flwr"]["app"]
RUN_CONFIG = FLWR["config"]

# Matches context.run_config["some-key"] / ['some-key']
RUN_CONFIG_KEY = re.compile(r"""run_config\[\s*["']([\w-]+)["']\s*\]""")


def keys_read_by_app() -> set[str]:
    keys = set()
    for path in APP_DIR.glob("*.py"):
        keys |= set(RUN_CONFIG_KEY.findall(path.read_text()))
    return keys


def test_app_reads_some_config():
    # Guards against the regex silently matching nothing after a refactor
    assert keys_read_by_app(), "found no run_config[...] reads; is the regex out of date?"


def test_every_key_the_app_reads_is_defined():
    missing = keys_read_by_app() - RUN_CONFIG.keys()
    assert not missing, f"read in code but missing from [tool.flwr.app.config]: {sorted(missing)}"


def test_no_unused_config_keys():
    unused = RUN_CONFIG.keys() - keys_read_by_app()
    assert not unused, f"defined in [tool.flwr.app.config] but never read: {sorted(unused)}"


@pytest.mark.parametrize(
    ("key", "kind", "check"),
    [
        ("num-server-rounds", int, lambda v: v >= 1),
        ("local-epochs", int, lambda v: v >= 1),
        ("batch-size", int, lambda v: v >= 1),
        ("learning-rate", float, lambda v: 0 < v < 1),
        ("fraction-evaluate", float, lambda v: 0 <= v <= 1),
        ("save-model", bool, lambda v: True),
    ],
)
def test_config_value_types_and_ranges(key, kind, check):
    value = RUN_CONFIG[key]
    # bool is a subclass of int, so check it explicitly for the int keys
    assert type(value) is kind, f"{key} should be {kind.__name__}, got {type(value).__name__}"
    assert check(value), f"{key}={value!r} is out of range"


@pytest.mark.parametrize("component", ["serverapp", "clientapp"])
def test_components_point_at_existing_files(component):
    module, _, attr = FLWR["components"][component].partition(":")
    path = ROOT / Path(*module.split(".")).with_suffix(".py")
    assert path.is_file(), f"{component} module {module} not found at {path}"
    assert re.search(rf"^{attr}\s*=", path.read_text(), re.MULTILINE), (
        f"{attr} not defined in {path}"
    )
