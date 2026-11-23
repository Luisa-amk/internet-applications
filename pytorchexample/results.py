"""Save each run's settings, per-round metrics, timing and hardware to a JSON file.

One file per run, written after every round so a crashed or stopped run still leaves
its partial results. `scripts/plot_results.py` reads these files.
"""

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Used when no results-dir is given (e.g. plain `flwr run`). The ServerApp's working
# directory is wherever the local SuperLink was first started, so a relative path
# would be unpredictable.
DEFAULT_RESULTS_DIR = Path.home() / "ia-a2-results"


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def hardware_info() -> dict:
    """What the run was executed on. Task 2 asks for this alongside training time."""
    import torch  # local import: keeps this module importable without torch for tests

    return {
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "flwr": _pkg_version("flwr"),
        "flwr_datasets": _pkg_version("flwr-datasets"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def default_run_name(config: dict, num_clients: int) -> str:
    """e.g. 'iid-c10-r20-s42' or 'dirichlet0.5-c10-r20-s42'."""
    data = config["partitioner"]
    if data == "dirichlet":
        data += f"{float(config['dirichlet-alpha']):g}"
    return f"{data}-c{num_clients}-r{config['num-server-rounds']}-s{config['seed']}"


@dataclass
class RunRecord:
    """Everything about one run. Serialised as-is to JSON."""

    name: str
    config: dict
    num_clients: int
    hardware: dict
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished: bool = False
    total_time_s: float | None = None
    # One entry per round (round 0 = the untrained initial model)
    rounds: list[dict] = field(default_factory=list)

    def path(self, results_dir: Path) -> Path:
        stamp = self.started_at.replace(":", "").replace("-", "")
        return results_dir / f"{self.name}_{stamp}.json"

    def save(self, results_dir: Path) -> Path:
        results_dir.mkdir(parents=True, exist_ok=True)
        path = self.path(results_dir)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)  # atomic, so a half-written file is never left behind
        return path


class RoundTimer:
    """Splits wall-clock time into training rounds and central evaluation.

    Flower calls the central evaluate function once before training (round 0) and
    then once after each round. The time between the end of one evaluation and the
    start of the next is the round itself: client training plus federated
    evaluation, but not the central evaluation on the full test set.
    """

    def __init__(self) -> None:
        self._last_eval_end: float | None = None
        self._eval_start: float | None = None

    def eval_started(self) -> float | None:
        """Call at the start of central evaluation. Returns the round's duration."""
        self._eval_start = time.perf_counter()
        if self._last_eval_end is None:
            return None
        return self._eval_start - self._last_eval_end

    def eval_finished(self) -> float:
        """Call at the end of central evaluation. Returns how long evaluation took."""
        self._last_eval_end = time.perf_counter()
        return self._last_eval_end - self._eval_start
