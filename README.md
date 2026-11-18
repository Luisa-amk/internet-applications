# CSU44000 Assignment 2: Distributed Machine Learning

Federated learning experiments built on Lab 10: FedAvg with the Flower framework and the Lab 10 CNN on CIFAR-10, run as a simulation on a single machine.

## Tasks

| Task | What | Notes |
|---|---|---|
| 1. Baseline | FedAvg, 10 clients, IID data | Tune rounds / local epochs / learning rate until test accuracy ≥ 50%. Record settings and plot accuracy over rounds. These settings are the baseline for everything else. |
| 2. Number of clients | 2, 5, 10 clients (IID) | Change only the client count. Compare accuracy over rounds and training time, and state the hardware used. |
| 3. Data distribution | IID vs non-IID, 10 clients | `DirichletPartitioner(partition_by="label", alpha=0.5)` in `task.py`. Also try α = 0.1 and 1.0. |
| 4. Extra tasks | 3 required for a team of 4 or 5 | ⌈(X+1)/2⌉ where X = team size. Each one must be justified in the report. |

Extra task ideas (pick 3):

From the brief:
- **Advanced FL algorithm** (e.g. FedProx / FedAdam) that beats FedAvg, using exactly the Task 1 settings
- **Results dashboard** (React/Vue) visualising Tasks 1–3 in the browser
- **More datasets**: MNIST, CIFAR-100
- **Weather prediction** across Irish cities (e.g. WeatherBench 2), building on Assignment 1
- **Decentralised FL** with no central server (e.g. blockchain)

Other ideas:
- **Client sampling**: only a fraction of clients train each round
- **Communication cost**: measure or compress model updates
- **Differential privacy**: privacy vs accuracy trade-off
- **Unreliable clients**: dropouts / stragglers

## Deadlines

| Date | Milestone |
|---|---|
| Fri 20 Nov, 23:59 | Email group registration ("CSU44000 A2 Group Registration") |
| ~Sat 21 Nov | Task 1 baseline locked; shared results format agreed |
| Thu 26 Nov | Draft A0 poster posted to the Discussion Board |
| Mon 30 Nov, 23:59 | Feedback given on the two neighbouring posters |
| **Tue 1 Dec, 23:59** | Final submission on Blackboard (declaration, report, poster) |

## Getting started

### Prerequisites

- **Python 3.11 or 3.12.** Flower 1.43+ needs 3.11+, and Ray (used for simulation) may not install on 3.13+. Check with `python3.12 --version`.
- Git, plus access to this repo on SCSS GitLab.
- No GPU needed. Everything runs on CPU.

### 1. Create a virtual environment

macOS / Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip flwr
pip install -r requirements-dev.txt

# Stop Flower reinstalling all packages (incl. PyTorch) before every run
echo 'export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1' >> .venv/bin/activate
deactivate && source .venv/bin/activate
```

Windows (PowerShell). Flower simulation on Windows is experimental, and WSL2 is recommended:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip flwr
pip install -r requirements-dev.txt
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = "1"   # set this in every new terminal
```

Conda works too (`conda create -n flower python=3.12`), as in the Lab 10 handout. `.venv/` is gitignored.

In VS Code, run **Python: Select Interpreter** and choose `.venv`.

### 2. Install the app

From the repo root (where `pyproject.toml` lives):

```sh
pip install -e .
```

### 3. Run an experiment

```sh
# Default settings (3 rounds, 2 clients)
flwr run . local --stream

# Custom rounds / learning rate / number of clients
flwr run . local --stream \
  --run-config "num-server-rounds=R learning-rate=0.001 fraction-evaluate=0.5" \
  --federation-config "num-supernodes=N"
```

`--run-config` overrides values under `[tool.flwr.app.config]` in `pyproject.toml` for that run only. The first run downloads CIFAR-10, so it takes longer.

### Lint and tests

```sh
ruff check .          # lint
ruff format .         # auto-format
pytest -q             # tests
```

The GitLab pipeline runs the same lint, format and syntax checks on every push, plus tests when a `tests/` folder exists. It never runs training.