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

> [!WARNING]
> **Keep the virtual environment on a path with no spaces.** Flower's local SuperLink breaks if its packages are installed under a path containing spaces (e.g. `.../Trinity Documents/Year 4/...`). It fails with `Failed to start local SuperLink` or `Internal server error`. To avoid this, put the venv in your home folder, not inside the repo. A venv also can't be moved after it's created, so if you move it, delete it and create it again.

macOS / Linux:

```sh
python3.12 -m venv ~/.venvs/ia-a2

# Stop Flower reinstalling all packages (incl. PyTorch) before every run
echo 'export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1' >> ~/.venvs/ia-a2/bin/activate

source ~/.venvs/ia-a2/bin/activate
which python                     # should print ~/.venvs/ia-a2/bin/python
python -m pip install -U pip
```

Windows (PowerShell). Flower simulation on Windows is experimental, and WSL2 is recommended:

```powershell
py -3.12 -m venv $HOME\.venvs\ia-a2
& $HOME\.venvs\ia-a2\Scripts\Activate.ps1
python -m pip install -U pip
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = "1"   # set this in every new terminal
```

Conda works too (`conda create -n flower python=3.12`), as in the Lab 10 handout. Conda envs live in your home folder, so the spaces problem doesn't apply.

In VS Code, run **Python: Select Interpreter**, choose **Enter interpreter path...**, and enter `~/.venvs/ia-a2/bin/python` (Windows: `%USERPROFILE%\.venvs\ia-a2\Scripts\python.exe`).

### 2. Install the app and dev tools

From the repo root (where `pyproject.toml` lives), with the venv active:

```sh
python -m pip install -e . -r requirements-dev.txt
```

This installs Flower, PyTorch, torchvision and Flower Datasets from `pyproject.toml`, plus ruff and pytest. PyTorch is a large download.

The app code is in `pytorchexample/`, copied from the Lab 10 quickstart:

| File | What it does |
|---|---|
| `task.py` | CNN model, CIFAR-10 loading and partitioning, train/test loops |
| `client_app.py` | What each client does: local training and evaluation |
| `server_app.py` | FedAvg strategy and evaluation on the full CIFAR-10 test set |
| `pyproject.toml` | Dependencies and default run settings (`[tool.flwr.app.config]`) |

### 3. Run an experiment

```sh
# Default settings (3 rounds, 2 clients)
flwr run . local --stream

# Custom rounds / learning rate / number of clients
flwr run . local --stream \
  --run-config "num-server-rounds=R learning-rate=0.001 fraction-evaluate=0.5" \
  --federation-config "num-supernodes=N"
```

`--run-config` overrides values under `[tool.flwr.app.config]` in `pyproject.toml` for that run only. The first run downloads CIFAR-10, so it takes longer. The `HF_TOKEN` warning is harmless.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `zsh: command not found: pip`, or `externally-managed-environment` | The venv isn't active or was moved. Recreate it (step 1) and use `python -m pip`. |
| `Failed to start local SuperLink` or `Internal server error` | Most likely the venv is on a path with spaces (see step 1). Stop the old SuperLink with `pkill -f "flower-superlink.*--port 39091"`, then check `~/.flwr/local-superlink/superlink.log`. |
| `Can't locate revision ...` when starting the SuperLink | `~/.flwr/local-superlink/state.db` was written by a different Flower version, or the spaces problem again. Move it aside (`mv state.db state.db.bak`) and rerun. |
| Stuck on `Installing application dependencies...` | `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` isn't set. Re-activate the venv and restart the SuperLink. |

### Lint and tests

```sh
ruff check .          # lint
ruff format .         # auto-format
pytest -q             # tests
```

Tests use small random data, so they need no downloads and finish in a few seconds. `tests/test_config.py` checks that every `run_config[...]` key the code reads is defined in `pyproject.toml` with a sensible value. Run it after adding a new setting.

### CI pipeline

Every push runs `lint → build → test` on GitLab. It **never runs training**.

| Stage | Job | What it checks |
|---|---|---|
| lint | `lint:ruff` | Lint errors, also shown in the MR's Code Quality widget |
| lint | `lint:format` | Formatting (warning only, so it doesn't block). Fix with `ruff format .` |
| build | `build:compile` | Every file compiles |
| build | `build:flower-app` | `flwr build` packages the app like `flwr run` does. The `.fab` is kept for a week |
| test | `test:config` | `pyproject.toml` settings match what the code reads (standard library only, seconds) |
| test | `test:unit` | Model shape, train/test loops, app wiring (installs CPU PyTorch, ~1–2 min uncached) |

Test results appear in the pipeline's **Tests** tab and on merge requests. When an MR is open, only the MR pipeline runs, not a duplicate branch pipeline.