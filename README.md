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
| `server_app.py` | FedAvg strategy, evaluation on the full CIFAR-10 test set, writes the results file |
| `results.py` | Results JSON format, round timing, hardware info |
| `pyproject.toml` | Dependencies and default run settings (`[tool.flwr.app.config]`) |
| `scripts/run.py` | Runs one experiment and saves its results to `results/` |
| `scripts/plot_results.py` | Plots results files and prints a summary table |

### 3. Run an experiment

Use `scripts/run.py`. It wraps `flwr run`, saves results to `results/`, and checks setting names before a long run starts:

```sh
# 10 clients, IID, custom rounds / learning rate
python scripts/run.py -n 10 num-server-rounds=20 learning-rate=0.01

# Non-IID (Task 3), with a custom results file name
python scripts/run.py -n 10 --name task3-alpha0.5 partitioner=dirichlet dirichlet-alpha=0.5

# Show the flwr command without running it
python scripts/run.py --dry-run -n 5 seed=7
```

`-n` sets the number of clients (default 10). Any `key=value` overrides a setting under `[tool.flwr.app.config]` in `pyproject.toml` for that run only:

| Setting | Default | Notes |
|---|---|---|
| `num-server-rounds` | 3 | FL rounds |
| `local-epochs` | 1 | Epochs each client trains per round |
| `learning-rate` | 0.1 | SGD with momentum 0.9 |
| `batch-size` | 32 | |
| `fraction-evaluate` | 1.0 | Share of clients that run federated evaluation each round |
| `partitioner` | `iid` | `iid` or `dirichlet` |
| `dirichlet-alpha` | 0.5 | Only for `dirichlet`. Lower = more skewed clients |
| `seed` | 42 | Same seed + same settings = identical results |
| `save-model` | false | Saves the final model as a `.pt` next to the results file |

The first run downloads CIFAR-10, so it takes longer. The `HF_TOKEN` warning is harmless. Plain `flwr run . local --stream` still works, but its results go to `~/ia-a2-results/` instead, because Flower runs the server from wherever the local SuperLink was first started.

### 4. Results and plots

Each run writes one JSON file to `results/`, named after its settings (e.g. `iid-c10-r20-s42_20261123T114112.json`). The file is updated after every round, so a stopped run still keeps its progress. It contains:

- `config`: every setting used, `num_clients`, and `hardware` (CPU, cores, RAM, OS, library versions)
- `rounds`: one entry per round. Round 0 is the untrained model. Each entry has:
  - `accuracy` and `loss` on the full CIFAR-10 test set
  - `round_time_s`: client training plus federated evaluation, **excluding** the central test-set evaluation (`central_eval_time_s`)
  - `client_train` / `client_eval`: client metrics averaged over clients, weighted by example count, including `train_time_s`
- `total_time_s` and `finished`

Round 1 also includes one-off start-up costs (starting the client processes, loading partitions), so it's slower than later rounds. When comparing training time (Task 2), use the mean of rounds 2 onwards or the total, and say which.

```sh
python scripts/plot_results.py                                       # all of results/, to figures/accuracy.png
python scripts/plot_results.py results/iid-c*.json -o figures/task2.png --x time
python scripts/plot_results.py results/dirichlet* --metric loss
```

It also prints a table of final and best accuracy, mean seconds per round, and total time for each run. Commit finished results files so everyone plots from the same data.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `zsh: command not found: pip`, or `externally-managed-environment` | The venv isn't active or was moved. Recreate it (step 1) and use `python -m pip`. |
| `Failed to start local SuperLink` or `Internal server error` | Most likely the venv is on a path with spaces (see step 1). Stop the old SuperLink with `pkill -f "flower-superlink.*--port 39091"`, then check `~/.flwr/local-superlink/superlink.log`. |
| `Can't locate revision ...` when starting the SuperLink | `~/.flwr/local-superlink/state.db` was written by a different Flower version, or the spaces problem again. Move it aside (`mv state.db state.db.bak`) and rerun. |
| Run stuck at `starting` (`flwr ls`) and nothing happens | The background SuperLink has stalled, e.g. after the laptop slept. Restart it with `pkill -f "flower-superlink.*--port 39091"` and rerun. |
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