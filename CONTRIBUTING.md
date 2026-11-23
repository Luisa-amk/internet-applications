# Team reference: assumptions, fixed settings and conventions

Read this before starting a task. The README covers set-up and how to run things. This page covers what must stay fixed so all our results are comparable, and how we work in the repo.

## 1. Fixed: don't change these

Every experiment is compared against the Task 1 baseline, so these stay as they are on `main`. If your task needs something different, add a **new setting** (section 3) whose default keeps the baseline behaviour, and change it only in your own runs.

| What | Value | Where |
|---|---|---|
| Model | Lab 10 CNN (`Net`) | `fedlearn/task.py` |
| Dataset | CIFAR-10 (`uoft-cs/cifar10`) | `fedlearn/task.py` |
| Strategy | FedAvg, aggregating in a fixed order (`OrderedFedAvg`) | `fedlearn/strategy.py` |
| Rounds | 20 | `pyproject.toml` |
| Learning rate | 0.01 (SGD, momentum 0.9) | `pyproject.toml` |
| Local epochs | 1 | `pyproject.toml` |
| Batch size | 32 | `pyproject.toml` |
| Clients | 10 | `scripts/run.py -n` |
| Partitioner | IID | `pyproject.toml` |
| Seed | 42 | `pyproject.toml` |
| Library versions | flwr 1.44.0, flwr-datasets 0.6.1, torch 2.10.0, torchvision 0.25.0 | `pyproject.toml` |
| Test set | Full CIFAR-10 test set (10,000 images), evaluated on the server | `server_app.py` |
| Results format | JSON fields in README "Results and plots". Add new fields, never rename or remove existing ones (the dashboard reads them) | `fedlearn/results.py` |

**Baseline result:** 58.46% after 20 rounds, at `results/task1-baseline_*.json`. Compare against this file rather than re-running the baseline. If you do re-run it with the defaults, it should give exactly the same numbers. If it doesn't, something changed the baseline.

## 2. Assumptions behind the numbers

Things to know when you read or report results:

- **Runs are exactly repeatable.** The same seed and settings give identical accuracy, every round. This holds because:
  - the seed covers model initialisation, partitioning, each client's train/validation split and shuffling, and each client's training (seeded per client and per round);
  - the server sorts client updates by partition ID before averaging. Flower hands them over in arrival order, and floating-point sums change slightly with the order, which caused about 0.1% drift before this was fixed.
- **Each client splits its partition 80/20** into local train and validation sets. Client-side `eval_acc` uses that validation split. The headline `accuracy` is the server's evaluation on the full test set. Report `accuracy`.
- **Round 0** is the untrained model, at about 10% (random guessing).
- **`round_time_s` excludes the server's test-set evaluation**, which is logged separately as `central_eval_time_s`. Round 1 also includes one-off start-up costs. For timing comparisons, use the mean of rounds 2+ or `total_time_s`, and say which.
- **Timing comparisons must come from the same machine**, with nothing heavy running. Every file records its hardware.
- **Non-IID** means `DirichletPartitioner(partition_by="label", alpha=…)`, with a minimum of 10 examples per client. Lower alpha means more skewed clients.
- **58% is close to this model's limit.** It's a small, LeNet-style CNN with about 60k parameters. Don't expect 80%+ without changing the model, which would break comparability.

## 3. Conventions

### Branches, issues and MRs
- Each extra task has one **master issue**, assigned to its owner. Sub-issues are optional, for tracking your own work. Link them to the master.
- One branch per task: `feat/<task>`, e.g. `feat/fedprox`. Merge into `main` through an MR. `main` is protected and the pipeline must pass.
- Write `Closes #<issue>` in the MR description.

### Adding a setting
1. Add it under `[tool.flwr.app.config]` in `pyproject.toml`, with a comment. **Its default must reproduce the baseline**, e.g. `proximal-mu = 0.0`.
2. Read it with `context.run_config["your-setting"]`.
3. Add a type and range check to the list in `tests/test_config.py`. The existing tests already fail if a setting is read in code but missing from `pyproject.toml`, or defined there but never read.
4. Add a row to the README settings table.

### Adding a strategy (e.g. FedProx, FedAdam)
Wrap it so updates are averaged in a fixed order. Otherwise run-to-run noise gets mixed into your comparison with FedAvg:

```python
from flwr.serverapp.strategy import FedProx
from fedlearn.strategy import OrderedReplies


class OrderedFedProx(OrderedReplies, FedProx):
    """FedProx with a fixed aggregation order."""
```

Choose the strategy with a setting (e.g. `strategy = "fedavg"` by default) instead of replacing `OrderedFedAvg` in `server_app.py`.

### Adding a dataset
Keep CIFAR-10 as the default and switch with a setting, e.g. `dataset = "cifar10"`. A new dataset may need different image sizes or channel counts in `Net`, its own transforms, and its own server-side test set. Results on a different dataset aren't comparable to the baseline accuracy, so compare the *effects* (e.g. IID vs non-IID drop) instead.

### Running experiments
- Always use `scripts/run.py`, so results land in the repo with every setting recorded.
- Name each run after what it tests, and keep each task's results in its own folder:
  ```sh
  python scripts/run.py -n 10 --results-dir results/fedprox --name fedprox-mu0.1 strategy=fedprox proximal-mu=0.1
  ```
- Commit the results JSON for every run that appears in the report or poster. Delete broken or test runs instead of committing them.
- Plot with `scripts/plot_results.py`, saving to `figures/<task>-<what>.png`, and commit the figure too.
- **Can't run it yourself?** Rachel can run experiments. Post the exact `scripts/run.py` command on your issue.

### Code
- `ruff check .`, `ruff format .` and `pytest -q` must pass before you push. CI runs the same checks.
- Add tests for new logic. Tests must not download data or train for real: use small random tensors (see `tests/test_task.py`).
- Don't commit large files: model checkpoints (`results/*.pt` is gitignored), datasets or venvs.

## 4. Checklist for each extra task

Copy this into your master issue:

```markdown
- [ ] Short justification: why this task, what question it answers (for the report)
- [ ] Code on `feat/<task>`, behind a setting whose default keeps the baseline
- [ ] Tests for the new logic, `tests/test_config.py` updated
- [ ] Runs done with `scripts/run.py`; results JSON committed under `results/<task>/`
- [ ] Figure in `figures/`
- [ ] Comparison against `results/task1-baseline_*.json`
- [ ] 1 paragraph + figure for the report, and a poster panel
- [ ] MR merged, pipeline green
```

## 5. Status

| Item | Owner | State |
|---|---|---|
| Set-up: CI, results logging, seeds, pinned dependencies, partitioner setting | Rachel | Done |
| Task 1 baseline | Rachel | Done (58.46%) |
| Task 2: 2 / 5 / 10 clients | | Open |
| Task 3: IID vs non-IID (alpha 0.5, also 0.1 and 1.0) | | Open |
| Centralised reference run: same CNN trained on one machine, as a ceiling for the report | | Open, optional |
| Extra tasks | see issues | Open |

**After pulling the package rename** (from `pytorchexample` to `fedlearn`), reinstall once:
```sh
python -m pip uninstall -y quickstart-pytorch && python -m pip install -e .
```
