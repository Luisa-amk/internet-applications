# CSU44000 Assignment 2: Distributed Machine Learning

Federated learning experiments built on Lab 10: FedAvg with the Flower framework and the Lab 10 CNN on CIFAR-10, run as a simulation on a single machine.

## Tasks

| Task | What | Notes |
|---|---|---|
| 1. Baseline | FedAvg, 10 clients, IID data | Tune rounds / local epochs / learning rate until test accuracy ≥ 50%. Record settings and plot accuracy over rounds. These settings are the baseline for everything else. |
| 2. Number of clients | 2, 5, 10 clients (IID) | Change only the client count. Compare accuracy over rounds and training time, and state the hardware used. |
| 3. Data distribution | IID vs non-IID, 10 clients | `DirichletPartitioner(partition_by="label", alpha=0.5)` in `task.py`. Also try α = 0.1 and 1.0. |
| 4. Extra tasks | 3 required for a team of 4 or 5 | ⌈(X+1)/2⌉ where X = team size. Each one must be justified in the report. |

Chosen extra tasks:
1. **Advanced FL algorithm** (e.g. FedProx / FedAdam) that beats FedAvg, using exactly the Task 1 settings.
2. **Results dashboard** (React/Vue) that visualises Tasks 1–3 in the browser.
3. **More datasets**: MNIST and CIFAR-100.

## Deadlines

| Date | Milestone |
|---|---|
| Fri 20 Nov, 23:59 | Email group registration ("CSU44000 A2 Group Registration") |
| ~Sat 21 Nov | Task 1 baseline locked; shared results format agreed |
| Thu 26 Nov | Draft A0 poster posted to the Discussion Board |
| Mon 30 Nov, 23:59 | Feedback given on the two neighbouring posters |
| **Tue 1 Dec, 23:59** | Final submission on Blackboard (declaration, report, poster) |

## Work split

| Role | Code / experiments | Writing | Owner |
|---|---|---|---|
| A | Task 1 baseline, shared results logging (CSV/JSON), Task 2 client sweep | Report: setup and baseline | _TBD_ |
| B | Task 3: IID vs non-IID (α = 0.1, 0.5, 1.0) | Report: data distribution analysis | _TBD_ |
| C | Extra 1: advanced FL algorithm | Report: algorithm comparison | _TBD_ |
| D | Extra 2: results dashboard | Poster layout owner | _TBD_ |
| E | Extra 3: MNIST / CIFAR-100 | Report editor, declaration, references | _TBD_ |

With 4 people, A also takes Task 3, and B/C/D take one extra task each.

## Ground rules

- **Task 1 blocks everything.** Nobody changes baseline settings after they're locked.
- Run all Task 2 timing experiments on **one machine** so the times are comparable, and record its specs.
- Every experiment writes results in the shared format, so the plots, dashboard and poster all use the same data.
- Commit your own work regularly. The repo history is expected to show steady progress over the two weeks.
- **No GenAI-generated text or images in the report or poster** (module rule). Write your own sections in your own words.
- The report uses the IEEE conference template, max 2 pages including references.
