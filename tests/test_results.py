"""Results JSON format, run naming and the round timer."""

import json

import pytest

from pytorchexample.results import RoundTimer, RunRecord, default_run_name, hardware_info

BASE = {"partitioner": "iid", "dirichlet-alpha": 0.5, "num-server-rounds": 20, "seed": 42}


def test_default_run_name_iid():
    assert default_run_name(BASE, num_clients=10) == "iid-c10-r20-s42"


def test_default_run_name_dirichlet_includes_alpha():
    cfg = BASE | {"partitioner": "dirichlet", "dirichlet-alpha": 0.1}
    assert default_run_name(cfg, num_clients=5) == "dirichlet0.1-c5-r20-s42"


def test_hardware_info_has_what_task2_needs():
    info = hardware_info()
    for key in ("cpu", "cpu_count", "ram_gb", "os", "torch", "flwr"):
        assert key in info
    assert info["cpu_count"] >= 1


def test_run_record_round_trips_through_json(tmp_path):
    record = RunRecord(name="iid-c2-r1-s42", config=BASE, num_clients=2, hardware={"cpu": "x"})
    record.rounds.append({"round": 0, "accuracy": 0.1, "loss": 2.3, "round_time_s": None})
    path = record.save(tmp_path)

    data = json.loads(path.read_text())
    assert path.name.startswith("iid-c2-r1-s42_")
    assert data["config"] == BASE
    assert data["rounds"][0]["accuracy"] == 0.1
    assert data["finished"] is False
    assert not list(tmp_path.glob("*.tmp"))  # temp file is renamed away


def test_run_record_save_overwrites_same_file(tmp_path):
    record = RunRecord(name="r", config=BASE, num_clients=2, hardware={})
    first = record.save(tmp_path)
    record.finished = True
    second = record.save(tmp_path)
    assert first == second
    assert json.loads(second.read_text())["finished"] is True
    assert len(list(tmp_path.iterdir())) == 1


def test_round_timer_excludes_central_eval(monkeypatch):
    clock = iter([0.0, 5.0, 15.0, 18.0])  # eval0 start/end, eval1 start/end
    monkeypatch.setattr("pytorchexample.results.time.perf_counter", lambda: next(clock))
    timer = RoundTimer()

    assert timer.eval_started() is None  # round 0: no training happened yet
    assert timer.eval_finished() == pytest.approx(5.0)
    assert timer.eval_started() == pytest.approx(10.0)  # 15 - 5: training only
    assert timer.eval_finished() == pytest.approx(3.0)
