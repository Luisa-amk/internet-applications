"""fedlearn: federated learning experiments with Flower and PyTorch."""

import time
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from fedlearn.results import (
    DEFAULT_RESULTS_DIR,
    RoundTimer,
    RunRecord,
    default_run_name,
    hardware_info,
)
from fedlearn.strategy import OrderedFedAvg
from fedlearn.task import Net, load_centralized_dataset, set_seed, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    seed: int = context.run_config["seed"]
    results_dir = Path(context.run_config["results-dir"] or DEFAULT_RESULTS_DIR).expanduser()

    # Same seed -> same initial global model, so runs differ only in what we change
    set_seed(seed)
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    num_clients = len(list(grid.get_node_ids()))
    config = dict(context.run_config)
    record = RunRecord(
        name=context.run_config["run-name"] or default_run_name(config, num_clients),
        config=config,
        num_clients=num_clients,
        hardware=hardware_info(),
    )
    print(f"\nSaving results to {record.path(results_dir)}\n")

    timer = RoundTimer()

    def evaluate_and_record(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        round_time = timer.eval_started()
        metrics = global_evaluate(server_round, arrays)
        eval_time = timer.eval_finished()

        record.rounds.append(
            {
                "round": server_round,
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "round_time_s": round_time,
                "central_eval_time_s": eval_time,
            }
        )
        record.save(results_dir)  # save every round so partial runs aren't lost
        return metrics

    # FedAvg, aggregating client updates in a fixed order (see strategy.py)
    strategy = OrderedFedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    t_start = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_and_record,
    )
    record.total_time_s = time.perf_counter() - t_start

    # Add the aggregated client-side metrics to each round
    for entry in record.rounds:
        rnd = entry["round"]
        if rnd in result.train_metrics_clientapp:
            entry["client_train"] = dict(result.train_metrics_clientapp[rnd])
        if rnd in result.evaluate_metrics_clientapp:
            entry["client_eval"] = dict(result.evaluate_metrics_clientapp[rnd])
    record.finished = True
    path = record.save(results_dir)
    print(f"\nResults saved to {path}")

    if context.run_config["save-model"]:
        # Save final model next to the results file
        model_path = path.with_suffix(".pt")
        print(f"\nSaving final model to {model_path}...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, model_path)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Entire CIFAR-10 test set (loaded once, then cached)
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
