"""pytorchexample: A Flower / PyTorch app."""

import time

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data, set_seed
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


def _partition_id(context: Context) -> int:
    return context.node_config["partition-id"]


def _load_client_data(context: Context):
    """This client's train/validation loaders, split the way the run config says."""
    return load_data(
        partition_id=_partition_id(context),
        num_partitions=context.node_config["num-partitions"],
        batch_size=context.run_config["batch-size"],
        partitioner=context.run_config["partitioner"],
        alpha=context.run_config["dirichlet-alpha"],
        seed=context.run_config["seed"],
    )


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    trainloader, _ = _load_client_data(context)

    # Seed per client and round: repeatable, but clients don't all shuffle identically
    seed = context.run_config["seed"]
    set_seed(seed + 1000 * msg.content["config"]["server-round"] + _partition_id(context))

    # Call the training function
    t_start = time.perf_counter()
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )
    train_time = time.perf_counter() - t_start

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "train_time_s": train_time,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    _, valloader = _load_client_data(context)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
