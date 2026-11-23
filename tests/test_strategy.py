"""Replies are aggregated in a fixed order, whatever order they arrive in."""

import random
from types import SimpleNamespace

import numpy as np
import torch
from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from fedlearn.strategy import META_KEY, OrderedFedAvg, client_meta, sort_replies


class FakeReply:
    """Stands in for flwr's Message, which can only be created inside a running app."""

    def __init__(self, content: RecordDict, partition_id: int):
        self.content = content
        self.metadata = SimpleNamespace(src_node_id=1000 + partition_id)

    def has_error(self) -> bool:
        return False

    def has_content(self) -> bool:
        return True


def reply(partition_id: int, value: float, n: int) -> FakeReply:
    content = RecordDict(
        {
            "arrays": ArrayRecord({"w": torch.full((1000,), value)}),
            "metrics": MetricRecord({"num-examples": n}),
            META_KEY: client_meta(partition_id),
        }
    )
    return FakeReply(content, partition_id)


def make_replies():
    rng = np.random.default_rng(0)
    return [reply(i, float(rng.random()), int(rng.integers(100, 5000))) for i in range(10)]


def partition_ids(replies):
    return [m.content.config_records[META_KEY]["partition-id"] for m in replies]


def test_sort_replies_orders_by_partition_id():
    replies = make_replies()
    random.Random(1).shuffle(replies)
    assert partition_ids(sort_replies(replies)) == list(range(10))


def test_aggregation_is_identical_for_any_arrival_order():
    strategy = OrderedFedAvg()
    results = set()
    for shuffle_seed in range(5):
        replies = make_replies()
        random.Random(shuffle_seed).shuffle(replies)
        arrays, _ = strategy.aggregate_train(1, replies)
        results.add(arrays["w"].numpy().tobytes())
    assert len(results) == 1


def test_ordered_fedavg_is_still_fedavg():
    replies = make_replies()
    ordered, _ = OrderedFedAvg().aggregate_train(1, replies)
    plain, _ = FedAvg().aggregate_train(1, replies)
    np.testing.assert_array_equal(ordered["w"].numpy(), plain["w"].numpy())
