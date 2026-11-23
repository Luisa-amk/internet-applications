"""Strategies that aggregate client replies in a fixed order, so runs repeat exactly.

Flower hands replies to the strategy in the order they arrive, which changes from run
to run. Floating-point addition isn't associative, so averaging the same updates in a
different order gives slightly different weights (~0.1% accuracy drift by round 20).
Sorting replies by partition ID first makes the same seed give identical results.

For another strategy (e.g. FedProx), combine the mixin the same way:
    class OrderedFedProx(OrderedReplies, FedProx): ...
"""

from collections.abc import Iterable

from flwr.app import ConfigRecord, Message
from flwr.serverapp.strategy import FedAvg

META_KEY = "meta"  # ConfigRecord each client adds to its reply: {"partition-id": int}


def client_meta(partition_id: int) -> ConfigRecord:
    return ConfigRecord({"partition-id": partition_id})


def sort_replies(replies: Iterable[Message]) -> list[Message]:
    """Replies ordered by the sender's partition ID; error replies go last."""

    def key(msg: Message) -> tuple[int, int]:
        if msg.has_error():
            return (1, 0)
        meta = msg.content.config_records.get(META_KEY)
        if meta is None:  # client didn't tag its reply: fall back to node ID
            return (0, msg.metadata.src_node_id)
        return (0, int(meta["partition-id"]))

    return sorted(replies, key=key)


class OrderedReplies:
    """Mixin: sort replies before the parent strategy aggregates them."""

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        return super().aggregate_train(server_round, sort_replies(replies))

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        return super().aggregate_evaluate(server_round, sort_replies(replies))


class OrderedFedAvg(OrderedReplies, FedAvg):
    """FedAvg with a fixed aggregation order."""
