"""Partitioner choice and seeding. Uses a small synthetic dataset, so nothing is downloaded."""

import numpy as np
import pytest
import torch
from datasets import Dataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

from pytorchexample.task import Net, make_partitioner, set_seed


def fake_labels(n=2000, classes=10):
    return Dataset.from_dict({"label": [i % classes for i in range(n)]})


def class_counts(partitioner, num_partitions, classes=10):
    return np.array(
        [
            np.bincount(partitioner.load_partition(i)["label"], minlength=classes)
            for i in range(num_partitions)
        ]
    )


def test_make_partitioner_iid():
    p = make_partitioner("iid", num_partitions=10, alpha=0.5, seed=42)
    assert isinstance(p, IidPartitioner)


def test_make_partitioner_dirichlet_uses_label_and_alpha():
    p = make_partitioner("dirichlet", num_partitions=10, alpha=0.5, seed=42)
    assert isinstance(p, DirichletPartitioner)
    assert p._partition_by == "label"
    assert np.allclose(p._alpha, 0.5)


def test_make_partitioner_accepts_int_alpha():
    # `dirichlet-alpha=1` on the command line arrives as an int
    p = make_partitioner("dirichlet", num_partitions=5, alpha=1, seed=42)
    assert np.allclose(p._alpha, 1.0)


def test_make_partitioner_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown partitioner"):
        make_partitioner("random", num_partitions=10, alpha=0.5, seed=42)


def test_dirichlet_is_skewed_and_iid_is_not():
    # The whole point of Task 3: Dirichlet clients have lopsided class mixes, IID don't
    iid = make_partitioner("iid", num_partitions=5, alpha=0.5, seed=42)
    iid.dataset = fake_labels().shuffle(seed=0)
    dirichlet = make_partitioner("dirichlet", num_partitions=5, alpha=0.5, seed=42)
    dirichlet.dataset = fake_labels()

    def mean_largest_class_share(counts):
        return (counts.max(axis=1) / counts.sum(axis=1)).mean()

    assert mean_largest_class_share(class_counts(iid, 5)) < 0.2  # ~0.1 for a fair mix
    assert mean_largest_class_share(class_counts(dirichlet, 5)) > 0.3


def test_dirichlet_split_is_repeatable_with_same_seed():
    def split(seed):
        p = make_partitioner("dirichlet", num_partitions=5, alpha=0.5, seed=seed)
        p.dataset = fake_labels()
        return class_counts(p, 5)

    assert np.array_equal(split(42), split(42))
    assert not np.array_equal(split(42), split(7))


def test_set_seed_gives_identical_initial_model():
    set_seed(42)
    a = Net().state_dict()
    set_seed(42)
    b = Net().state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
