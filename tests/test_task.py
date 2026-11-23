"""Smoke tests for the model and the train/test loops in task.py.

Uses small random data, so nothing is downloaded and the tests run on CPU in seconds.
"""

import math

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from pytorchexample.task import Net, apply_transforms, train
from pytorchexample.task import test as evaluate

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def fake_loader(n=16, batch_size=8):
    """A tiny CIFAR-10-shaped dataset in the same {"img", "label"} format as Flower Datasets."""
    data = [{"img": torch.randn(3, 32, 32), "label": torch.tensor(i % 10)} for i in range(n)]
    return DataLoader(data, batch_size=batch_size)


def test_net_output_shape():
    out = Net()(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)


def test_net_is_the_lab10_cnn():
    # Tasks 2-4 must use the same model as the Task 1 baseline.
    # If you change the CNN on purpose, update this number and say so in the report.
    n_params = sum(p.numel() for p in Net().parameters())
    assert n_params == 62_006


def test_apply_transforms_normalises_images():
    img = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
    out = apply_transforms({"img": [img]})["img"][0]
    assert out.shape == (3, 32, 32)
    # Normalize((0.5,)*3, (0.5,)*3) maps pixel 255 -> 1.0
    assert torch.allclose(out, torch.ones_like(out))


def test_train_updates_weights_and_returns_finite_loss():
    net = Net()
    before = [p.detach().clone() for p in net.parameters()]

    loss = train(net, fake_loader(), epochs=1, lr=0.01, device=CPU)

    assert isinstance(loss, float) and math.isfinite(loss)
    assert any(not torch.equal(b, a) for b, a in zip(before, net.parameters(), strict=True))


def test_evaluate_returns_loss_and_accuracy_in_range():
    loss, acc = evaluate(Net(), fake_loader(), device=CPU)
    assert math.isfinite(loss)
    assert 0.0 <= acc <= 1.0


def test_training_reduces_loss_on_a_fixed_batch():
    # A model that can't overfit 16 examples has a broken training loop
    net, loader = Net(), fake_loader()
    first = train(net, loader, epochs=1, lr=0.05, device=CPU)
    for _ in range(10):
        last = train(net, loader, epochs=1, lr=0.05, device=CPU)
    assert last < first
