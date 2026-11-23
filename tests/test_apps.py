"""Check the Flower ServerApp and ClientApp import and are wired up."""

from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp


def test_server_app_imports():
    from fedlearn.server_app import app

    assert isinstance(app, ServerApp)


def test_client_app_imports():
    from fedlearn.client_app import app

    assert isinstance(app, ClientApp)
