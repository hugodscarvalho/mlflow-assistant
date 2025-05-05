"""
Tests for the MLflow connection module.
"""

import pytest
from unittest.mock import patch

from mlflow_assistant.core.connection import MLflowConnection
from mlflow_assistant.utils.definitions import (
    MLflowConnectionConfig,
    DEFAULT_MLFLOW_TRACKING_URI,
    LOCAL_CONNECTION,
    REMOTE_CONNECTION,
)
from mlflow_assistant.utils.exceptions import MLflowConnectionError


class TestMLflowConnectionConfig:
    """Tests for the MLflowConnectionConfig class."""

    def test_connection_type_local(self):
        config = MLflowConnectionConfig(tracking_uri="file:///tmp/mlruns")
        assert config.connection_type == LOCAL_CONNECTION

        config = MLflowConnectionConfig(tracking_uri="/tmp/mlruns")
        assert config.connection_type == LOCAL_CONNECTION

    def test_connection_type_remote(self):
        config = MLflowConnectionConfig(tracking_uri="http://localhost:5000")
        assert config.connection_type == REMOTE_CONNECTION

        config = MLflowConnectionConfig(tracking_uri="https://mlflow.example.com")
        assert config.connection_type == REMOTE_CONNECTION


class TestMLflowConnection:
    """Tests for the MLflowConnection class."""

    def test_init_default(self):
        conn = MLflowConnection()
        assert conn.config.tracking_uri == DEFAULT_MLFLOW_TRACKING_URI
        assert conn.client is None
        assert conn.is_connected() is False

    def test_init_with_params(self):
        conn = MLflowConnection(tracking_uri="http://mlflow.example.com")
        assert conn.config.tracking_uri == "http://mlflow.example.com"

    def test_load_config_from_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-mlflow.example.com")
        conn = MLflowConnection()
        assert conn.config.tracking_uri == "http://env-mlflow.example.com"

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.tracking.MlflowClient")
    def test_connect_local_success(self, mock_client_class, mock_set_tracking_uri):
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.search_experiments.return_value = []

        conn = MLflowConnection(
            tracking_uri="file:///tmp/mlruns",
            client_factory=lambda tracking_uri: mock_client_instance,
        )
        success, message = conn.connect()

        assert success is True
        assert "Successfully connected" in message
        assert conn.is_connected() is True
        mock_set_tracking_uri.assert_called_once_with("file:///tmp/mlruns")
        mock_client_instance.search_experiments.assert_called_once()

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.tracking.MlflowClient")
    def test_connect_remote_success(self, mock_client_class, mock_set_tracking_uri):
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.search_experiments.return_value = []

        conn = MLflowConnection(
            tracking_uri="http://mlflow.example.com",
            client_factory=lambda tracking_uri: mock_client_instance,
        )
        success, message = conn.connect()

        assert success is True
        assert "Successfully connected" in message
        assert conn.is_connected() is True
        mock_set_tracking_uri.assert_called_once_with("http://mlflow.example.com")
        mock_client_instance.search_experiments.assert_called_once()

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.tracking.MlflowClient")
    def test_connect_error(self, mock_client_class, mock_set_tracking_uri):
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.search_experiments.side_effect = Exception("Connection failed")

        conn = MLflowConnection(
            tracking_uri="http://invalid-server",
            client_factory=lambda tracking_uri: mock_client_instance,
        )
        success, message = conn.connect()

        assert success is False
        assert "Failed to connect" in message
        assert conn.is_connected() is False

    def test_get_client_not_connected(self):
        conn = MLflowConnection()
        with pytest.raises(MLflowConnectionError, match="Not connected"):
            conn.get_client()

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.tracking.MlflowClient")
    def test_get_client_connected(self, mock_client_class, mock_set_tracking_uri):
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.search_experiments.return_value = []

        conn = MLflowConnection(
            tracking_uri="file:///tmp/mlruns",
            client_factory=lambda tracking_uri: mock_client_instance,
        )
        conn.connect()

        client = conn.get_client()
        assert client == mock_client_instance

    def test_get_connection_info(self):
        conn = MLflowConnection(tracking_uri="http://mlflow.example.com")
        info = conn.get_connection_info()
        assert info["tracking_uri"] == "http://mlflow.example.com"
        assert info["connection_type"] == REMOTE_CONNECTION
        assert info["is_connected"] is False

        with patch("mlflow.tracking.MlflowClient") as mock_client_class:
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.search_experiments.return_value = []
            conn.client_factory = lambda tracking_uri: mock_client_instance
            conn.connect()
            info = conn.get_connection_info()
            assert info["is_connected"] is True