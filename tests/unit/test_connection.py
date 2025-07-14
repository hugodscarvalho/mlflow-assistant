"""Unit tests for the MLflow connection module.

This module contains unit tests for the `MLflowConnection` and `MLflowConnectionConfig`
classes, which are responsible for managing connections to MLflow Tracking Servers.
The tests cover various scenarios, including:

- Identifying connection types (local vs. remote).
- Initializing connections with default and custom parameters.
- Loading configuration from environment variables.
- Establishing successful connections to local and remote MLflow Tracking Servers.
- Handling connection failures gracefully.
- Retrieving connection information and client instances.

These tests ensure the robustness and correctness of the MLflow connection logic.
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
        """Test that the connection type is correctly identified as local.

        This test verifies that the connection type is set to LOCAL_CONNECTION
        when the tracking URI points to a local file system.
        """
        config = MLflowConnectionConfig(tracking_uri="file:///tmp/mlruns")
        assert config.connection_type == LOCAL_CONNECTION

        config = MLflowConnectionConfig(tracking_uri="/tmp/mlruns")
        assert config.connection_type == LOCAL_CONNECTION

    def test_connection_type_remote(self):
        """Test that the connection type is correctly identified as remote.

        This test verifies that the connection type is set to REMOTE_CONNECTION
        when the tracking URI points to a remote server.
        """
        config = MLflowConnectionConfig(tracking_uri="http://localhost:5000")
        assert config.connection_type == REMOTE_CONNECTION

        config = MLflowConnectionConfig(tracking_uri="http://localhost:5000")
        assert config.connection_type == REMOTE_CONNECTION


class TestMLflowConnection:
    """Tests for the MLflowConnection class."""

    def test_init_default(self):
        """Test the default initialization of MLflowConnection.

        This test verifies that the default tracking URI is set to
        DEFAULT_MLFLOW_TRACKING_URI and that the client is None.
        """
        conn = MLflowConnection()
        assert conn.config.tracking_uri == DEFAULT_MLFLOW_TRACKING_URI
        assert conn.client is None
        assert conn.is_connected() is False

    def test_init_with_params(self):
        """Test initialization of MLflowConnection with custom parameters.

        This test verifies that the tracking URI is correctly set when
        provided during initialization.
        """
        conn = MLflowConnection(tracking_uri="http://localhost:5000")
        assert conn.config.tracking_uri == "http://localhost:5000"

    def test_load_config_from_env(self, monkeypatch):
        """Test loading the tracking URI from environment variables.

        This test verifies that the tracking URI is correctly loaded
        from the MLFLOW_TRACKING_URI environment variable.
        """
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        conn = MLflowConnection()
        assert conn.config.tracking_uri == "http://localhost:5000"

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.tracking.MlflowClient")
    def test_connect_local_success(self, mock_client_class, mock_set_tracking_uri):
        """Test successful connection to a local MLflow Tracking Server.

        This test verifies that the connection is successful when the
        tracking URI points to a local file system.
        """
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
        """Test successful connection to a remote MLflow Tracking Server.

        This test verifies that the connection is successful when the
        tracking URI points to a remote server.
        """
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.search_experiments.return_value = []

        conn = MLflowConnection(
            tracking_uri="http://localhost:5000",
            client_factory=lambda tracking_uri: mock_client_instance,
        )
        success, message = conn.connect()

        assert success is True
        assert "Successfully connected" in message
        assert conn.is_connected() is True
        mock_set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_client_instance.search_experiments.assert_called_once()

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.tracking.MlflowClient")
    def test_connect_error(self, mock_client_class, mock_set_tracking_uri):
        """Test connection failure to an invalid MLflow Tracking Server.

        This test verifies that the connection fails and an appropriate
        error message is returned when the server is unreachable.
        """
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
        """Test getting the client when not connected.

        This test verifies that an MLflowConnectionError is raised when
        attempting to get the client without establishing a connection.
        """
        conn = MLflowConnection()
        with pytest.raises(MLflowConnectionError, match="Not connected"):
            conn.get_client()

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.tracking.MlflowClient")
    def test_get_client_connected(self, mock_client_class, mock_set_tracking_uri):
        """Test getting the client after a successful connection.

        This test verifies that the client is correctly returned after
        establishing a connection to the MLflow Tracking Server.
        """
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
        """Test retrieving connection information.

        This test verifies that the connection information is correctly
        returned, including the tracking URI, connection type, and
        connection status.
        """
        conn = MLflowConnection(tracking_uri="http://localhost:5000")
        info = conn.get_connection_info()
        assert info["tracking_uri"] == "http://localhost:5000"
        assert info["connection_type"] == REMOTE_CONNECTION
        assert info["is_connected"] is False

        with patch("mlflow.tracking.MlflowClient") as mock_client_class:
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.search_experiments.return_value = []
            conn.client_factory = lambda tracking_uri: mock_client_instance
            conn.connect()
            info = conn.get_connection_info()
            assert info["is_connected"] is True
