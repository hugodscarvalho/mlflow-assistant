"""Tests for the core MLflow Assistant functionality.

This module contains tests for the core module, including MLflow client
initialization and basic functionality testing.
"""
from unittest.mock import patch
from mlflow_assistant.core.core import get_mlflow_client


def test_get_mlflow_client():
    """Test that get_mlflow_client returns a properly initialized MLflow client."""
    with patch("mlflow_assistant.core.core.MlflowClient") as mock_mlflow_client:
        client = get_mlflow_client()
        mock_mlflow_client.assert_called_once()  # Ensure MlflowClient was called
        assert (
            client == mock_mlflow_client.return_value
        )  # Ensure the returned client is the mock instance
