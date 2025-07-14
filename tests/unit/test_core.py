"""Unit tests for the core utilities in MLflow Assistant.

This module contains unit tests for the core functionality provided by the
`mlflow_assistant.core.core` module, including utilities for interacting
with MLflow clients.
"""

from unittest.mock import patch
from mlflow_assistant.core.core import get_mlflow_client


def test_get_mlflow_client():
    """Test the `get_mlflow_client` function.

    This test verifies that the `get_mlflow_client` function correctly
    initializes and returns an instance of `MlflowClient`. It ensures
    that the `MlflowClient` constructor is called exactly once and that
    the returned client is the mock instance.
    """
    with patch("mlflow_assistant.core.core.MlflowClient") as mock_mlflow_client:
        client = get_mlflow_client()
        mock_mlflow_client.assert_called_once()  # Ensure MlflowClient was called
        assert client == mock_mlflow_client.return_value  # Ensure the returned client is the mock instance
