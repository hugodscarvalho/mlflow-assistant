"""Integration tests for MLflowConnection.

This module contains integration tests for verifying the connection to a real
MLflow Tracking Server.
"""

import pytest
from mlflow_assistant.core.connection import MLflowConnection


@pytest.mark.integration()
def test_connect_to_real_server():
    """Test connection to a real MLflow Tracking Server.

    This test verifies that the MLflowConnection class can successfully connect
    to a running MLflow Tracking Server at the specified URI.

    Asserts:
        - The connection is successful.
        - The success message contains "Successfully connected".
        - The connection status is True.
    """
    conn = MLflowConnection(tracking_uri="http://localhost:5000")
    success, message = conn.connect()

    assert success is True
    assert "Successfully connected" in message
    assert conn.is_connected() is True
