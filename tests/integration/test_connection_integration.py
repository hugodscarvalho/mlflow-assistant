import pytest
from mlflow_assistant.core.connection import MLflowConnection

@pytest.mark.integration
def test_connect_to_real_server():
    """Test connection to a real MLflow Tracking Server."""
    conn = MLflowConnection(tracking_uri="http://localhost:5000")
    success, message = conn.connect()

    assert success is True
    assert "Successfully connected" in message
    assert conn.is_connected() is True