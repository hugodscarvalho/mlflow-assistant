from unittest.mock import patch
from mlflow_assistant.core.core import get_mlflow_client

def test_get_mlflow_client():
    with patch("mlflow_assistant.core.core.MlflowClient") as MockMlflowClient:
        client = get_mlflow_client()
        MockMlflowClient.assert_called_once()  # Ensure MlflowClient was called
        assert client == MockMlflowClient.return_value  # Ensure the returned client is the mock instance