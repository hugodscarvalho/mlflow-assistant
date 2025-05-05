import pytest
import requests
import os

def is_mlflow_server_up(uri: str) -> bool:
    try:
        response = requests.get(f"{uri}/health", timeout=int(os.getenv("MLFLOW_HEALTH_TIMEOUT", 5)))
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"MLflow server health check failed: {e}")
        return False

@pytest.fixture(scope="session", autouse=True)
def check_mlflow_server():
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if not is_mlflow_server_up(uri):
        pytest.skip(f"Skipping integration tests — no MLflow server at {uri}", allow_module_level=True)