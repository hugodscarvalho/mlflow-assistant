"""Integration test configuration for MLflow Assistant.

This module provides shared fixtures and utilities for integration tests, including
a health check to ensure the MLflow Tracking Server is running before tests are executed.
"""
import pytest
import requests
import os


def is_mlflow_server_up(uri: str) -> bool:
    """Check if the MLflow server is running.

    Args:
    ----
        uri (str): The URI of the MLflow server.

    Returns:
    -------
        bool: True if the server is running and responds with status code 200, False otherwise.

    """
    try:
        response = requests.get(f"{uri}/health", timeout=int(os.getenv("MLFLOW_HEALTH_TIMEOUT", 5)))
        return response.status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture(scope="session", autouse=True)
def check_mlflow_server():
    """Fixture to check if the MLflow server is running before executing integration tests.

    This fixture runs once per test session and skips all integration tests if the
    MLflow server is not running.

    Raises
    ------
        pytest.skip: If the MLflow server is not running.

    """
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if not is_mlflow_server_up(uri):
        pytest.skip(f"Skipping integration tests — no MLflow server at {uri}", allow_module_level=True)
