"""Pytest configuration and shared fixtures for MLflow Assistant tests.

This module provides common test fixtures and configuration for both unit
and integration tests, including temporary directories for configuration
files and mock configurations.
"""
import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Make sure mlflow_assistant package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for configuration files."""
    temp_dir = tempfile.mkdtemp()
    old_env = dict(os.environ)

    # Set environment variable to override config dir
    os.environ["MLFLOW_ASSISTANT_CONFIG_DIR"] = temp_dir

    yield temp_dir

    # Cleanup
    os.environ.clear()
    os.environ.update(old_env)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration with test values."""
    return {
        "mlflow_uri": "http://test-mlflow:5000",
        "provider": {"type": "openai", "model": "test-model"},
    }
