import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make sure mlflow_assistant package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for configuration files."""
    temp_dir = tempfile.mkdtemp()
    old_env = dict(os.environ)
    
    # Set environment variable to override config dir
    os.environ["MLFLOW_ASSISTANT_CONFIG_DIR"] = temp_dir
    
    # This line would be added to config.py to support testing
    # CONFIG_DIR = Path(os.environ.get("MLFLOW_ASSISTANT_CONFIG_DIR", Path.home() / ".mlflow-assistant"))
    
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
        "provider": {
            "type": "openai",
            "model": "test-model"
        }
    }