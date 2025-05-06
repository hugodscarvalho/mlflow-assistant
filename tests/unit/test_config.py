import os
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from mlflow_assistant.utils.config import (
    load_config, save_config, get_mlflow_uri, get_provider_config, ensure_config_dir
)

class TestConfig:
    """Test configuration loading and saving."""
    
    def test_ensure_config_dir(self, temp_config_dir):
        """Test creating the configuration directory."""
        # Remove temp dir to simulate it not existing yet
        os.rmdir(temp_config_dir)
        
        # Now call the function that should create it
        with patch('mlflow_assistant.utils.config.CONFIG_DIR', Path(temp_config_dir)):
            ensure_config_dir()
        
        # Check that it was created
        assert os.path.exists(temp_config_dir)
    
    def test_save_and_load_config(self, temp_config_dir):
        """Test saving and loading configuration."""
        test_config = {
            "mlflow_uri": "http://test-server:5000",
            "provider": {
                "type": "openai",
                "model": "gpt-3.5-turbo"
            }
        }
        
        # Mock the CONFIG_FILE path
        config_file = Path(temp_config_dir) / "config.yaml"
        
        # Test saving
        with patch('mlflow_assistant.utils.config.CONFIG_FILE', config_file):
            with patch('mlflow_assistant.utils.config.CONFIG_DIR', Path(temp_config_dir)):
                save_config(test_config)
                
                # Test loading
                loaded_config = load_config()
                
                # Compare configs
                assert loaded_config == test_config
    
    def test_get_mlflow_uri_from_config(self, mock_config):
        """Test getting MLflow URI from config."""
        with patch('mlflow_assistant.utils.config.load_config', return_value=mock_config):
            # Test with no environment variable
            with patch.dict('os.environ', {}, clear=True):
                uri = get_mlflow_uri()
                assert uri == "http://test-mlflow:5000"
    
    def test_get_mlflow_uri_from_env(self, mock_config):
        """Test that environment variable takes precedence."""
        with patch('mlflow_assistant.utils.config.load_config', return_value=mock_config):
            # Test with environment variable set
            with patch.dict('os.environ', {"MLFLOW_TRACKING_URI": "http://env-uri:9000"}):
                uri = get_mlflow_uri()
                assert uri == "http://env-uri:9000"
    
    def test_get_provider_config_openai(self, mock_config):
        """Test getting OpenAI provider config."""
        with patch('mlflow_assistant.utils.config.load_config', return_value=mock_config):
            provider = get_provider_config()
            assert provider["type"] == "openai"
            assert provider["model"] == "test-model"
    
    def test_get_provider_config_with_env_key(self, mock_config):
        """Test that OpenAI key from environment takes precedence."""
        with patch('mlflow_assistant.utils.config.load_config', return_value=mock_config):
            with patch.dict('os.environ', {"OPENAI_API_KEY": "test-key-from-env"}):
                provider = get_provider_config()
                assert provider["api_key"] == "test-key-from-env"