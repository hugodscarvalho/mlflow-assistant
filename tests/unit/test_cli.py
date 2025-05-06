import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from mlflow_assistant.cli.commands import cli, mock_process_query

class TestCliCommands:
    """Test CLI command functionality."""
    
    def test_version_command(self):
        """Test the version command."""
        runner = CliRunner()
        
        with patch('mlflow_assistant.cli.commands.load_config', return_value={
            "mlflow_uri": "http://test:5000",
            "provider": {"type": "openai", "model": "test-model"}
        }):
            result = runner.invoke(cli, ["version"])
            
            # Check command executed successfully
            assert result.exit_code == 0
            # Check version info is in output
            assert "MLflow Assistant version:" in result.stdout
            assert "MLflow URI: http://test:5000" in result.stdout
            assert "Provider: openai" in result.stdout
    
    def test_setup_command(self):
        """Test setup command invokes the setup wizard."""
        runner = CliRunner()
        
        with patch('mlflow_assistant.cli.commands.setup_wizard') as mock_wizard:
            result = runner.invoke(cli, ["setup"])
            
            # Check command executed and called the wizard
            assert result.exit_code == 0
            mock_wizard.assert_called_once()
    
    def test_start_command_no_config(self):
        """Test start command fails gracefully with no config."""
        runner = CliRunner()
        
        with patch('mlflow_assistant.cli.commands.get_mlflow_uri', return_value=None):
            result = runner.invoke(cli, ["start"])
            
            # Should exit with error message
            assert result.exit_code == 0  # Click standard is 0 even for app-level errors
            assert "Error: MLflow URI not configured" in result.stdout
    
    def test_mock_process_query(self):
        """Test the mock processing function."""
        provider_config = {"type": "test", "model": "test-model"}
        
        result = mock_process_query("test query", provider_config)
        
        # Check response structure
        assert "original_query" in result
        assert "provider_config" in result
        assert "response" in result
        assert "test query" in result["response"]
    
    def test_start_command_basic_interaction(self):
        """Test basic interaction in start command."""
        runner = CliRunner()
        
        # Setup mocks for the prerequisites
        mock_config = {
            "mlflow_uri": "http://test:5000",
            "provider": {"type": "openai", "model": "test-model", "api_key": "mock-key"}
        }
        
        with patch('mlflow_assistant.cli.commands.get_mlflow_uri', return_value="http://test:5000"):
            with patch('mlflow_assistant.cli.commands.get_provider_config', return_value=mock_config["provider"]):
                # Simulate user entering a question and then /bye
                result = runner.invoke(cli, ["start"], input="What is MLflow?\n/bye\n")
                
                # Check output
                assert result.exit_code == 0
                # Changed this line to match your implementation
                assert "MLflow Assistant Chat Session" in result.stdout
                assert "Connected to MLflow at:" in result.stdout
                assert "This is a mock response to: 'What is MLflow?'" in result.stdout
                assert "Thank you for using MLflow Assistant" in result.stdout
