# src/mlflow_assistant/cli/setup.py
import os
import click
import requests
import logging
import subprocess
import time
from pathlib import Path
from ..utils.config import load_config, save_config, ensure_config_dir, CONFIG_DIR

logger = logging.getLogger("mlflow_assistant.setup")

def validate_mlflow_uri(uri):
    """Validate MLflow URI by attempting to connect."""
    # Try multiple MLflow endpoints to validate the connection
    endpoints = [
        "/api/2.0/mlflow/experiments/list",  # Standard REST API
        "/ajax-api/2.0/mlflow/experiments/list",  # Alternative path
        "/",  # Root path (at least check if the server responds)
    ]
    
    for endpoint in endpoints:
        try:
            # Try with trailing slash trimmed
            clean_uri = uri.rstrip('/')
            url = f"{clean_uri}{endpoint}"
            logger.debug(f"Trying to connect to MLflow at: {url}")
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Successfully connected to MLflow at {url}")
                return True
            else:
                logger.debug(f"Response from {url}: {response.status_code}")
        except Exception as e:
            logger.debug(f"Failed to connect to {endpoint}: {str(e)}")
    
    # If we get here, none of the endpoints worked
    logger.warning(f"Could not validate MLflow at {uri} on any standard endpoint")
    return False

def setup_wizard():
    """Interactive setup wizard for mlflow-assistant."""
    click.echo("┌──────────────────────────────────────────────────────┐")
    click.echo("│             MLflow Assistant Setup Wizard            │")
    click.echo("└──────────────────────────────────────────────────────┘")
    
    click.echo("\nThis wizard will help you configure MLflow Assistant.")
    
    # Initialize config
    config = load_config()
    previous_provider = config.get('provider', {}).get('type')
    
    # MLflow URI
    mlflow_uri = click.prompt(
        "Enter your MLflow URI",
        default=config.get('mlflow_uri', 'http://localhost:5000')
    )
    
    if not validate_mlflow_uri(mlflow_uri):
        click.echo("\n⚠️  Warning: Could not connect to MLflow at the provided URI.")
        click.echo("    Please ensure MLflow is running and accessible at this address.")
        click.echo("    Common MLflow URLs: http://localhost:5000, http://localhost:8080")
        if not click.confirm("Continue anyway? (Choose Yes if you're sure MLflow is running)"):
            click.echo("Setup aborted. Please ensure MLflow is running and try again.")
            return
        else:
            click.echo("Continuing with setup using the provided MLflow URI.")
    else:
        click.echo("✅ Successfully connected to MLflow!")
    
    config['mlflow_uri'] = mlflow_uri
    
    # AI Provider
    provider_options = ['OpenAI', 'Ollama']
    provider_choice = click.prompt(
        "\nWhich AI provider would you like to use?",
        type=click.Choice(provider_options, case_sensitive=False),
        default=config.get('provider', {}).get('type', 'OpenAI').capitalize()
    )
    
    current_provider_type = provider_choice.lower()
    
    # Check if provider is changing and handle default models
    provider_changed = previous_provider and previous_provider != current_provider_type
    
    if provider_choice.lower() == 'openai':
        # Default OpenAI model
        default_model = "gpt-3.5-turbo"
        
        # If switching from another provider, show a message
        if provider_changed:
            click.echo(f"\n✅ Switching to OpenAI provider")
        
        # Initialize provider config
        provider_config = {
            'type': 'openai',
            'model': default_model  # Will be updated after user selection
        }
        
        # Check for OpenAI API key
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            click.echo("\n⚠️  OpenAI API key not found in environment variables.")
            click.echo("Please export your OpenAI API key as OPENAI_API_KEY.")
            click.echo("Example: export OPENAI_API_KEY='your-key-here'")
            if not click.confirm("Continue without API key?"):
                click.echo("Setup aborted. Please set the API key and try again.")
                return
        else:
            click.echo("✅ Found OpenAI API key in environment!")
        
        # Always ask for model choice
        model_options = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o']
        
        # If changing providers, suggest the default, otherwise use previous config
        if provider_changed:
            suggested_model = default_model
        else:
            current_model = config.get('provider', {}).get('model', default_model)
            suggested_model = current_model if current_model in model_options else default_model
        
        model = click.prompt(
            "Choose an OpenAI model",
            type=click.Choice(model_options, case_sensitive=False),
            default=suggested_model
        )
        provider_config['model'] = model
    
    elif provider_choice.lower() == 'ollama':
        # Default Ollama settings
        default_model = "llama3"
        default_uri = "http://localhost:11434"
        
        # If switching from another provider, automatically set defaults
        if provider_changed:
            click.echo(f"\n✅ Switching to Ollama provider with default URI and model")
        
        # Ollama configuration - always ask for URI
        ollama_uri = click.prompt(
            "\nEnter your Ollama server URI",
            default=config.get('provider', {}).get('uri', default_uri)
        )
        
        # Initialize provider config with default model and user-specified URI
        provider_config = {
            'type': 'ollama',
            'uri': ollama_uri,
            'model': default_model  # Will be updated if user selects a different model
        }
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{ollama_uri}/api/tags", timeout=2)
            if response.status_code == 200:
                click.echo("✅ Ollama server is running!")
                
                # Get available models
                try:
                    # Get models from Ollama
                    models_data = response.json()
                    available_models = [m.get('name') for m in models_data.get('models', [])]
                    
                    if available_models:
                        click.echo(f"\nAvailable Ollama models: {', '.join(available_models)}")
                        
                        # If changing providers, suggest the default, otherwise use previous config
                        if provider_changed:
                            suggested_model = default_model if default_model in available_models else available_models[0]
                        else:
                            current_model = config.get('provider', {}).get('model')
                            suggested_model = current_model if current_model in available_models else available_models[0]
                        
                        ollama_model = click.prompt(
                            "Choose an Ollama model",
                            type=click.Choice(available_models, case_sensitive=True),
                            default=suggested_model
                        )
                        provider_config['model'] = ollama_model
                    else:
                        ollama_model = click.prompt(
                            "Enter the Ollama model to use",
                            default=config.get('provider', {}).get('model', default_model)
                        )
                        provider_config['model'] = ollama_model
                except Exception as e:
                    click.echo(f"⚠️  Could not get available models: {e}")
                    ollama_model = click.prompt(
                        "Enter the Ollama model to use",
                        default=config.get('provider', {}).get('model', default_model)
                    )
                    provider_config['model'] = ollama_model
            else:
                click.echo("\n⚠️  Warning: Ollama server not running or not accessible at this URI.")
                if not click.confirm("Continue anyway?"):
                    click.echo("Setup aborted. Please start Ollama server and try again.")
                    return
                
                # Still prompt for model name
                ollama_model = click.prompt(
                    "Enter the Ollama model to use",
                    default=config.get('provider', {}).get('model', default_model)
                )
                provider_config['model'] = ollama_model
        except Exception as e:
            click.echo(f"\n⚠️  Warning: Could not connect to Ollama server: {e}")
            if not click.confirm("Continue anyway?"):
                click.echo("Setup aborted. Please start Ollama server and try again.")
                return
            
            # Still prompt for model name
            ollama_model = click.prompt(
                "Enter the Ollama model to use",
                default=config.get('provider', {}).get('model', default_model)
            )
            provider_config['model'] = ollama_model
    
    config['provider'] = provider_config
    
    # Save the configuration
    save_config(config)
    
    click.echo("\n✅ Configuration saved successfully!")
    click.echo("\n┌──────────────────────────────────────────────────┐")
    click.echo("│               Getting Started                    │")
    click.echo("└──────────────────────────────────────────────────┘")
    click.echo("\nYou can now use MLflow Assistant with the following commands:")
    click.echo("  mlflow-assistant start     - Start an interactive chat session with MLflow Assistant.")
    click.echo("  mlflow-assistant version   - Show MLflow Assistant version information.")
    
    click.echo("\nFor more information, use 'mlflow-assistant --help'")