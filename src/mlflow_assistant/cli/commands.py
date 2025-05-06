import click
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Internal imports
from ..utils.config import load_config, get_mlflow_uri, get_provider_config
from .setup import setup_wizard

# Set up logging
logger = logging.getLogger("mlflow_assistant.cli")

# Mock function for process_query since it's not implemented yet
def mock_process_query(query: str, provider_config: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Mock function that simulates the query processing workflow.
    This will be replaced with the actual implementation later.
    
    Args:
        query: The user's query
        provider_config: The AI provider configuration
        verbose: Whether to show verbose output
        
    Returns:
        Dictionary with mock response information
    """
    # Create a mock response
    return {
        "original_query": query,
        "provider_config": {
            "type": provider_config.get("type", "unknown"),
            "model": provider_config.get("model", "unknown"),
        },
        "enhanced": False,
        "response": f"This is a mock response to: '{query}'\n\nThe MLflow integration will be implemented soon!"
    }

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """MLflow Assistant: Interact with MLflow using LLMs.
    
    This CLI tool helps you query and interact with MLflow resources using natural language.
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@cli.command()
def setup():
    """Run the interactive setup wizard.
    
    This wizard helps you configure MLflow Assistant with your MLflow server and preferred AI provider.
    """
    setup_wizard()

@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Show verbose output')
def start(verbose):
    """Start an interactive chat session with MLflow Assistant.
    
    This opens an interactive chat session where you can ask multiple questions about 
    your MLflow experiments, models, and data. Type /bye to exit the session.
    
    Examples of questions you can ask:
    - What are my best performing models for classification?
    - Show me details of experiment 'customer_churn'
    - Compare runs abc123 and def456
    - Which hyperparameters should I try next for my regression model?
    
    Commands:
    - /bye: Exit the chat session
    - /help: Show help about available commands
    - /clear: Clear the screen
    """
    mlflow_uri = get_mlflow_uri()
    if not mlflow_uri:
        click.echo("❌ Error: MLflow URI not configured. Run 'mlflow-assistant setup' first.")
        return
    
    # Get provider config
    provider_config = get_provider_config()
    if not provider_config or not provider_config.get('type'):
        click.echo("❌ Error: AI provider not configured. Run 'mlflow-assistant setup' first.")
        return
    
    # Ensure OpenAI has an API key if that's the configured provider
    if provider_config.get('type') == 'openai' and not provider_config.get('api_key'):
        click.echo("❌ Error: OpenAI API key not found in environment. Set OPENAI_API_KEY.")
        return
    
    # Print welcome message and instructions
    provider_type = provider_config.get('type', 'unknown')
    model = provider_config.get('model', 'default')
    
    click.echo(f"\n🤖 MLflow Assistant Chat Session")
    click.echo(f"Connected to MLflow at: {mlflow_uri}")
    click.echo(f"Using {provider_type.upper()} with model: {model}")
    click.echo(f"\nType your questions and press Enter. Type /bye to exit.")
    click.echo("=" * 70)
    
    # Start interactive loop
    history = []
    while True:
        # Get user input with a prompt
        try:
            query = click.prompt("\n🧑", prompt_suffix="").strip()
        except (KeyboardInterrupt, EOFError):
            click.echo("\nExiting chat session...")
            break
        
        # Handle special commands
        if query.lower() == '/bye':
            click.echo("\nThank you for using MLflow Assistant! Goodbye.")
            break
        elif query.lower() == '/help':
            click.echo("\nAvailable commands:")
            click.echo("  /bye   - Exit the chat session")
            click.echo("  /help  - Show this help message")
            click.echo("  /clear - Clear the screen")
            continue
        elif query.lower() == '/clear':
            # This is a simple approximation of clear screen
            click.echo("\n" * 50)
            continue
        elif not query:
            continue  # Skip empty queries
        
        # Process the query
        try:
            # In the actual implementation, this would call the process_query function
            result = mock_process_query(query, provider_config, verbose)
            
            # Add to history
            history.append({"query": query, "response": result["response"]})
            
            # Display response
            click.echo(f"\n🤖 {result['response']}")
            
            # Show verbose info if requested
            if verbose:
                click.echo("\n--- Debug Information ---")
                click.echo(f"Provider: {provider_type}")
                click.echo(f"Model: {model}")
                click.echo(f"Query processed with mock function")
                click.echo("-------------------------")
                
        except Exception as e:
            click.echo(f"\n❌ Error processing query: {str(e)}")

@cli.command()
def version():
    """Show MLflow Assistant version information."""
    from .. import __version__
    
    click.echo(f"MLflow Assistant version: {__version__}")
    
    # Show configuration
    config = load_config()
    mlflow_uri = config.get('mlflow_uri', 'Not configured')
    provider = config.get('provider', {}).get('type', 'Not configured')
    model = config.get('provider', {}).get('model', 'default')
    
    click.echo(f"MLflow URI: {mlflow_uri}")
    click.echo(f"Provider: {provider}")
    click.echo(f"Model: {model}")

if __name__ == "__main__":
    cli()