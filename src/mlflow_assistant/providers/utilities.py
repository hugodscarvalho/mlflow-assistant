"""Providers utilities."""
import logging
import subprocess  # noqa: S404

import shutil
import requests
from mlflow_assistant.utils.constants import DEFAULT_OLLAMA_URI

from .definitions import FALLBACK_MODELS

logger = logging.getLogger("mlflow_assistant.engine.utilities")


def verify_ollama_running(uri: str = DEFAULT_OLLAMA_URI) -> bool:
    """Verify if Ollama is running at the given URI."""
    try:
        response = requests.get(f"{uri}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def get_ollama_models(uri: str = DEFAULT_OLLAMA_URI) -> list:
    """Fetch the list of available Ollama models."""
    # Try using direct API call first
    try:
        response = requests.get(f"{uri}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("name") for m in data.get("models", [])]
            if models:
                return models
    except Exception as e:
        logger.debug(f"Failed to get Ollama models from API: {e}")

    try:
        # Execute the Ollama list command
        ollama_path = shutil.which("ollama")

        result = subprocess.run(  # noqa: S603
            [ollama_path, "list"], capture_output=True, text=True, check=False,
        )

        # Check if command executed successfully
        if result.returncode != 0:
            logger.warning(f"ollama list failed: {result.stderr}")
            return FALLBACK_MODELS

        # Parse the output to extract model names
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:  # Only header line or empty
            return FALLBACK_MODELS

        # Skip header line and extract the first column (model name)
        models = [line.split()[0] for line in lines[1:]]
        return models or FALLBACK_MODELS

    except (subprocess.SubprocessError, FileNotFoundError, IndexError) as e:
        logger.warning(f"Error fetching Ollama models: {e!s}")
        return FALLBACK_MODELS
