"""Ollama provider for MLflow Assistant."""
import logging

from langchain_ollama import ChatOllama
from mlflow_assistant.utils.constants import DEFAULT_OLLAMA_URI, OllamaModel, Provider

from .base import AIProvider
from .definitions import ParameterKeys

logger = logging.getLogger("mlflow_assistant.engine.ollama")


class OllamaProvider(AIProvider):
    """Ollama provider implementation."""

    def __init__(self, uri=None, model=None, temperature=None, **kwargs):
        """Initialize the Ollama provider with URI and model."""
        # Handle None URI case to prevent attribute errors
        if uri is None:
            logger.warning(
                f"Ollama URI is None. Using default URI: {DEFAULT_OLLAMA_URI}",
            )
            self.uri = DEFAULT_OLLAMA_URI
        else:
            self.uri = uri.rstrip("/")

        self.model_name = model or OllamaModel.LLAMA32.value
        self.temperature = (
            temperature or Provider.get_default_temperature(Provider.OLLAMA.value)
        )

        # Store kwargs for later use when creating specialized models
        self.kwargs = kwargs

        # Build parameters dict with only non-None values
        model_params = {
            "base_url": self.uri,
            "model": self.model_name,
            "temperature": temperature,
        }

        # Only add optional parameters if they're not None
        for param in ParameterKeys.get_parameters(Provider.OLLAMA.value):
            if param in kwargs and kwargs[param] is not None:
                model_params[param] = kwargs[param]

        # Use langchain-ollama's dedicated ChatOllama class
        self.model = ChatOllama(**model_params)

        logger.info(
            f"Ollama provider initialized with model {self.model_name} at {self.uri}",
        )

    def langchain_model(self):
        """Get the underlying LangChain model."""
        return self.model
