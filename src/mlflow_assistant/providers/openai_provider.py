"""OpenAI provider for MLflow Assistant."""
import logging

from langchain_openai import ChatOpenAI
from mlflow_assistant.utils.constants import OpenAIModel, Provider

from .base import AIProvider
from .definitions import ParameterKeys

logger = logging.getLogger("mlflow_assistant.engine.openai")


class OpenAIProvider(AIProvider):
    """OpenAI provider implementation."""

    def __init__(
        self,
        api_key=None,
        model=OpenAIModel.GPT35.value,
        temperature: float | None = None,
        **kwargs,
    ):
        """Initialize the OpenAI provider with API key and model."""
        self.api_key = api_key
        self.model_name = model or OpenAIModel.GPT35.value
        self.temperature = (
            temperature or Provider.get_default_temperature(Provider.OPENAI.value)
        )
        self.kwargs = kwargs

        if not self.api_key:
            logger.warning("No OpenAI API key provided. Responses may fail.")

        # Build parameters dict with only non-None values
        model_params = {
            "api_key": api_key,
            "model": self.model_name,
            "temperature": temperature,
        }

        # Only add optional parameters if they're not None
        for param in ParameterKeys.get_parameters(Provider.OLLAMA.value):
            if param in kwargs and kwargs[param] is not None:
                model_params[param] = kwargs[param]

        # Initialize with parameters matching the documentation
        self.model = ChatOpenAI(**model_params)

        logger.debug(f"OpenAI provider initialized with model {self.model_name}")

    def langchain_model(self):
        """Get the underlying LangChain model."""
        return self.model
