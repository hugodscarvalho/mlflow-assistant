"""Base class for AI providers."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from mlflow_assistant.utils.config import (
    CONFIG_KEY_API_KEY,
    CONFIG_KEY_MODEL,
    CONFIG_KEY_PROVIDER,
    CONFIG_KEY_TYPE,
    CONFIG_KEY_URI,
    Provider,
)

from .definitions import ParameterKeys

logger = logging.getLogger("mlflow_assistant.engine.base")


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    # Registry for provider classes
    _providers: Dict[str, Type["AIProvider"]] = {}

    def __init_subclass__(cls, **kwargs):
        """Auto-register provider subclasses."""
        super().__init_subclass__(**kwargs)
        # Register the provider using the class name
        provider_type = cls.__name__.lower().replace(CONFIG_KEY_PROVIDER, "")
        AIProvider._providers[provider_type] = cls
        logger.debug(f"Registered provider: {provider_type}")

    @property
    @abstractmethod
    def langchain_model(self):
        """Get the underlying LangChain model."""
        pass

    @classmethod
    def create(cls, config: Dict[str, Any]) -> "AIProvider":
        """Factory method to create an AI provider based on configuration."""
        provider_type = config.get(CONFIG_KEY_TYPE)

        if not provider_type:
            raise ValueError("Provider type not specified in configuration")

        provider_type = provider_type.lower()

        # Extract common parameters
        kwargs = {}
        for param in ParameterKeys.PARAMETERS_ALL.value:
            if param in config:
                kwargs[param] = config[param]

        # Import providers dynamically to avoid circular imports
        if provider_type == Provider.OPENAI.value:
            from .openai_provider import OpenAIProvider

            logger.info(
                f"Creating OpenAI provider with model {config.get(CONFIG_KEY_MODEL, Provider.get_default_model(Provider.OPENAI))}"
            )
            return OpenAIProvider(
                api_key=config.get(CONFIG_KEY_API_KEY),
                model=config.get(
                    CONFIG_KEY_MODEL, Provider.get_default_model(Provider.OPENAI)
                ),
                temperature=config.get(
                    ParameterKeys.TEMPERATURE.value,
                    Provider.get_default_temperature(Provider.OPENAI),
                ),
                **kwargs,
            )
        elif provider_type == Provider.OLLAMA.value:
            from .ollama_provider import OllamaProvider

            logger.info(
                f"Creating Ollama provider with model {config.get(CONFIG_KEY_MODEL)}"
            )
            return OllamaProvider(
                uri=config.get(CONFIG_KEY_URI),
                model=config.get(CONFIG_KEY_MODEL),
                temperature=config.get(
                    ParameterKeys.TEMPERATURE.value,
                    Provider.get_default_temperature(Provider.OLLAMA),
                ),
                **kwargs,
            )
        elif provider_type == Provider.DATABRICKS.value:
            from .databricks_provider import DatabricksProvider

            logger.info(
                f"Creating Databricks provider with model {config.get(CONFIG_KEY_MODEL)}"
            )
            return DatabricksProvider(
                model=config.get(CONFIG_KEY_MODEL),
                temperature=config.get(
                    ParameterKeys.TEMPERATURE.value,
                    Provider.get_default_temperature(Provider.DATABRICKS),
                ),
                **kwargs,
            )
        else:
            if provider_type not in cls._providers:
                raise ValueError(
                    f"Unknown provider type: {provider_type}. Available types: {', '.join(cls._providers.keys())}"
                )
            # Generic initialization for future providers
            provider_class = cls._providers[provider_type]
            return provider_class(config)
