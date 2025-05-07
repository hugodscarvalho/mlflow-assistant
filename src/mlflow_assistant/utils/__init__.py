"""
Utility modules for MLflow Assistant.
"""

from .config import (
    load_config,
    save_config,
    get_mlflow_uri,
    get_provider_config
)

from .constants import (
    Provider,
    OpenAIModel,
    OllamaModel,
    Command,
    CONFIG_KEY_MLFLOW_URI,
    CONFIG_KEY_PROVIDER,
    CONFIG_KEY_TYPE,
    CONFIG_KEY_MODEL,
    CONFIG_KEY_URI,
    CONFIG_KEY_API_KEY,
    DEFAULT_MLFLOW_URI,
    DEFAULT_OLLAMA_URI,
    MLFLOW_URI_ENV,
    OPENAI_API_KEY_ENV,
    DEFAULT_STATUS_NOT_CONFIGURED,
    DEFAULT_STATUS_UNKNOWN,
    MLFLOW_CONNECTION_TIMEOUT,
    OLLAMA_CONNECTION_TIMEOUT,
    OLLAMA_TAGS_ENDPOINT,
    MLFLOW_VALIDATION_ENDPOINTS,
    CONFIG_DIRNAME,
    CONFIG_FILENAME,
    LOG_FORMAT,
)

__all__ = [
    # Config functions
    "load_config",
    "save_config",
    "get_mlflow_uri",
    "get_provider_config",
    # Enums
    "Provider",
    "OpenAIModel",
    "OllamaModel",
    "Command",
    # Constants
    "CONFIG_KEY_MLFLOW_URI",
    "CONFIG_KEY_PROVIDER",
    "CONFIG_KEY_TYPE",
    "CONFIG_KEY_MODEL",
    "CONFIG_KEY_URI",
    "CONFIG_KEY_API_KEY",
    "DEFAULT_MLFLOW_URI",
    "DEFAULT_OLLAMA_URI",
    "MLFLOW_URI_ENV",
    "OPENAI_API_KEY_ENV",
    "DEFAULT_STATUS_NOT_CONFIGURED",
    "DEFAULT_STATUS_UNKNOWN",
    "MLFLOW_CONNECTION_TIMEOUT",
    "OLLAMA_CONNECTION_TIMEOUT",
    "OLLAMA_TAGS_ENDPOINT",
    "MLFLOW_VALIDATION_ENDPOINTS",
    "CONFIG_DIRNAME",
    "CONFIG_FILENAME",
    "LOG_FORMAT",
]
