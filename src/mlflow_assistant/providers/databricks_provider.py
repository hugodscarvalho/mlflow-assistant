"""Databricks provider for MLflow Assistant."""
import logging
import os

from databricks_langchain import ChatDatabricks
from mlflow_assistant.utils.constants import Provider

from .base import AIProvider
from .definitions import DATABRICKS_CREDENTIALS, ParameterKeys

logger = logging.getLogger("mlflow_assistant.engine.databricks")


class DatabricksProvider(AIProvider):
    """Databricks provider implementation."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs,
    ):
        """Initialize the Databricks provider with model."""
        self.model_name = (
            model or Provider.get_default_model(Provider.DATABRICKS.value)
        )
        self.temperature = (
            temperature or Provider.get_default_temperature(Provider.DATABRICKS.value)
        )
        self.kwargs = kwargs

        for var in DATABRICKS_CREDENTIALS:
            if var not in os.environ:
                logger.warning(
                    f"Missing environment variable: {var}. "
                    "Responses may fail if you are running outside Databricks.",
                )

        # Build parameters dict with only non-None values
        model_params = {"endpoint": self.model_name, "temperature": temperature}

        # Only add optional parameters if they're not None
        for param in ParameterKeys.get_parameters(Provider.DATABRICKS.value):
            if param in kwargs and kwargs[param] is not None:
                model_params[param] = kwargs[param]

        # Initialize with parameters matching the documentation
        self.model = ChatDatabricks(**model_params)

        logger.info(f"Databricks provider initialized with model {self.model_name}")

    def langchain_model(self):
        """Get the underlying LangChain model."""
        return self.model
