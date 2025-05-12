"""OpenAI provider for MLflow Assistant."""
import logging
from typing import Optional, Type

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from mlflow_assistant.utils.constants import OpenAIModel, Provider

from .base import AIProvider
from .definitions import HUMAN_ROLE, STRUCTURED_TEMPERATURE, SYSTEM_ROLE, ParameterKeys

logger = logging.getLogger("mlflow_assistant.engine.openai")


class OpenAIProvider(AIProvider):
    """OpenAI provider implementation."""

    def __init__(
        self,
        api_key=None,
        model=OpenAIModel.GPT35.value,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the OpenAI provider with API key and model."""
        self.api_key = api_key
        self.model_name = model if model else OpenAIModel.GPT35.value
        self.temperature = (
            temperature
            if temperature
            else Provider.get_default_temperature(Provider.OPENAI.value)
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

        logger.info(f"OpenAI provider initialized with model {self.model_name}")

    @property
    def langchain_model(self):
        """Get the underlying LangChain model."""
        return self.model

    def with_structured_output(
        self, output_schema: Type[BaseModel], method="function_calling"
    ):
        """Get a version of the langchain model with structured output."""
        # For OpenAI, create a new instance with lower temperature for structured outputs
        model_params = {
            "api_key": self.api_key,
            "model": self.model_name,
            "temperature": STRUCTURED_TEMPERATURE,  # Lower temperature for structured outputs
        }

        # Add any additional parameters that were passed to the constructor
        for param in ParameterKeys.get_parameters(Provider.OPENAI.value):
            if param in self.kwargs and self.kwargs[param] is not None:
                model_params[param] = self.kwargs[param]

        # Create a specialized model for structured output
        structured_model = ChatOpenAI(**model_params)

        # Now apply the with_structured_output method
        return structured_model.with_structured_output(output_schema, method=method)

    async def generate_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a response using OpenAI."""
        try:
            logger.debug(f"Sending prompt to OpenAI: {prompt[:50]}...")

            # If temperature is specified for this specific request, create a new model instance
            if temperature is not None and temperature != self.temperature:
                model_params = {
                    "api_key": self.api_key,
                    "model": self.model_name,
                    "temperature": temperature,
                }

                # Add any additional parameters that were passed to the constructor
                for param in ParameterKeys.get_parameters(Provider.OLLAMA.value):
                    if param in self.kwargs and self.kwargs[param] is not None:
                        model_params[param] = self.kwargs[param]

                model = ChatOpenAI(**model_params)
            else:
                model = self.model

            # Build messages array using the tuple format
            messages = []

            # Add system message if provided
            if system_message:
                messages.append((SYSTEM_ROLE, system_message))

            # Add user message
            messages.append((HUMAN_ROLE, prompt))

            # Invoke the model asynchronously
            response = await model.ainvoke(messages)

            return response.content
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            return f"Error generating response: {str(e)}"
