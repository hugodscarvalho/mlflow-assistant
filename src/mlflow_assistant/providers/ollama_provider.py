"""Ollama provider for MLflow Assistant."""
import logging
from typing import Optional, Type

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from mlflow_assistant.utils.constants import DEFAULT_OLLAMA_URI, OllamaModel, Provider

from .base import AIProvider
from .definitions import (
    HUMAN_ROLE,
    JSON_FORMAT,
    STRUCTURED_TEMPERATURE,
    SYSTEM_ROLE,
    ParameterKeys,
)

logger = logging.getLogger("mlflow_assistant.engine.ollama")


class OllamaProvider(AIProvider):
    """Ollama provider implementation."""

    def __init__(self, uri=None, model=None, temperature=0.7, **kwargs):
        """Initialize the Ollama provider with URI and model."""
        # Handle None URI case to prevent attribute errors
        if uri is None:
            logger.warning(
                f"Ollama URI is None. Using default URI: {DEFAULT_OLLAMA_URI}"
            )
            self.uri = DEFAULT_OLLAMA_URI
        else:
            self.uri = uri.rstrip("/")

        self.model_name = model if model else OllamaModel.LLAMA32.value
        self.temperature = temperature

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
            f"Ollama provider initialized with model {self.model_name} at {self.uri}"
        )

    @property
    def langchain_model(self):
        """Get the underlying LangChain model."""
        return self.model

    def with_structured_output(
        self, output_schema: Type[BaseModel], method="function_calling"
    ):
        """Get a version of the langchain model with structured output."""
        # For Ollama, create a new instance with format="json" when using structured output
        model_params = {
            "base_url": self.uri,
            "model": self.model_name,
            "temperature": STRUCTURED_TEMPERATURE,  # Lower temperature for structured outputs
            "format": JSON_FORMAT,  # Ensure JSON format for structured outputs
        }

        # Add any additional parameters that were passed to the constructor
        for param in ParameterKeys.get_parameters(Provider.OLLAMA.value):
            if param in self.kwargs and self.kwargs[param] is not None:
                model_params[param] = self.kwargs[param]

        # Create a specialized model for structured output
        structured_model = ChatOllama(**model_params)

        # Always use parser method for Ollama models regardless of the method parameter
        return structured_model.with_structured_output(output_schema, method="parser")

    async def generate_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a response using Ollama."""
        try:
            logger.debug(f"Sending prompt to Ollama: {prompt[:50]}...")

            # If temperature is specified for this specific request, create a new model instance
            if temperature is not None and temperature != self.temperature:
                model_params = {
                    "base_url": self.uri,
                    "model": self.model_name,
                    "temperature": temperature,
                }

                # Add any additional parameters that were passed to the constructor
                for param in ParameterKeys.get_parameters(Provider.OLLAMA.value):
                    if param in self.kwargs and self.kwargs[param] is not None:
                        model_params[param] = self.kwargs[param]

                model = ChatOllama(**model_params)
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
            logger.error(f"Error generating response with Ollama: {e}")
            return f"Error generating response: {str(e)}"
