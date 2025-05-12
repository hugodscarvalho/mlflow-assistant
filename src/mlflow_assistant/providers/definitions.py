"""Constants for the MLflow Assistant providers."""
# Defaults Ollama
FALLBACK_MODELS = ["llama2", "mistral", "gemma", "phi"]

DATABRICKS_CREDENTIALS = ["DATABRICKS_TOKEN", "DATABRICKS_HOST"]


# Provider parameters
class ParameterKeys:
    """Keys and default parameter groupings for supported providers."""

    # Individual parameter keys
    TEMPERATURE = "temperature"
    MAX_TOKENS = "max_tokens"
    TIMEOUT = "timeout"
    MAX_RETRIES = "max_retries"
    ORGANIZATION = "organization"
    BASE_URL = "base_url"

    # Grouped by provider
    PARAMETERS_OPENAI = [MAX_TOKENS, TIMEOUT, MAX_RETRIES, ORGANIZATION, BASE_URL]
    PARAMETERS_OLLAMA = [MAX_TOKENS, TIMEOUT, MAX_RETRIES]
    PARAMETERS_DATABRICKS = [MAX_TOKENS]

    # All known parameters
    PARAMETERS_ALL = [TEMPERATURE] + PARAMETERS_OPENAI

    @classmethod
    def get_parameters(cls, provider: str) -> list[str]:
        """Return the list of parameters for the given provider name."""
        provider_map = {
            "openai": cls.PARAMETERS_OPENAI,
            "ollama": cls.PARAMETERS_OLLAMA,
            "databricks": cls.PARAMETERS_DATABRICKS,
        }
        return provider_map.get(provider.lower(), [])
