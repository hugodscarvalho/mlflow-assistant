"""
Query processor that leverages the workflow engine for processing user queries and generating responses using an AI provider.
"""
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage

logger = logging.getLogger("mlflow_assistant.engine.processor")


async def process_query(
    query: str, provider_config: Dict[str, Any], verbose: bool = False
) -> Dict[str, Any]:
    """
    Process a query through the MLflow Assistant workflow.

    Args:
        query: The query to process
        provider_config: AI provider configuration
        verbose: Whether to show verbose output

    Returns:
        Dict containing the response
    """
    import time

    from .workflow import create_workflow

    # Track start time for duration calculation
    start_time = time.time()

    try:
        # Create workflow
        workflow = create_workflow()

        # Run workflow with provider config
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "provider_config": provider_config,
        }

        if verbose:
            logger.info(f"Running workflow with query: {query}")
            logger.info(f"Using provider: {provider_config.get('type')}")
            logger.info(f"Using model: {provider_config.get('model', 'default')}")

        result = await workflow.ainvoke(initial_state)

        # Calculate duration
        duration = time.time() - start_time

        response = {
            "original_query": query,
            "response": result.get("messages")[-1],
            "duration": duration,  # Add duration to response
        }

        return response

    except Exception as e:
        # Calculate duration even for errors
        duration = time.time() - start_time

        logger.error(f"Error processing query: {e}")
        error_response = {
            "error": str(e),
            "original_query": query,
            "response": f"Error processing query: {str(e)}",
        }

        return error_response
