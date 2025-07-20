"""Core LangGraph-based workflow engine for processing user queries and generating responses using an AI provider.

This workflow supports tool-augmented generation: tool calls are detected and executed in a loop
until a final AI response is produced.
"""
import logging
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from mlflow_assistant.engine.definitions import (
    STATE_KEY_MESSAGES,
    STATE_KEY_PROVIDER_CONFIG,
)
from providers import AIProvider
from tools import get_model_details, get_system_info, list_experiments, list_models
from typing_extensions import TypedDict

# Configure logging to ensure output appears in console
logger = logging.getLogger("mlflow_assistant.engine.workflow")

# Define available tools
tools = [list_models, list_experiments, get_model_details, get_system_info]


# Define the state schema
class State(TypedDict):
    """State schema for the workflow engine."""

    messages: Annotated[list[BaseMessage], add_messages]
    provider_config: dict[str, Any]  # Model/provider configuration
    mlflow_uri: str  # MLflow URI


# Workflow creation function
def create_workflow():
    """Create and return a compiled LangGraph workflow."""
    graph_builder = StateGraph(State)

    def call_model(state: State) -> State:
        """Call the AI model and return updated state with response."""
        messages = state[STATE_KEY_MESSAGES]
        provider_config = state.get(STATE_KEY_PROVIDER_CONFIG, {})
        try:
            provider = AIProvider.create(provider_config)
            model = provider.langchain_model().bind_tools(tools)
            response = model.invoke(messages)
            return {**state, STATE_KEY_MESSAGES: [response]}
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return {**state, STATE_KEY_MESSAGES: messages}

    # Add nodes
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_node("model", call_model)

    # Define graph transitions
    graph_builder.add_edge("tools", "model")
    graph_builder.add_conditional_edges("model", tools_condition)
    graph_builder.set_entry_point("model")

    return graph_builder.compile()
