"""
graph/state.py - Defines the "shared notebook" that
all agents read and write to.

In LangGraph, every node reads from state and writes back to state.
Think of it like a shared Google Doc that everyone updates as the chat progresses.

NOTE: We use plain `list` for messages (not `add_messages` reducer) because
the add_messages reducer in newer LangGraph versions has compatibility issues.
We manage message appending manually inside chatbot_node instead.
"""

from __future__ import annotations
from typing import Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    The SHARED STATE - the "memory" of the entire conversation graph.
    """

    # Full conversation history - plain list, we manage appending ourselves
    messages: list

    # The current user question
    user_query: str

    # Web search results (empty list if no search was done)
    search_results: list

    # Did the chatbot decide to search the web?
    should_search: bool

    # The final answer produced by the chatbot
    final_response: Optional[str]

    # Unique session ID (so different users don't mix up their chat histories)
    session_id: str

    # Running summary of messages older than the last MAX_HISTORY.
    conversation_summary: Optional[str]

    # How many messages from the start have been compressed into the summary
    summarized_up_to: int
