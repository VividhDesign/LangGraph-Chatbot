"""
graph/state.py - Defines the shared state for the LangGraph chatbot.

NOTE: We use 'chat_history' (not 'messages') because newer LangGraph versions
treat any field named 'messages' as special and auto-apply add_messages reducer,
which causes compatibility crashes. Renaming avoids all of that magic.
"""

from __future__ import annotations
from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Full conversation history (plain list - we manage appending ourselves)
    chat_history: list

    # The current user question
    user_query: str

    # Web search results (empty list if no search was done)
    search_results: list

    # Did the chatbot decide to search the web?
    should_search: bool

    # The final answer produced by the chatbot
    final_response: Optional[str]

    # Unique session ID
    session_id: str

    # Running summary of older messages
    conversation_summary: Optional[str]

    # How many messages have been compressed into the summary
    summarized_up_to: int
