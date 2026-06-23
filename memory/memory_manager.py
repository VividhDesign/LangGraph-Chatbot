"""
memory/memory_manager.py - Manages conversation history per session.

why do we  need this ? 
Every time a user types a message, streamlit re-runs the whole Python script , 
from top to bottom.
We use st.session_state to persist data across re-runs

This file provides helper functions to :
   - Get the full history for the current session
   - Add a new message (user, or assistant) to the history
   - Clear/reset the history
"""


from langchain_core.messages import HumanMessage, AIMessage

def get_history(session_state: dict) -> list:
    """
    Gets the conversation message history from streamlit session state.

    Args:
        session_state: Streamlit's st.session_state dictionary

    Returns:
        A list of LangChain Message objects(HumanMessage / AIMessage)
    """

    if "messages" not in session_state:
        session_state["messages"] = []
    return session_state["messages"]

def add_user_message(session_state:dict, content:str) -> None:
    """
    Adds a new message to the conversation history .
    Args:
        session_state: Streamlit s st.session_state dictionary
        content : The text the user typed
    """

    if "messages" not in session_state:
        session_state["messages"] = []
    session_state["messages"].append(HumanMessage(content = content))


def add_ai_message(session_state: dict, content: str) -> None:
    """
    Adds an AI response to the conversation history.
    Args:
        session_state: Streamlits st.session_state dictionary 
        content: The text the AI replied with
    """

    if "messages" not in session_state:
        session_state["messages"] = []
    session_state["messages"].append(AIMessage(content = content))


def clear_history(session_state: dict) -> None:
    """
    Clears all conversations history (fresh start).
    Args:
        session_state: Streamlits st.session_state dictionary
    """
    session_state["messages"] = []


def trim_history(session_state:dict, max_messages:int = 10) -> None:
    """
    Keeps only the last N messages to avoid sending too much to LLM
    This prevents the context window from getting too large.
    Args:
        session_state : Streamlits st.session_state dictionary
        max_message : Maximum number of messages to keep
    """

    if "messages" in session_state:
        if len(session_state["messages"]) > max_messages:

            #keep only the most recent messages
            session_state["messages"] = session_state["messages"][-max_messages:]
