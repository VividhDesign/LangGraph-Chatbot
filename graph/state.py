"""
graph/state.py - Defines the "shared notebook" that
all agents read and write to.

In LangGraph, every node reads from state and writes back to state.
Think of it like a shared Google Doc that everyone updates as the chat progresses.
"""

from __future__ import annotations
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    This is the SHARED STATE - the "memory" of the entire conversation graph.
    Fields:
         messages : Full Conversation History(Langgraph auto-manages this)
         user_query : The latest question the user typed
         search_results : Results from DuckDuckGo web Search (if triggered)
         should_search : True if the chatbot decides to search the web
         final_response : The finished answer to show the user 
         session_id : Unique ID per chat session (for memory isolation)
    """

    #add_messages is a special LangGraph reducer: it automatically appends a new message instead of overwriting

    messages : Annotated[list[BaseMessage], add_messages]
    
    # the current user question
    user_query : str

    #web search results (empty list if no search was done)
    search_results : list[str]

    #did the chatbot decide to search the web ? True or false 
    should_search : bool

    #the final anser produced by the chatbot 
    final_response : Optional[str]

    #unique session ID (so different users dont mix up their chat histories)
    session_id : str 

    #running summary of messages older than the last 10. 
    #llm writes this summary so we never forget old context. 

    conversation_summary : Optional[str]

    #Tracks how many messages from the beginning have been COMPRESSED
    #old messaes are still represented - but as compressed text inside 
    #conversation_summary, NOT re-sent as raw text to Gemini again
    #each turn we only pass the newy old messages(always 2 : 1 human + 1 ai)
    #as raw text to the summarizer , then merge with existing summary
    #starts at 0 (nothing compressed yet).
    summarized_up_to : int




