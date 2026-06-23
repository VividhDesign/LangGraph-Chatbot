"""
graph/chatbot_graph.py - Assembles the LangGraph computation graph.

Uses MemorySaver (RAM-based checkpointing) instead of SqliteSaver.
This avoids corrupted state issues on cloud deployments — state resets
cleanly on each app restart, which is fine since Streamlit Cloud is ephemeral.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import AgentState
from agents.chatbot_agent import chatbot_node


def build_graph():
    """Builds and compiles the LangGraph chatbot graph."""

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # MemorySaver: RAM-only checkpointing.
    # Pro: no disk corruption, no version-compat issues with sqlite schema.
    # Con: memory resets on app restart (acceptable for cloud deployment).
    checkpointer = MemorySaver()
    graph = graph_builder.compile(checkpointer=checkpointer)

    print("[Graph] LangGraph chatbot graph built successfully!")
    return graph


# Pre-build once when module is imported
chatbot_graph = build_graph()


def run_chat(user_message: str, session_id: str) -> dict:
    """
    Runs one turn of the chatbot conversation.
    Returns: {"response": str, "used_search": bool, "search_results": list}
    """
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "user_query": user_message,
        "chat_history": [],
        "search_results": [],
        "should_search": False,
        "final_response": None,
        "session_id": session_id,
        "conversation_summary": None,
        "summarized_up_to": 0,
    }

    final_state = chatbot_graph.invoke(initial_state, config=config)

    return {
        "response": final_state.get("final_response", "Sorry, I couldn't generate a response."),
        "used_search": final_state.get("should_search", False),
        "search_results": final_state.get("search_results", []),
    }