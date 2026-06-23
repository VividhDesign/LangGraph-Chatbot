"""
graph/chatbot_graph.py - Assembles the LangGraph computation graph.

Think of this file as the "blueprint" or "flowChart" of the chatbot.
We define:
       -Which nodes(agents/steps) exist
       -How they connect to each other 
       -Where the flow starts (entry point) and ends(END)

Our graph is simple : START->chatbot->END

"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from graph.state import AgentState
from agents.chatbot_agent import chatbot_node

def build_graph():
    """
    Builds and compiles the langGraph chatbot graph

    Steps:
    1. Create a stateGraph using our Agentstate as the shared state schema
    2. Add the chatbot_node as a node named "chatbot"
    3. Add edges: START->chatbot->END
    4. Add a memorySaver Checkpointer (enables multi-turn memory across sessions)
    5. Compile and return the ready to use graph

    Returns:
        A compiled LangGraph graph object
    """

    #Step 1:create the graph blueprint
    # StateGraph takes our TypedDict class - it knows what field exists in state
    graph_builder = StateGraph(AgentState)

    #step 2 : add nodes 
    graph_builder.add_node("chatbot", chatbot_node)

    # step 3 : add edges

    graph_builder.add_edge(START, "chatbot")

    graph_builder.add_edge("chatbot", END)


    #step 4: add checkpointer for memory
    # SqliteSaver stores conversation state in a file(chatbot_memory.db).
    # This means memory SURVIVES app restarts unlike MemorySaver(RAM only)
    # each unique thread_id (session_id) still gets its own isolated history
    conn = sqlite3.connect("chatbot_memory.db", check_same_thread = False)
    checkpointer = SqliteSaver(conn)

    #step 5 : compile the graph

    graph = graph_builder.compile(checkpointer = checkpointer)

    print("[Graph] LangGraph chatbot graph built successfully!")
    return graph

# ── Convenience: pre-build graph once when this module is imported ────────────
# This way, the graph is only compiled once, not on every user message.
chatbot_graph = build_graph()


def run_chat(user_message: str, session_id: str) -> dict:
    """
    Runs one turn of the chatbot conversation.
    
    This is the function streamlit will call every time the user seads a messages
    Args:
        user_message : The text the user typed 
        session_id : Unique ID for this user s sesion(keeps histories separate)
    
    returns: 
        A dict with keys: - "response" : the ai s answer(string)
                          - "used_search" : t/f - did it search the web? 
                          - "search_results" : list of search results string(may be empty)
    """

    #langgraph uses a "config" to identify sessions via thread_id
    config = {"configurable" : {"thread_id": session_id}}

     # The initial state we provide — only set the fields we know now.
    # LangGraph will fill in the rest using defaults from prior checkpoints.
    initial_state = {
        "user_query": user_message,
        "messages": [],            # will be loaded from checkpoint by LangGraph
        "search_results": [],
        "should_search": False,
        "final_response": None,
        "session_id": session_id,
        # NOTE: conversation_summary and summarized_up_to are NOT set here
        # so LangGraph loads them from the checkpoint on subsequent calls
    }

    #invoke() runs the graph synchronously and returns the final state
    final_state = chatbot_graph.invoke(initial_state, config = config)

    return {
        "response": final_state.get("final_response", "Sorry, I couldn't generate a response."),
        "used_search": final_state.get("should_search", False),
        "search_results": final_state.get("search_results", []),
    }