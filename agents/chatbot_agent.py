"""
agents/chatbot_agent.py - The main chatbot brain node.

This agent:
1. Reads the conversation history from state
2. Decides if a web search is needed
3. Calls the LLM (Gemini or Groq) to generate a response
4. Returns the updated state

In LangGraph terms, this is a "node" — a function that takes state
in and returns updated state out.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from graph.state import AgentState
from config import get_llm, MAX_HISTORY
from tools.search_tool import web_search
from database.profile_manager import load_profile


def extract_text(content) -> str:
    """
    Safely extracts plain text from LLM response content.
    Newer langchain-google-genai returns a list of dicts like:
        [{'type': 'text', 'text': '...'}]
    Older versions and Groq return a plain string.
    This handles both so the app works locally and on Streamlit Cloud.
    """
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


SYSTEM_PROMPT = """You are a highly intelligent, helpful, and friendly AI assistant
built with LangGraph.
Your capabilities:
- Answer questions on any topic clearly and accurately
- Use web search results when provided to give up-to-date information
- Maintain context across the conversation
- Be concise but thorough — complete answers without being verbose

Rules:
- If web search results are provided in the context, USE them and cite the source
- If you are unsure about something, say so honestly
- Always be respectful and professional
- Format responses clearly using markdown when helpful (bullet points, bold, etc.)
"""


def should_search_web(query: str) -> bool:
    """
    Rule-based decision: should we search the web for this query?
    Returns True if the query contains keywords suggesting the user
    wants current/live information.
    """
    query_lower = query.lower()
    search_triggers = [
        "latest", "recent", "current", "today", "now",
        "2024", "2025", "2026",
        "news", "update", "price", "weather", "stock",
        "who won", "what happened", "search", "find", "look up", "google",
    ]
    return any(trigger in query_lower for trigger in search_triggers)


def create_summary(newly_old_messages: list, existing_summary: str, llm) -> str:
    """
    Summarizes newly-old messages and merges them with the existing summary.
    Called when conversation history exceeds MAX_HISTORY.

    Args:
        newly_old_messages: The messages being rotated out of the active window.
        existing_summary: The previously accumulated summary.
        llm: The LLM instance to use for summarization.

    Returns:
        Updated summary string (under 200 words).
    """
    new_text = ""
    for msg in newly_old_messages:
        if isinstance(msg, HumanMessage):
            new_text += f"User: {extract_text(msg.content)}\n"
        elif isinstance(msg, AIMessage):
            new_text += f"Assistant: {extract_text(msg.content)}\n"

    if existing_summary:
        prompt = f"""You have a conversation summary. Update it by adding the new messages below.
Existing summary:
{existing_summary}

New messages to add:
{new_text}

Write an updated summary under 200 words. Keep all important info from the existing summary."""
    else:
        prompt = f"""Summarize this conversation in under 150 words.
Focus on names, topics discussed, key facts.
Conversation:
{new_text}"""

    response = llm.invoke([
        SystemMessage(content="You are a conversation summarizer. Be brief and factual."),
        HumanMessage(content=prompt),
    ])
    return extract_text(response.content)


def chatbot_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    The main chatbot agent node — the heart of the LangGraph graph.

    Steps:
    1. Reads the user's latest question from state
    2. Decides if web search is needed
    3. If yes, searches the web and collects results
    4. Builds a prompt with conversation history + search results
    5. Calls the LLM and returns the updated state

    Args:
        state: The current AgentState (shared notebook)
        config: LangGraph RunnableConfig containing optional model/api_key/provider overrides

    Returns:
        A dict with the updated state fields
    """
    cfg = (config or {}).get("configurable", {})
    selected_model = cfg.get("model", None)
    api_key = cfg.get("api_key", None)
    provider = cfg.get("provider", "gemini")

    llm = get_llm(temperature=0.7, model=selected_model, api_key=api_key, provider=provider)

    user_query = state["user_query"]
    existing_messages = state.get("chat_history", [])
    if not isinstance(existing_messages, list):
        existing_messages = []

    existing_summary = state.get("conversation_summary", None)
    if existing_summary is not None and not isinstance(existing_summary, str):
        existing_summary = str(existing_summary) if existing_summary else None

    summarized_up_to = state.get("summarized_up_to", 0)
    if not isinstance(summarized_up_to, int):
        summarized_up_to = 0

    # Step 1: Decide if web search is needed
    needs_search = should_search_web(user_query)
    search_results = []

    # Step 2: Perform web search if needed
    if needs_search:
        print(f"[Search] Searching web for: {user_query}")
        search_results = web_search(user_query, max_results=3)

    # Step 3: Handle message summarisation if history is too long
    if len(existing_messages) > MAX_HISTORY:
        newly_old = existing_messages[summarized_up_to: len(existing_messages) - MAX_HISTORY]
        recent_messages = existing_messages[-MAX_HISTORY:]
        if newly_old:
            print(f"[Memory] Summarising {len(newly_old)} old messages...")
            new_summary = create_summary(newly_old, existing_summary or "", llm)
            new_summarized_up_to = len(existing_messages) - MAX_HISTORY
        else:
            new_summary = existing_summary
            new_summarized_up_to = summarized_up_to
    else:
        recent_messages = existing_messages
        new_summary = existing_summary
        new_summarized_up_to = summarized_up_to

    # Step 4: Build the system message (personality + user profile + summary)
    system_content = SYSTEM_PROMPT

    user_profile = load_profile()
    if user_profile:
        system_content += f"\n\n[What you know about the user — use naturally, don't announce it]\n{user_profile}\n[End of user profile]"
    if new_summary:
        system_content += f"\n\n[Earlier conversation summary — use as background context]\n{new_summary}\n[End of summary]"

    messages_to_send = [SystemMessage(content=system_content)]
    messages_to_send.extend(recent_messages)

    # Step 5: Inject web search results or just the user question
    if search_results:
        search_context = "\n\n---\n".join(search_results)
        messages_to_send.append(HumanMessage(
            content=f'[Web Search Results for: "{user_query}"]\n{search_context}\n\n-- Now answer using the search results above: {user_query}'
        ))
    else:
        messages_to_send.append(HumanMessage(content=user_query))

    # Step 6: Call the LLM
    print(f"[Chatbot] Calling {provider} ({selected_model or 'default'})... (search={'yes' if needs_search else 'no'})")
    response = llm.invoke(messages_to_send)
    answer = extract_text(response.content)

    # Step 7: Return updated state
    updated_messages = list(existing_messages) + [
        HumanMessage(content=user_query),
        AIMessage(content=answer),
    ]

    return {
        "chat_history": updated_messages,
        "search_results": search_results,
        "should_search": needs_search,
        "final_response": answer,
        "conversation_summary": new_summary,
        "summarized_up_to": new_summarized_up_to,
    }
