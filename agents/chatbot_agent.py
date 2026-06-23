"""
agents/chatbot_agent.py - The Main Chatbot Brain node
This is the agent that:
1. Reads the conversation history from state
2. Decides if it needs to search the web
3. Calls the LLM (Gemini) to generat a response
4. Returns the updated State

In LangGraph terms, this is a "node" - a function that takes state in
and returns updated state out. Simple as that.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.state import AgentState
from config import get_llm, MAX_HISTORY
from tools.search_tool import web_search
from database.profile_manager import load_profile

## -- System prompt 
# this is a "personality" and "instructions" for our chatbot.
# it is sent to Gemini every time as context.


SYSTEM_PROMPT = """You are a highly intelligent, helpful, and friendly AI assistant 
built with LangGraph and Google Gemini. 
Your capabilities:
- Answer questions on any topic clearly and accurately
- Use web search results when provided to give up-to-date information  
- Maintain context across the conversation (you remember what was said before)
- Be concise but thorough — give complete answers without being verbose
Rules:
- If web search results are provided in the context, USE them and cite the source
- If you are unsure about something, say so honestly
- Always be respectful and professional
- Format your responses clearly using markdown when helpful (bullet points, bold, etc.)
"""

def should_search_web(query: str) -> bool: 
    """
    Simple rule-based decision : should we search the web for this query ? 
    
    We search when the query contains the keywords that sugest the user wants 
    current / live information that the LLM may not know

    Args:
        query : the user s question

    Returns:
        True if we should search, False otherwise
    """

    query_lower = query.lower()

    #keywords that suggest the user wants fresh/ current information

    search_triggers = [
        "latest", "recent", "current", "today", "now", "2024", "2025", "2026"
        , "news", "update", "price", "weather", "stock", "who won", "what happened", "search", "find", "look up", "google"
    ]

    for trigger in search_triggers:
        if trigger in query_lower:
            return True
    return False


def create_summary(newly_old_messages:list, existing_summary:str, llm) -> str:
    """
    Summarizes the newly old messages and merges them with the existing summary.
    KEY POINT: newly_old_messages is exactly 2 messages(1 human + 1 ai), because each conversation turn adds exactly 2 messages to state
    We never re-send old messages as raw text - they are already compressed.
    
    Args :
        newly_old_messages : the last conversation turn (2 messages = 1 human + 1 ai)
        existing_summary : the previously accumulated summary 
        llm : the LLM instance to use for summarization
    
    Returns:
        new_summary : the updated summary(under 200 words)
    """

    #step 1 : build readable text from the new old messages only
    new_text = ""
    for msg in newly_old_messages:
        if isinstance(msg, HumanMessage):
            new_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            new_text += f"Assistant: {msg.content}\n"
    
    #step 2 : build the prompt for gemini
    if existing_summary:
        #we already have a summary - just UPDAT it with the few new messages 
        #we are not resending old messages, jst the new ones

        prompt = f"""You have a conversation summary. Update it by adding the new messages below.
Existing summary:
{existing_summary}
New messages to add (just these, nothing else):
{new_text}
Write an updated summary under 200 words. Keep all important info from the existing summary."""
    else:
        #first time summarizing - no existing summary yet
        prompt =  f"""Summarize this conversation in under 150 words.
Focus on names, topics discussed, key facts.
Conversation:
{new_text}"""
    
    # step 3: call gemini to make / update the summary
    response = llm.invoke([
        SystemMessage(content = "You are a conversation summarizer. Be brief and factual."),
        HumanMessage(content = prompt),
    ])
    return response.content





def chatbot_node(state: AgentState) -> dict:
    """ 
    The main chatbot agent node - the heart of Langgraph graph.
    what it does step by step:
    1. Reads the user s latest question from state
    2. Decides if web search is needed 
    3. if yes, searches the web and collects results
    4. Builds a prompt with conversation history + search results
    5. Sends to Gemini and gets a response
    6. Returns updated state fields

    Args:
        state: the current AgentState(shared notebook)

    Returns: 
        A dict with the fields we want to update in the state
    """

    llm = get_llm(temperature=0.7)

    user_query = state["user_query"]
    existing_messages = state.get("messages", [])

    # Step 1: decide if web search is needed

    needs_search = should_search_web(user_query)
    search_results = []

    # Step 2: Perform web search if needed 

    if needs_search:
        print(f"Searching web for {user_query}")
        search_results = web_search(user_query, max_results = 3)
    
    #Step 3: Build the prompt using summary + recent messages

    #read summary related fields from state
    existing_summary = state.get("conversation_summary", None)
    summarized_up_to = state.get("summarized_up_to", 0)

    if len(existing_messages) > MAX_HISTORY:
        # there are messages beyond the recent 10
        newly_old = existing_messages[summarized_up_to : len(existing_messages) - MAX_HISTORY]
        recent_messages = existing_messages[-MAX_HISTORY:]
        if newly_old:
            print(f"[Memory] Summarizing {len(newly_old)} new old messages...")
            new_summary = create_summary(newly_old, existing_summary or "", llm)
            new_summarized_up_to = len(existing_messages) - MAX_HISTORY
        else:
            new_summary = existing_summary
            new_summarized_up_to = summarized_up_to
    else:
        recent_messages = existing_messages
        new_summary = existing_summary
        new_summarized_up_to = summarized_up_to
        
    #build the single SystemMessage(Gemini ony allows one at position 0)-
    # we merge 3 things into one system message: 
    # 1. The base system prompt (personality + rules)
    # 2. The user profile (cross-session memory- who the user is)
    # 3. The conversation summary(within-session memory - what we talked about)

    system_content = SYSTEM_PROMPT

    #add user profile if one exists (loaded from profile.json on disk)
    user_profile = load_profile()
    if user_profile:
        system_content += (
            f"\n\n[What you know about the user - use naturally dont assume it]\n" + 
            user_profile + "\n[end of user profile]"
        )
    #add within session summary if it exists
    if new_summary:
        system_content += (
            f"\n\n[earlier conversation summary - use this as background context]\n" + 
            new_summary + "\n[End of summary]"
        )

    # Now build the messages list — only ONE SystemMessage, everything merged in
    messages_to_send = [SystemMessage(content=system_content)]
    messages_to_send.extend(recent_messages)




    #if we search results , inject them into the prompt
    if search_results:
        search_context = "\n\n---\n".join(search_results)
        search_message = HumanMessage(
            content = f"""[Web search Results for: "{user_query}"]{search_context} 
            -- Now answer this question using the search results above {user_query}"""
        )

        messages_to_send.append(search_message)
    else:
        #no search - just send the user question directly 

        messages_to_send.append(HumanMessage(content=user_query))

    #step 4: Call gemini
    print(f"[Chatbot] Calling gemini... (search = {'yes' if needs_search else 'no'})")
    response = llm.invoke(messages_to_send)
    answer = response.content

    #step 5: return updated state
    
    # we returna a dict - LangGraph merges this into the shared state automatically

    return {
        "messages": [HumanMessage(content = user_query),
        AIMessage(content = answer), 
        ],
        "search_results" : search_results,
        "should_search" : needs_search, 
        "final_response" : answer,
        "conversation_summary" : new_summary, 
        "summarized_up_to" : new_summarized_up_to
    }
