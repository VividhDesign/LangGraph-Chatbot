"""
app.py — Streamlit frontend for the LangGraph Chatbot.
This is the ONLY file you need to run to start the chatbot.
It provides a beautiful chat interface that:
  - Loads API keys silently from .env or Streamlit Cloud secrets
  - Shows a sidebar with model selector, session list, and stats
  - Displays the full conversation history
  - Sends user messages to the LangGraph graph
  - Streams responses word-by-word (like ChatGPT)
  - Shows an indicator when web search is used
  - Auto-renames new chats based on the first prompt using AI
  - Lets you clear the chat history
"""

import streamlit as st
import uuid
import json
import os
import time
import traceback
from datetime import datetime
from database.profile_manager import load_profile, save_profile, update_profile

# Page configuration — must be the first Streamlit command
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Beautiful Dark Theme ─────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background (ChatGPT-like) */
    .stApp {
        background: #212121 !important;
        color: #ececf1 !important;
    }

    /* Main header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 600;
        color: #ececf1;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #9b9b9b;
        font-size: 1rem;
        font-weight: 400;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #171717 !important;
        border-right: 1px solid #2f2f2f;
    }
    .sidebar-section h4 {
        color: #9b9b9b;
        margin-bottom: 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .stat-item {
        display: flex;
        justify-content: space-between;
        color: #9b9b9b;
        font-size: 0.83rem;
        padding: 0.2rem 0;
    }
    .stat-value {
        color: #ececf1;
        font-weight: 500;
    }

    /* Chat message containers — dark theme polish */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin: 0.2rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* Chat input area */
    [data-testid="stChatInput"] textarea {
        background: #2f2f2f !important;
        border: 1px solid #424242 !important;
        border-radius: 12px !important;
        color: #ececf1 !important;
        font-size: 1rem !important;
        caret-color: #10a37f !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #565869 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #6b7280 !important;
    }
    /* Chat input send button */
    [data-testid="stChatInput"] button {
        background: #10a37f !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* Selectbox (model picker) */
    [data-baseweb="select"] > div {
        background: #2f2f2f !important;
        border: 1px solid #424242 !important;
        border-radius: 8px !important;
        color: #ececf1 !important;
    }

    /* Buttons */
    .stButton > button {
        background: #10a37f !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: opacity 0.2s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* Hide default Streamlit elements */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }

    /* Hide the << collapse button — it's not needed and causes confusion */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarHeader"] button,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-header"],
    [data-testid="stSidebar"] [data-testid="baseButton-header"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Force sidebar open: Streamlit 1.50 saves collapsed state in localStorage.
# st.components.v1.html actually runs JS (unlike st.markdown which strips scripts).
import streamlit.components.v1 as components
components.html("""
<script>
// Clear any saved "sidebar collapsed" state from localStorage
Object.keys(localStorage).forEach(key => {
    if (key.startsWith('stSidebarCollapsed')) {
        localStorage.removeItem(key);
    }
});
</script>
""", height=0, width=0)



# ── API Key Helper ────────────────────────────────────────────────────────────
def get_api_key(provider: str = "gemini") -> str:
    """
    Gets the API key from Streamlit Cloud secrets or environment variables.
    No user input needed — key is pre-configured by the app owner.
    """
    if provider == "groq":
        try:
            return st.secrets["GROQ_API_KEY"]
        except (KeyError, FileNotFoundError, AttributeError):
            pass
        return os.getenv("GROQ_API_KEY", "")
    else:
        try:
            return st.secrets["GOOGLE_API_KEY"]
        except (KeyError, FileNotFoundError, AttributeError):
            pass
        return os.getenv("GOOGLE_API_KEY", "")


# ── Persistent sessions: save and load from JSON ──────────────────────────────
SESSIONS_FILE = "sessions_data.json"


def save_sessions_to_disk():
    """Saves all sessions to a JSON file on disk."""
    try:
        with open(SESSIONS_FILE, "w") as f:
            json.dump(st.session_state.sessions, f, indent=2)
    except Exception as e:
        print(f"[Sessions] Could not save: {e}")


def load_sessions_from_disk():
    """Loads all sessions from disk. Returns empty dict if no file exists."""
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ── Helper: create a new empty chat session ───────────────────────────────────
def make_new_session():
    """Creates a new chat session named 'New Chat' and returns its ID."""
    new_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.sessions[new_id] = {
        "name": "New Chat",
        "history": [],
        "created_at": timestamp,
    }
    return new_id


# ── Helper: get active session data ──────────────────────────────────────────
def get_active_session():
    """Returns the dict for the currently open session."""
    return st.session_state.sessions[st.session_state.active_session]


# ── Helper: auto-rename chat using AI after first exchange ────────────────────
def auto_rename_chat(session_id: str, first_user_msg: str, first_ai_msg: str, provider: str = "gemini"):
    """
    Calls the LLM to generate a short (≤5 words) title for the chat based
    on the first user message + AI reply. Updates the session name in place.
    """
    try:
        from config import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = get_llm(provider=provider)
        prompt = (
            f"User asked: \"{first_user_msg}\"\n"
            f"Assistant replied: \"{first_ai_msg[:200]}\"\n\n"
            "Give this conversation a short title (maximum 5 words, no punctuation, no quotes). "
            "Only return the title, nothing else."
        )
        response = llm.invoke([
            SystemMessage(content="You are a chat title generator. Be concise and descriptive."),
            HumanMessage(content=prompt),
        ])
        raw = response.content
        # Handle list-of-dicts content (newer langchain-google-genai)
        if isinstance(raw, list):
            title = " ".join(
                part.get("text", "") for part in raw
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            title = str(raw)
        title = title.strip().strip('"').strip("'")
        if title and len(title) < 60:
            st.session_state.sessions[session_id]["name"] = title
            save_sessions_to_disk()
            print(f"[Chat] Auto-renamed session {session_id} → '{title}'")
    except Exception as e:
        print(f"[Chat] Auto-rename failed: {e}")


# ── Streaming helper ─────────────────────────────────────────────────────────
def stream_response(text, placeholder):
    """
    Displays text word-by-word inside a Streamlit placeholder,
    with a blinking cursor (▌) — exactly like ChatGPT.
    """
    displayed = ""
    words = text.split(' ')
    for i, word in enumerate(words):
        displayed += word + (' ' if i < len(words) - 1 else '')
        placeholder.markdown(displayed + " ▌")
        time.sleep(0.03)
    # Final render without cursor
    placeholder.markdown(displayed)


# ── Session state initialization ──────────────────────────────────────────────
if "sessions" not in st.session_state:
    loaded = load_sessions_from_disk()
    st.session_state.sessions = loaded if loaded else {}

if "active_session" not in st.session_state:
    first_id = make_new_session()
    st.session_state.active_session = first_id

if "total_searches" not in st.session_state:
    st.session_state.total_searches = 0

if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0

if "provider" not in st.session_state:
    st.session_state.provider = "gemini"

if "selected_model" not in st.session_state:
    from config import DEFAULT_MODEL
    st.session_state.selected_model = DEFAULT_MODEL
if "selected_groq_model" not in st.session_state:
    from config import DEFAULT_GROQ_MODEL
    st.session_state.selected_groq_model = DEFAULT_GROQ_MODEL


# ── Dynamic model fetching ────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_available_models(api_key: str, provider: str):
    """Dynamically fetches all available models for the given provider."""
    if provider == "groq":
        if not api_key:
            return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        try:
            import requests
            headers = {"Authorization": f"Bearer {api_key}"}
            res = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
            data = res.json()
            models = [m["id"] for m in data.get("data", [])]
            if models:
                return sorted(models)
        except Exception as e:
            print(f"[API] Could not fetch Groq models: {e}")
        from config import get_available_groq_models
        return get_available_groq_models()

    else:
        # Gemini
        if not api_key:
            return ["gemini-3.1-flash-lite", "gemini-3.1-flash"]
        try:
            import requests
            res = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
            data = res.json()
            models = [
                m['name'].replace('models/', '')
                for m in data.get('models', [])
                if 'generateContent' in m.get('supportedGenerationMethods', [])
            ]
            gemini_models = sorted([m for m in models if "gemini" in m], reverse=True)
            if gemini_models:
                return gemini_models
        except Exception as e:
            print(f"[API] Could not fetch Gemini models: {e}")

        from config import get_available_models
        return get_available_models()


# ── Load LangGraph (only once) ────────────────────────────────────────────────
@st.cache_resource
def load_graph():
    """Loads and compiles the LangGraph chatbot. Cached so it only runs once."""
    try:
        from graph.chatbot_graph import run_chat
        return run_chat
    except Exception as e:
        st.error(f"Failed to load LangGraph: {e}")
        return None

run_chat_fn = load_graph()


def update_profile_from_session(history: list):
    """Save profile from session history. Called on New Chat / session switch."""
    if not history:
        return
    try:
        from config import get_llm
        profile_llm = get_llm(provider=st.session_state.provider)
        session_text = ""
        for m in history:
            role = "User" if m["role"] == "user" else "Assistant"
            session_text += f"{role}: {m['content']}\n"
        existing = load_profile()
        new_prof = update_profile(session_text, existing, profile_llm)
        save_profile(new_prof)
        print("[Profile] Saved from session.")
    except Exception as e:
        print(f"[Profile] Update failed: {e}")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 LangGraph Chatbot")
    st.markdown("*Powered by Gemini + LangGraph*")
    st.markdown("---")

    # ── New Chat Button ───────────────────────────────────────────────────────
    if st.button("✏️ New Chat", use_container_width=True):
        update_profile_from_session(get_active_session()["history"])
        new_id = make_new_session()
        st.session_state.active_session = new_id
        st.session_state.total_messages = 0
        st.session_state.total_searches = 0
        st.rerun()

    st.markdown("---")

    # ── LLM Provider & Model Selector ─────────────────────────────────────────
    st.markdown("### 🧠 LLM Settings")

    new_provider = st.radio("Provider", ["gemini", "groq"], index=0 if st.session_state.provider=="gemini" else 1, horizontal=True)
    if new_provider != st.session_state.provider:
        st.session_state.provider = new_provider
        st.rerun()

    ak = get_api_key(st.session_state.provider)
    available_models = fetch_available_models(ak, st.session_state.provider)

    # Use the appropriate session state variable for the selected provider
    if st.session_state.provider == "gemini":
        if st.session_state.selected_model not in available_models:
            st.session_state.selected_model = available_models[0]
        current_idx = available_models.index(st.session_state.selected_model)
    else:
        if st.session_state.selected_groq_model not in available_models:
            st.session_state.selected_groq_model = available_models[0]
        current_idx = available_models.index(st.session_state.selected_groq_model)

    chosen_model = st.selectbox(
        "Model",
        options=available_models,
        index=current_idx,
        label_visibility="collapsed",
        help="Select which model to use for responses",
    )

    if st.session_state.provider == "gemini":
        if chosen_model != st.session_state.selected_model:
            st.session_state.selected_model = chosen_model
            st.rerun()
    else:
        if chosen_model != st.session_state.selected_groq_model:
            st.session_state.selected_groq_model = chosen_model
            st.rerun()

    st.markdown(
        f"<div style='color:#6b7280; font-size:0.78rem; margin-top:-0.3rem; margin-bottom:0.5rem;'>"
        f"Dynamic model fetched from Google API</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── Conversation History List ─────────────────────────────────────────────
    st.markdown("### 💬 Your Chats")

    all_sessions = list(st.session_state.sessions.items())
    all_sessions.reverse()

    for sid, sdata in all_sessions:
        is_active = (sid == st.session_state.active_session)
        msg_count = len(sdata["history"])
        display_label = f"{'🟣' if is_active else '⚫'} {sdata['name']} ({msg_count} msgs)"

        col_chat, col_del = st.columns([5, 1])
        with col_chat:
            if st.button(display_label, key=f"session_btn_{sid}", use_container_width=True):
                update_profile_from_session(get_active_session()["history"])
                st.session_state.active_session = sid
                h = sdata["history"]
                st.session_state.total_messages = len(h)
                st.session_state.total_searches = sum(
                    1 for m in h if m.get("used_search") and m["role"] == "ai"
                )
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_btn_{sid}", help="Delete this chat"):
                del st.session_state.sessions[sid]
                if st.session_state.active_session == sid:
                    if st.session_state.sessions:
                        st.session_state.active_session = list(st.session_state.sessions.keys())[-1]
                    else:
                        new_id = make_new_session()
                        st.session_state.active_session = new_id
                    st.session_state.total_messages = 0
                    st.session_state.total_searches = 0
                save_sessions_to_disk()
                st.rerun()

    st.markdown("---")

    # ── Stats for current session ─────────────────────────────────────────────
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", st.session_state.total_messages)
    with col2:
        st.metric("Searches", st.session_state.total_searches)

    st.markdown(f"**Session ID:** `{st.session_state.active_session}`")
    st.markdown("---")

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown("### 🔍 How It Works")
    st.markdown("""
    1. **You** type a message  
    2. **LangGraph** routes it to the chatbot node  
    3. **Chatbot** decides: search web or answer directly?  
    4. **Gemini** generates the response  
    5. **Memory** stores the conversation  
    """)
    st.markdown("---")

    # ── Controls ──────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Controls")
    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        get_active_session()["history"] = []
        st.session_state.total_messages = 0
        st.session_state.total_searches = 0
        st.success("Chat cleared!")
        st.rerun()

    st.markdown("---")
    active_mod = st.session_state.selected_model if st.session_state.provider == "gemini" else st.session_state.selected_groq_model
    st.markdown(f"""
    <div style="color: #6b7280; font-size: 0.8rem; text-align: center;">
        Built with LangGraph + Streamlit<br>
        {active_mod}
    </div>
    """, unsafe_allow_html=True)


# ── MAIN PAGE ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 LangGraph Chatbot</h1>
    <p>An intelligent multi-turn chatbot with web search · built with LangGraph & Gemini</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Chat History Display ──────────────────────────────────────────────────────
history = get_active_session()["history"]

if not history:
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 3rem 0;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">💬</div>
        <div style="font-size: 1.1rem; font-weight: 500; color: #9ca3af;">
            Start a conversation!
        </div>
        <div style="font-size: 0.9rem; margin-top: 0.5rem; color: #6b7280;">
            Ask me anything — I can search the web for the latest information.
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in history:
    role = "assistant" if msg["role"] == "ai" else "user"
    avatar = "🤖" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar):
        if msg.get("used_search") and role == "assistant":
            st.caption("🔍 Web Search Used")
        st.markdown(msg["content"])

# ── Chat Input (pinned at bottom, sends on Enter) ────────────────────────────
if prompt := st.chat_input("Type your message and press Enter..."):
    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Add user message to history
    get_active_session()["history"].append({
        "role": "user",
        "content": prompt,
        "used_search": False,
    })
    st.session_state.total_messages += 1

    # ── Generate AI response ──────────────────────────────────────────────────
    ai_response = ""
    used_search = False

    if run_chat_fn is None:
        ai_response = "❌ Error: LangGraph could not be loaded. Check your configuration."
        with st.chat_message("assistant", avatar="🤖"):
            st.error(ai_response)
    else:
        try:
            active_mod = (
                st.session_state.selected_model
                if st.session_state.provider == "gemini"
                else st.session_state.selected_groq_model
            )

            # Spinner shows OUTSIDE chat_message so it doesn't block streaming
            with st.spinner("🤔 Thinking..."):
                result = run_chat_fn(
                    user_message=prompt,
                    session_id=st.session_state.active_session,
                    model=active_mod,
                    provider=st.session_state.provider,
                )

            ai_response = result["response"]
            used_search = result["used_search"]

            # Now stream inside the assistant bubble
            with st.chat_message("assistant", avatar="🤖"):
                if used_search:
                    st.caption("🔍 Web Search Used")

                # ── Stream the response word-by-word (ChatGPT-style) ──────────
                message_placeholder = st.empty()
                full_response = ""
                for word in ai_response.split(' '):
                    full_response += word + ' '
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.04)
                # Final render — remove cursor
                message_placeholder.markdown(full_response.strip())

        except Exception as e:
            full_trace = traceback.format_exc()
            print(f"[ERROR] {full_trace}")
            ai_response = f"❌ Error: {str(e)}"
            with st.chat_message("assistant", avatar="🤖"):
                st.error(ai_response)

    # Add AI reply to history
    get_active_session()["history"].append({
        "role": "ai",
        "content": ai_response,
        "used_search": used_search,
    })
    st.session_state.total_messages += 1

    if used_search:
        st.session_state.total_searches += 1

    # ── Auto-rename chat after the very first exchange ────────────────────────
    current_history = get_active_session()["history"]
    if len(current_history) == 2 and get_active_session()["name"] == "New Chat":
        auto_rename_chat(
            session_id=st.session_state.active_session,
            first_user_msg=current_history[0]["content"],
            first_ai_msg=current_history[1]["content"],
            provider=st.session_state.provider,
        )

    # ── Update user profile every 4 exchanges (8 messages) ───────────────────
    if st.session_state.total_messages % 8 == 0 and st.session_state.total_messages > 0:
        try:
            from config import get_llm
            profile_llm = get_llm(provider=st.session_state.provider)
            recent_history = get_active_session()["history"][-12:]
            recent_text = ""
            for m in recent_history:
                role = "User" if m["role"] == "user" else "Assistant"
                recent_text += f"{role}: {m['content']}\n"
            existing_profile = load_profile()
            new_profile = update_profile(recent_text, existing_profile, profile_llm)
            save_profile(new_profile)
            print(f"[Profile] Updated after {st.session_state.total_messages} messages.")
        except Exception as e:
            print(f"[Profile] Update failed: {e}")

    save_sessions_to_disk()
