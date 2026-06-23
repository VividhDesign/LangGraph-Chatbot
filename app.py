"""
app.py — Streamlit frontend for the LangGraph Chatbot.
This is the ONLY file you need to run to start the chatbot.
It provides a beautiful chat interface that:
  - Shows a sidebar with settings and info
  - Displays the full conversation history
  - Sends user messages to the LangGraph graph
  - Shows an indicator when web search is used
  - Lets you clear the chat history
"""

import streamlit as st
import uuid
import json
import os
from datetime import datetime
from database.profile_manager import load_profile, save_profile, update_profile

#Page configuration
#must be the first stremlit commmand

st.set_page_config(
    page_title = "LangGraph Chatbot",
    page_icon = "🤖", 
    layout = "wide",
    initial_sidebar_state = "expanded",
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
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e8e8f0;
    }
    /* Main header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #9ca3af;
        font-size: 1rem;
        font-weight: 300;
    }
    /* Chat message bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #6d28d9, #4f46e5);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(109, 40, 217, 0.3);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .ai-bubble {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #e8e8f0;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 85%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        font-size: 0.95rem;
        line-height: 1.7;
        backdrop-filter: blur(10px);
    }
    /* Search indicator badge */
    .search-badge {
        display: inline-block;
        background: linear-gradient(90deg, #059669, #10b981);
        color: white;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        margin-bottom: 0.5rem;
        letter-spacing: 0.03em;
    }
    /* Sender labels */
    .sender-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
        color: #9ca3af;
    }
    .sender-label.user { text-align: right; color: #c4b5fd; }
    .sender-label.ai   { text-align: left;  color: #60a5fa; }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.9) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .sidebar-section {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .sidebar-section h4 {
        color: #a78bfa;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .stat-item {
        display: flex;
        justify-content: space-between;
        color: #9ca3af;
        font-size: 0.83rem;
        padding: 0.2rem 0;
    }
    .stat-value {
        color: #e8e8f0;
        font-weight: 500;
    }
    /* Input area */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #e8e8f0 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.3) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #6b7280 !important;
    }
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
    }
    /* Clear button — red variant */
    .clear-btn > button {
        background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    }
    /* Divider */
    hr { border-color: rgba(255,255,255,0.08) !important; }
    /* Hide default Streamlit elements */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
    /* Scrollable chat area */
    .chat-container {
        max-height: 62vh;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# - persistent sessions: save and load from a JSON file 
SESSIONS_FILE = "sessions_data.json"

def save_sessions_to_disk():
    """
    Saves all sessions to a JSON file on a disk.
    This runs after every messagee so nothing is ever lost.
    """
    try:
        with open(SESSIONS_FILE, "w") as f:
            json.dump(st.session_state.sessions, f, indent = 2)
    except Exception as e:
        print(f"[Sessions] Could not save: {e}")

def load_sessions_from_disk():
    """
    loads all sessions from the JSON file on a disk. Called once when the app starts
    Returns an empty dict if no file exists yet.
    """
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


#helper : create a new empty chat session

def make_new_session():
    """Creates a new chat sesion and returns its ID."""
    new_id = str(uuid.uuid4())[:8]
    timestamp  = datetime.now().strftime("%H:%M")
    session_number = len(st.session_state.get("sessions", {}))+ 1
    st.session_state.sessions[new_id] = {
        "name" : f"Chat {session_number}", 
        "history" : [], 
        "created_at" : timestamp, 
    }
    return new_id

#helper : get the currently active session s data

def get_active_session():
    """Returns the dict for the currently open session."""
    return st.session_state.sessions[st.session_state.active_session]

# "session" state initialization
# sessions - all conversations ever started 
# load from disk on first run so sessions survive app restarts
if "sessions" not in st.session_state:
    loaded = load_sessions_from_disk()
    st.session_state.sessions = loaded if loaded else {}


# "active_session" = which conversation is open right now
if "active_session" not in st.session_state:
    first_id = make_new_session()
    st.session_state.active_session = first_id

if "total_searches" not in st.session_state:
    st.session_state.total_searches = 0

if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0

if "graph_loaded" not in st.session_state:
    st.session_state.graph_loaded = False

if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

#enter - key submit flag (keep this - needed for enter to work)
if "submitted" not in st.session_state:
    st.session_state.submitted = False






# ── Load LangGraph (only once)
# We use st.cache_resource so the graph is built only once, not on every re-run.
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
        profile_llm = get_llm()
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
        st.session_state.input_counter += 1
        st.rerun()

    st.markdown("---")

    # ── Conversation History List ─────────────────────────────────────────────
    st.markdown("### 💬 Your Chats")

    # Show all sessions, newest first
    all_sessions = list(st.session_state.sessions.items())
    all_sessions.reverse()

    for sid, sdata in all_sessions:
        is_active = (sid == st.session_state.active_session)
        msg_count = len(sdata["history"])
        display_label = f"{'🟣' if is_active else '⚫'} {sdata['name']} ({msg_count} msgs)"

        # Each chat row = chat button + delete button side by side
        col_chat, col_del = st.columns([5, 1])
        with col_chat:
            if st.button(display_label, key=f"session_btn_{sid}", use_container_width=True):
                # FIRST: save knowledge from current session before switching
                update_profile_from_session(get_active_session()["history"])
                st.session_state.active_session = sid
                h = sdata["history"]
                st.session_state.total_messages = len(h)
                st.session_state.total_searches = sum(
                    1 for m in h if m.get("used_search") and m["role"] == "ai"
                )
                st.session_state.input_counter += 1
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_btn_{sid}", help="Delete this chat"):
                del st.session_state.sessions[sid]
                # If we deleted the active chat, switch to another one
                if st.session_state.active_session == sid:
                    if st.session_state.sessions:
                        # Switch to the most recent remaining session
                        st.session_state.active_session = list(st.session_state.sessions.keys())[-1]
                    else:
                        # No sessions left — create a fresh one
                        new_id = make_new_session()
                        st.session_state.active_session = new_id
                    st.session_state.total_messages = 0
                    st.session_state.total_searches = 0
                save_sessions_to_disk()
                st.session_state.input_counter += 1
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
        st.session_state.input_counter += 1
        st.success("Chat cleared!")
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="color: #6b7280; font-size: 0.8rem; text-align: center;">
        Built with LangGraph + Streamlit<br>
        Gemini 3.1 Flash-Lite
    </div>
    """, unsafe_allow_html=True)

# ── MAIN PAGE ─────────────────────────────────────────────────────────────────
# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 LangGraph Chatbot</h1>
    <p>An intelligent multi-turn chatbot with web search · built with LangGraph & Gemini</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")
# ── Chat History Display ───────────────────────────────────────────────────────
if not get_active_session()["history"]:
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
else:
    # Display each message
    for msg in get_active_session()["history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="sender-label user">👤 YOU</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="sender-label ai">🤖 ASSISTANT</div>', unsafe_allow_html=True)
            # Show search badge if web search was used
            if msg.get("used_search"):
                st.markdown('<div class="search-badge">🔍 Web Search Used</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
# ── Input Area ────────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])

def on_enter():
    """Called by Streamlit when the user presses Enter inside the text box."""
    st.session_state.submitted = True

with col_input:
    user_input = st.text_input(
        label="Message",
        placeholder="Type your message and press Enter or click Send ➤",
        label_visibility="collapsed",
        key=f"user_input_{st.session_state.input_counter}",
        on_change=on_enter,   # fires when user presses Enter
    )
with col_btn:
    send_clicked = st.button("Send ➤", use_container_width=True)
# ── Handle Send ───────────────────────────────────────────────────────────────
# Trigger when user clicks Send button OR presses Enter in the text box
if (send_clicked or st.session_state.submitted) and user_input.strip():
    st.session_state.submitted = False  # reset flag immediately to prevent loop
    user_message = user_input.strip()

    # Add user message to the ACTIVE SESSION's history
    get_active_session()["history"].append({
        "role": "user",
        "content": user_message,
        "used_search": False,
    })
    st.session_state.total_messages += 1

    # Show a spinner while the AI thinks
    with st.spinner("🤔 Thinking..."):
        if run_chat_fn is None:
            ai_response = "❌ Error: LangGraph could not be loaded. Check your .env file."
            used_search = False
        else:
            try:
                result = run_chat_fn(
                    user_message=user_message,
                    session_id=st.session_state.active_session,
                )
                ai_response = result["response"]
                used_search = result["used_search"]
            except Exception as e:
                ai_response = f"❌ Error: {str(e)}"
                used_search = False

    # Add AI reply to the ACTIVE SESSION's history
    get_active_session()["history"].append({
        "role": "ai",
        "content": ai_response,
        "used_search": used_search,
    })
    st.session_state.total_messages += 1

    if used_search:
        st.session_state.total_searches += 1

    # ── Update user profile every 4 exchanges (8 messages) ───────────────────
    # NOT inside 'if used_search' — runs after EVERY message regardless of search
    # total_messages counts both user + AI messages, so 8 = 4 full exchanges
    if st.session_state.total_messages % 8 == 0 and st.session_state.total_messages > 0:
        try:
            from config import get_llm
            profile_llm = get_llm()
            # Build text of the recent messages for profile extraction
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

    # Save sessions to disk — ALWAYS, not just when search is used
    save_sessions_to_disk()

    # Clear the input box
    st.session_state.input_counter += 1
    st.rerun()
