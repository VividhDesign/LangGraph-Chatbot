# 🤖 LangGraph Chatbot

An intelligent, multi-turn chatbot built with **LangGraph**, supporting **Google Gemini** and **Groq** (Llama, Mixtral, Gemma) — with persistent memory, web search, and multi-session management.

## ✨ Features

- 🧠 **Sliding-window summary memory** — Remembers your full conversation without hitting token limits
- 💬 **Multiple chat sessions** — Switch between conversations like ChatGPT
- 🔍 **Web search** — Automatically searches the web for real-time information
- 🗂️ **Persistent sessions** — Chats saved to disk, survive app restarts
- 👤 **Cross-session user profile** — Bot learns who you are over time
- 🏷️ **AI-powered chat naming** — Chats are auto-named based on your first message
- 🔑 **API key persistence** — Keys saved in your browser's localStorage (never sent to any server)
- 🌐 **Multi-provider support** — Switch between Google Gemini and Groq in the sidebar
- 📋 **Dynamic model list** — All models fetched live from the provider's API

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM Providers | Google Gemini, Groq (Llama, Mixtral, Gemma) |
| Agent Framework | LangGraph |
| Frontend | Streamlit |
| Memory Checkpointing | LangGraph MemorySaver |
| Web Search | DuckDuckGo Search |

## 🚀 Run Locally

1. Clone the repo:
```bash
git clone https://github.com/VividhDesign/LangGraph-Chatbot.git
cd LangGraph-Chatbot
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add GOOGLE_API_KEY and/or GROQ_API_KEY
```
> If you skip this step, the app will ask for your API key on first launch and save it in your browser.

5. Run the app:
```bash
streamlit run app.py
```

## 🔑 Getting API Keys

- **Google Gemini API Key:** [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (free)
- **Groq API Key:** [console.groq.com/keys](https://console.groq.com/keys) (free)
- **LangSmith (optional):** [smith.langchain.com](https://smith.langchain.com) (for debugging traces)

## 🏗️ Project Structure

```
LangGraph_Chatbot/
├── app.py                  # Streamlit frontend
├── config.py               # LLM configuration (Gemini + Groq)
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
├── .streamlit/
│   └── config.toml         # Streamlit theme (ChatGPT-style dark mode)
├── agents/
│   └── chatbot_agent.py    # Main LangGraph chatbot node
├── graph/
│   ├── chatbot_graph.py    # LangGraph graph assembly
│   └── state.py            # Shared state definition
├── tools/
│   └── search_tool.py      # DuckDuckGo web search tool
└── database/
    └── profile_manager.py  # Cross-session user profile (profile.json)
```

## 💡 How Memory Works

```
[User Profile]     ← facts about you, across ALL sessions (profile.json)
[Session Summary]  ← compressed older messages (this session)
[Last 10 Messages] ← recent conversation in full
[Your Question]    ← current message
```

All four layers are merged into one system message, keeping token usage constant regardless of conversation length.
