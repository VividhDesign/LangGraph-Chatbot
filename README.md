# 🤖 LangGraph Chatbot

An intelligent, multi-turn chatbot built with **LangGraph** and **Google Gemini**, featuring persistent memory, web search, and multi-session management.

## ✨ Features

- 🧠 **Sliding-window summary memory** — Remembers your full conversation without hitting token limits
- 💬 **Multiple chat sessions** — Switch between conversations like ChatGPT
- 🔍 **Web search** — Automatically searches the web for real-time information
- 🗂️ **Persistent sessions** — Chats saved to disk, survive app restarts
- 👤 **Cross-session user profile** — Bot learns who you are over time
- 🗑️ **Delete chats** — Manually manage your conversation history

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini (via LangChain) |
| Agent Framework | LangGraph |
| Frontend | Streamlit |
| Memory Checkpointing | LangGraph SqliteSaver |
| Web Search | DuckDuckGo Search |

## 🚀 Run Locally

1. Clone the repo:
```bash
git clone https://github.com/yourusername/LangGraph-Chatbot.git
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

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

5. Run the app:
```bash
streamlit run app.py
```

## 🔑 Getting API Keys

- **Google AI API Key:** [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (free)
- **LangSmith (optional):** [smith.langchain.com](https://smith.langchain.com) (for debugging)

## 🏗️ Project Structure

```
LangGraph_Chatbot/
├── app.py                  # Streamlit frontend
├── config.py               # LLM configuration
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
├── agents/
│   └── chatbot_agent.py    # Main LangGraph chatbot node
├── graph/
│   ├── chatbot_graph.py    # LangGraph graph assembly
│   └── state.py            # Shared state definition
├── tools/
│   └── search_tool.py      # Web search tool
├── database/
│   └── profile_manager.py  # Cross-session user profile
└── memory/
    └── memory_manager.py   # Memory utilities
```

## 💡 How Memory Works

```
[User Profile]     ← facts about you, across ALL sessions
[Session Summary]  ← compressed older messages (this session)
[Last 10 Messages] ← recent conversation in full
[Your Question]    ← current message
```

All four layers are merged into one system message sent to Gemini, keeping token usage constant regardless of conversation length.
