"""
config.py loads environment variables and gives the 
LLM (Gemini)
Think of this as the "settings file"
for the whole project.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# load the environment variables from the .env file
load_dotenv()


def get_llm(temperature: float = 0.7):
    """
    Returns a Gemini LLM instance.

    Note: Gemini 3.x models (like gemini-3.5-flash) do NOT recommend
    setting temperature/top_p/top_k — the model is optimized for its
    own defaults. So we skip temperature for 3.x models.
    """
    # Try Streamlit Cloud secrets first (for public deployment)
    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        pass

    # Fall back to .env file (for local development)
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found! Add it to .env or Streamlit secrets.")

    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        # temperature intentionally NOT passed — Gemini 3.x docs say
        # "strongly recommend not changing the default values"
    )
#project name shown in langsmith traces
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "LangGraph_Chatbot")

#Maximum number of past messages to keep in memory
MAX_HISTORY = 10

