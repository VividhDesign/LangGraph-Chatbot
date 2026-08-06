"""
config.py loads environment variables and gives the 
LLM (Gemini or Groq)
Think of this as the "settings file"
for the whole project.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# load the environment variables from the .env file
load_dotenv()

# Available fallback models for the model selector
AVAILABLE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
]
GROQ_AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

def get_available_models() -> list:
    """Returns the list of fallback Gemini models."""
    return AVAILABLE_MODELS

def get_available_groq_models() -> list:
    """Returns the list of fallback Groq models."""
    return GROQ_AVAILABLE_MODELS

def get_llm(temperature: float = 0.7, model: str = None, api_key: str = None, provider: str = "gemini"):
    """
    Returns a Gemini or Groq LLM instance.

    Args:
        temperature: Sampling temperature.
        model: Which model to use.
        api_key: Optional API key override (e.g. from browser localStorage).
        provider: "gemini" or "groq".
    """
    if provider == "groq":
        selected_model = model or DEFAULT_GROQ_MODEL
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                pass
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found! Add it to .env, Streamlit secrets, or enter it in the app.")
            
        return ChatGroq(
            model_name=selected_model,
            temperature=temperature,
            api_key=api_key,
        )
    else:
        selected_model = model or DEFAULT_MODEL
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GOOGLE_API_KEY")
            except Exception:
                pass
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found! Add it to .env, Streamlit secrets, or enter it in the app.")

        return ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=api_key,
        )

#project name shown in langsmith traces
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "LangGraph_Chatbot")

#Maximum number of past messages to keep in memory
MAX_HISTORY = 10
