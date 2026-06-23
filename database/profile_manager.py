"""
database / profile _manager.py - manages the cross session user profile.

The user profile is a short paragraph of key facts extracted by Gemini from the coversation histoyr.
it s stored in profile.json and injected into every new session so the chatbot remembers the user.
think of it like chatgpt s "memory" feature.
"""
import json  
import os
from langchain_core.messages import SystemMessage, HumanMessage 

PROFILE_FILE = "profile.json"

def load_profile() -> str:
    """
    Loads the user profile from disk.
    Returns an empty string if no profile exists yet.
    Always returns a plain string, even if the file was corrupted.
    """
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r") as f:
                data = json.load(f)
                value = data.get("profile", "")
                # Defensive: handle case where value was saved as a list
                if isinstance(value, list):
                    return " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in value
                    )
                return str(value) if value else ""
        except Exception:
            return ""
    return ""

def save_profile(profile_text: str):
    """Saves the user profile text to disk."""
    try:
        with open(PROFILE_FILE, "w") as f:
            json.dump({"profile": str(profile_text)}, f, indent=2)
        print("[Profile] Profile saved.")
    except Exception as e:
        print(f"[Profile] Could not save profile: {e}")

def update_profile(recent_conversation: str, existing_profile: str, llm) -> str:
    """
    Asks Gemini to update the user profile based on recent conversation.
    Always returns a plain string.
    """
    if existing_profile:
        prompt = f"""You maintain a user profile for a Personal AI assistant.
Update the profile below based on new information from the recent conversation.
Only add genuinely new, useful facts. Keep total profile under 150 words.
Remove outdated info if needed. Focus on: name, interests, skills, ongoing projects.

Current profile: {existing_profile}

Recent conversation: {recent_conversation}

Write the updated profile only (no extra text):"""
    else:
        prompt = f"""You maintain a user profile for a personal AI assistant.
Extract key facts about the user from this conversation.
Keep it under 100 words. Focus on: name, interests, skills, ongoing projects.

Conversation: {recent_conversation}

Write the user profile only (no extra text):"""

    response = llm.invoke([
        SystemMessage(content="You extract and maintain concise user profiles for AI assistants."),
        HumanMessage(content=prompt),
    ])
    # Handle both string and list response formats
    content = response.content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)