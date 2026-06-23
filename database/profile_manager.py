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
    """

    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r") as f:
                data = json.load(f)
                return data.get("profile", "")
        except Exception:
            return ""
        return ""

def save_profile(profile_text: str):
    """
    Saves the user profile text to disk.
    """
    try:
        with open(PROFILE_FILE, "w") as f:
            json.dump({"profile": profile_text}, f, indent = 2)
            print(f"[Profile] Profile saved.")
    except Exception as e:
        print(f"[Profile] Could not save profile: {e}")

def update_profile(recent_conversation:str, existing_profile:str, llm) -> str:
    """
    Asks Gemini to update the user profile based on recent coversation.
    Args:
        recent_conversation : a string of recent messaes (human + ai)
        existing_profile : the current profile(empty string if none yet)
        llm : the Gemini LLM instance
    returns:
        Updated profile sting(under 150 words)
    """

    if existing_profile:
        prompt = f"""You maintain a user profile for a Personal AI assistant.
        Update the profile below based on new information from the recent conversation.
        Ony add genuinly new , useful facts. keep total profile under 150 words.
        Remove outdate info if needed. Focus on: name, interests, skills, ongoing projects.
        Current profile : {existing_profile}
        Recent conversation to extract info from : {recent_conversation}
        write the updated profile only (no extra text): """
    else:
        prompt = f"""You maintain a user profile for a personal AI assistant.
        Extract key facts about the user from  this conversation.
        Keep it under 100 words. Focus on : name, interests, skills, ongoing projects.
        Conversation : {recent_conversation}
        write the user profile only (no extra text): """

    response = llm.invoke([
        SystemMessage(content = "You extract and maintain concise user profiles for AI assistants."), 
        HumanMessage(content = prompt),
    ])
    return response.content