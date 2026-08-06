"""
tools/search_tool.py - Web search using DuckDuckGo (free, no API key needed).

Gives the chatbot the ability to search the internet for fresh information
that its training data may not have.
"""

from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 3) -> list:
    """
    Searches DuckDuckGo and returns result snippets.

    Args:
        query: The search query string (e.g. "latest Python version")
        max_results: How many results to return (default: 3)

    Returns:
        A list of formatted result strings, or an empty list if search fails.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                body = r.get("body", "")
                url = r.get("href", "")
                results.append(f"**{title}**\n{body}\nSource: {url}")
        return results
    except Exception as e:
        print(f"[Search] Failed: {e}")
        return []