"""
tools/search_tool - web search using DuckDuckGo(free, no API key needed)

This gives our chatbot the ability to search the internet when it needs
fresh information that its training data does nt have.
"""

from duckduckgo_search import DDGS

def web_search(query:str, max_results:int = 3) -> list[str]:
    """
    searches DuckDuckGo and returns a result snippets.
    Args:
        query : The search query string (e.g. "latest Python version")
        max_results : How many results to return (default : 3)
    Returns:
        A list of strings , each being a search result snippet.
        Returns an empty list if the search fails.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results):
                title = r.get("title", "")
                body = r.get("body", "")
                url = r.get("href", "")

                #format each results as a readable strings
                results.append(f"**{title}**\n{body}\nSource: {url}")
                return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []