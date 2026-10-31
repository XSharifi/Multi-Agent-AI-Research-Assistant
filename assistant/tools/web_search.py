"""Web tools for the Research Agent: search (DuckDuckGo, no key) and page reader."""
import re

import requests
from bs4 import BeautifulSoup


def web_search(query: str, max_results: int = 5) -> str:
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS
    try:
        results = list(DDGS().text(query.strip(), max_results=max_results))
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
        for i, r in enumerate(results, 1)
    )


def read_url(url: str, max_chars: int = 6000) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return "Invalid URL: must start with http:// or https://"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (research-assistant)"}, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return f"Could not fetch page: {e}"
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    return text[:max_chars] + (" ...[truncated]" if len(text) > max_chars else "")
