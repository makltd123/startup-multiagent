"""
Web search tool using DuckDuckGo (no API key required).
Returns formatted search results as a string ready for LLM context injection.
"""

import asyncio
from duckduckgo_search import DDGS


def _search_sync(query: str, max_results: int) -> list:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


async def web_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return results as formatted text."""
    try:
        results = await asyncio.to_thread(_search_sync, query, max_results)
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    {r['href']}")
            lines.append(f"    {r['body']}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        # Search is best-effort — don't fail the pipeline
        print(f"  [WebSearch] failed: {e}")
        return ""
