"""
Semantic memory using NVIDIA nv-embed-v1 embeddings.
Stores agent outputs as vectors — agents can retrieve relevant past context
by semantic similarity instead of exact key lookup.

Usage:
    store = EmbeddingStore(api_key)
    await store.add("t1", "Researcher found that HR market is $12B...")
    results = await store.search("what did we learn about competitors?", top_k=3)
"""

import json
import math
import httpx
from pathlib import Path
from datetime import datetime


NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v1"  # multilingual (26 langs incl. Russian)
EMBED_FALLBACK = "nvidia/nv-embed-v1"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingStore:
    """
    Simple in-process vector store.
    No external DB needed — persists to JSON on disk.
    For production replace with ChromaDB or Qdrant.
    """

    def __init__(self, api_key: str, persist_path: str = "memory/embeddings.json", session_id: str = None):
        self.api_key = api_key
        self.session_id = session_id
        self._path = Path(persist_path)
        self._path.parent.mkdir(exist_ok=True)
        # Each entry: {"key": str, "text": str, "vector": list[float], "ts": str, "session_id": str}
        self._entries: list[dict] = []
        if self._path.exists():
            self._entries = json.loads(self._path.read_text(encoding="utf-8"))

    async def _embed(self, text: str) -> list[float]:
        """Get embedding vector from NVIDIA API, with fallback model."""
        async with httpx.AsyncClient(timeout=30) as client:
            for model in (EMBED_MODEL, EMBED_FALLBACK):
                resp = await client.post(
                    f"{NVIDIA_API_BASE}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": model,
                        "input": text[:8000],
                        "input_type": "query",
                        "encoding_format": "float",
                    },
                )
                if resp.status_code == 200:
                    return resp.json()["data"][0]["embedding"]
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    async def add(self, key: str, text: str):
        """Embed and store a text entry, scoped to current session_id."""
        vector = await self._embed(text)
        entry = {
            "key": key,
            "text": text,
            "vector": vector,
            "ts": datetime.now().isoformat(),
            "session_id": self.session_id,
        }
        # Replace if key+session exists
        self._entries = [
            e for e in self._entries
            if not (e["key"] == key and e.get("session_id") == self.session_id)
        ]
        self._entries.append(entry)
        self._persist()
        print(f"  [EmbeddingStore] stored '{key}' ({len(text)} chars)")

    async def search(self, query: str, top_k: int = 3, session_only: bool = True) -> list[dict]:
        """Return top_k most relevant entries. If session_only, restrict to current session."""
        entries = self._entries
        if session_only and self.session_id:
            entries = [e for e in entries if e.get("session_id") == self.session_id]
        if not entries:
            return []
        q_vec = await self._embed(query)
        scored = [
            {"key": e["key"], "text": e["text"], "score": _cosine_similarity(q_vec, e["vector"])}
            for e in entries
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    async def search_text(self, query: str, top_k: int = 3) -> str:
        """Convenience: return search results as formatted string for prompt injection."""
        results = await self.search(query, top_k)
        if not results:
            return ""
        parts = [f"[{r['key']} | score: {r['score']:.2f}]\n{r['text']}" for r in results]
        return "\n\n---\n\n".join(parts)

    def get_all_keys(self) -> list[str]:
        return [e["key"] for e in self._entries]

    def clear(self, all_sessions: bool = False):
        """Clear current session entries (or all if all_sessions=True)."""
        if all_sessions:
            self._entries = []
        else:
            self._entries = [e for e in self._entries if e.get("session_id") != self.session_id]
        self._persist()

    def _persist(self):
        # Don't persist vectors to save space — only text + key + ts
        # Re-embed on next load if needed (trade-off: simplicity vs speed)
        saveable = [
            {"key": e["key"], "text": e["text"], "vector": e["vector"], "ts": e["ts"]}
            for e in self._entries
        ]
        self._path.write_text(json.dumps(saveable, ensure_ascii=False, indent=2), encoding="utf-8")
