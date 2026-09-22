"""
Simple in-memory context store with optional disk persistence.
Agents read from and write to this store via the orchestrator.
"""

import json
from pathlib import Path
from datetime import datetime


class ContextStore:
    def __init__(self, persist_path: str = "memory/context.json"):
        self._store: dict[str, str] = {}
        self._path = Path(persist_path)
        self._path.parent.mkdir(exist_ok=True)
        if self._path.exists():
            self._store = json.loads(self._path.read_text(encoding="utf-8"))

    def store(self, key: str, value: str):
        self._store[key] = value
        self._persist()

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def get_many(self, keys: list[str]) -> dict[str, str]:
        return {k: self._store[k] for k in keys if k in self._store}

    def all(self) -> dict[str, str]:
        return dict(self._store)

    def clear(self):
        self._store = {}
        self._persist()

    def _persist(self):
        self._path.write_text(json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8")
