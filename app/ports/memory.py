# app/ports/memory.py
from __future__ import annotations
from typing import Any, Protocol


class MemoryPort(Protocol):
    async def recall(self, query: str, *, k: int = 5) -> list[dict[str, Any]]: ...
    async def store(self, items: list[dict[str, Any]]) -> None: ...
    async def store_episode_from_history(
        self, history: list[dict[str, str]]
    ) -> dict[str, Any]: ...
