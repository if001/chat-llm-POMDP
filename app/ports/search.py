# app/ports/search.py
from __future__ import annotations
from typing import Any, Protocol


class WebSearchPort(Protocol):
    async def search(
        self, query: str, *, limit: int = 5, **kwargs: Any
    ) -> list[dict[str, Any]]: ...
