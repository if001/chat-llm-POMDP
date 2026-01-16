# app/ports/search.py
from __future__ import annotations
from typing import Any, Optional, Protocol
from dataclasses import dataclass, field


@dataclass
class WebContent:
    title: str
    content: str
    url: Optional[str]


class WebSearchPort(Protocol):
    async def search(
        self, query: str, *, limit: int = 5, **kwargs: Any
    ) -> list[WebContent]: ...
