# app/adapters/firecrawl_search.py
from __future__ import annotations
from typing import Any
from firecrawl import (
    Firecrawl,
)  # docs: Firecrawl Python SDK :contentReference[oaicite:1]{index=1}
from app.ports.search import WebSearchPort


class FirecrawlSearchAdapter(WebSearchPort):
    def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
        # base_url が必要なself-host構成は環境により異なるため、ここでは保持だけ
        self._client = Firecrawl(api_key=api_key)
        self._base_url = base_url

    async def search(
        self, query: str, *, limit: int = 5, **kwargs: Any
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Firecrawl call is out of scope for this skeleton")
