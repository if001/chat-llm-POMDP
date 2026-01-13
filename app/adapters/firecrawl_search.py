# app/adapters/firecrawl_search.py
from __future__ import annotations
from typing import Any
from firecrawl import (
    Firecrawl,
)
from firecrawl.client import Document
from firecrawl.v2.types import (
    Location,
)
from app.ports.search import WebContent, WebSearchPort


class FirecrawlSearchAdapter(WebSearchPort):
    def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
        self._app = Firecrawl(api_key=api_key, api_url=base_url)
        self.fetch_full_page = True

    def _inner(self, query: str, limit=2) -> list[WebContent]:
        search_response = self._app.search(
            query=query, limit=limit, location="ja", timeout=30
        )
        if search_response.web is None:
            return []
        docs: list[WebContent] = []
        for v in search_response.web:
            if isinstance(v, Document):
                r = WebContent(
                    title="",
                    url="",
                    content=v.markdown or "",
                )
                docs.append(r)
                continue
            url = v.url
            if url is None:
                continue
            # content = v.description
            raw_content = v.description
            if self.fetch_full_page:
                try:
                    scrape_result = self._app.scrape(
                        url,
                        formats=["markdown"],
                        location=Location(country="jp"),
                        only_main_content=True,
                        block_ads=True,
                        wait_for=30,
                        timeout=30,
                    )
                    raw_content = scrape_result.markdown
                except Exception as e:
                    print(f"query: {query}")
                    print("scrape error", e)
                    continue
            r = WebContent(
                title=v.title or "",
                url=url,
                content=raw_content or "",
            )
            docs.append(r)
        return docs

    async def search(
        self, query: str, *, limit: int = 5, **kwargs: Any
    ) -> list[WebContent]:
        try:
            return self._inner(query, limit)
        except Exception:
            return []
