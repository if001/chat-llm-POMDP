# app/ports/llm.py
from __future__ import annotations
from typing import Any, Protocol
from collections.abc import AsyncIterator


class LLMPort(Protocol):
    def ainvoke(self, messages: list[dict[str, Any]]) -> Any: ...
    def astream(self, messages: list[dict[str, Any]]) -> AsyncIterator[Any]: ...


class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
