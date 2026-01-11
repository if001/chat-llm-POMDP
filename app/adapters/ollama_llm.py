# app/adapters/ollama_llm.py
from __future__ import annotations
from typing import Any
from collections.abc import AsyncIterator

from langchain_ollama import ChatOllama, OllamaEmbeddings
from app.ports.llm import LLMPort, Embedder


class OllamaChatAdapter(LLMPort):
    def __init__(self, model: str, base_url: str, temperature: float = 0.2):
        self._llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)

    async def ainvoke(self, messages: list[dict[str, Any]]) -> Any:
        return await self._llm.ainvoke(messages)

    async def astream(self, messages: list[dict[str, Any]]) -> AsyncIterator[Any]:
        async for chunk in self._llm.astream(messages):
            yield chunk


class OllamaEmbedder(Embedder):
    def __init__(self, model: str, base_url: str):
        self._emb = OllamaEmbeddings(model=model, base_url=base_url)

    def embed_query(self, text: str) -> list[float]:
        return self._emb.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed_documents(texts)
