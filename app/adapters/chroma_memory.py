# app/adapters/chroma_memory.py
from __future__ import annotations
from typing import Any
from datetime import datetime, timedelta

from app.ports.memory import MemoryPort

import chromadb
from chromadb.api.types import Embedding
from chromadb.utils.embedding_functions import EmbeddingFunction


def unix_now() -> int:
    return int(datetime.utcnow().timestamp())


def days_ago(days: int) -> int:
    return int((datetime.utcnow() - timedelta(days=days)).timestamp())


def weeks_ago(weeks: int) -> int:
    return int((datetime.utcnow() - timedelta(weeks=weeks)).timestamp())


def months_ago(months: int) -> int:
    # 厳密な暦月ではなく「30日×months」とする（Chroma用途では十分）
    return int((datetime.utcnow() - timedelta(days=30 * months)).timestamp())


def years_ago(years: int) -> int:
    # 365日×years（うるう年は無視）
    return int((datetime.utcnow() - timedelta(days=365 * years)).timestamp())


class _ChromaEF(EmbeddingFunction):
    def __init__(self, embed_query):
        self._embed_query = embed_query

    def __call__(self, input: list[str]) -> list[Embedding]:
        # chromadb embedding function expects list[str] -> list[vectors]
        return [self._embed_query(x) for x in input]


class ChromaMemoryAdapter(MemoryPort):
    def __init__(
        self, *, persist_dir: str, collection_name: str, embed_query, **kwargs: Any
    ) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=_ChromaEF(embed_query),
            metadata={"hnsw:space": "cosine"},
        )

    async def recall(self, query: str, *, k: int = 5, tw="all") -> list[dict[str, Any]]:
        t = None
        if tw == "7d":
            t = weeks_ago(1)
        if tw == "30d":
            t = months_ago(1)
        if tw == "180d":
            t = months_ago(3)
        where = None
        if t is not None:
            where = {"created_at": {"$gte": t}}
        res = self._col.query(query_texts=[text], n_results=k, where=where)
        docs = (res.get("documents") or [[]])[0]
        return [d for d in docs if d]

    async def store(self, items: list[dict[str, Any]], metadata: dict) -> None:
        for item in items:
            text = item["text"]
            metadata = item["metadata"]
            _id = metadata.get("id") or metadata.get("turn_id") or str(hash(text))
            self._col.add(ids=[str(_id)], documents=[text], metadatas=[metadata])
