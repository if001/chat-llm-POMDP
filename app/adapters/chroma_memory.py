# app/adapters/chroma_memory.py
from __future__ import annotations

import asyncio
from typing import Any, Optional
from datetime import datetime, timedelta
import json
import re
import uuid

from app.graph.nodes.prompt_utils import format_wm_messages
from app.ports.memory import MemoryPort
from app.ports.llm import Embedder, LLMPort

import chromadb
from chromadb.api.types import Embedding
from chromadb.utils.embedding_functions import EmbeddingFunction


def unix_now() -> int:
    return int(datetime.now().timestamp())


def days_ago(days: int) -> int:
    return int((datetime.now() - timedelta(days=days)).timestamp())


def weeks_ago(weeks: int) -> int:
    return int((datetime.now() - timedelta(weeks=weeks)).timestamp())


def months_ago(months: int) -> int:
    # 厳密な暦月ではなく「30日×months」とする（Chroma用途では十分）
    return int((datetime.now() - timedelta(days=30 * months)).timestamp())


def years_ago(years: int) -> int:
    # 365日×years（うるう年は無視）
    return int((datetime.now() - timedelta(days=365 * years)).timestamp())


class _ChromaEF(EmbeddingFunction):
    def __init__(self, embed_query):
        self._embed_query = embed_query

    def __call__(self, input: list[str]) -> list[Embedding]:
        # chromadb embedding function expects list[str] -> list[vectors]
        return [self._embed_query(x) for x in input]


def _safe_tags(tags: Any) -> list[str]:
    """
    tagsはChromaのmetadataで安全に扱えるように正規化する。
    Chromaのmetadataは基本的にプリミティブ型を期待するため、
    最終的には JSON文字列として格納する。
    """
    if tags is None:
        return []
    if isinstance(tags, str):
        # "a,b,c" もしくは "['a','b']" っぽいのを許容
        s = tags.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s)
                if isinstance(v, list):
                    return [str(x) for x in v]
            except Exception:
                pass
        # カンマ区切り
        return [t.strip() for t in s.split(",") if t.strip()]
    if isinstance(tags, list):
        return [str(x) for x in tags if str(x).strip()]
    return [str(tags)]


def _normalize_episode_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """
    要件: metadataは type, topic, created_ts, importance, tags のみ。
    tags は JSON 文字列として格納する（Chroma metadataの互換性重視）。
    """
    t = str(meta.get("type") or "episodic")
    topic = str(meta.get("topic") or "general")
    created_ts = int(meta.get("created_ts") or unix_now())

    importance_raw = meta.get("importance", 0)
    try:
        importance = float(importance_raw)
    except Exception:
        importance = 0.5
    # clamp
    if importance < 0.0:
        importance = 0.0
    if importance > 1.0:
        importance = 1.0

    tags_list = _safe_tags(meta.get("tags"))
    tags_json = json.dumps(tags_list, ensure_ascii=False)

    return {
        "type": t,
        "topic": topic,
        "created_ts": created_ts,
        "importance": importance,
        "tags": tags_json,
    }


def _extract_json_object(s: str) -> Optional[dict[str, Any]]:
    """
    文字列内にJSONが混ざるケースを想定して、最初の{から最後の}までを抜く。
    ネストは「最外」の範囲で抜く（厳密パースはjson.loadsに任せる）。
    """
    if not s:
        return None
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    cand = s[start : end + 1]
    try:
        v = json.loads(cand)
        if isinstance(v, dict):
            return v
    except Exception:
        return None
    return None


class ChromaMemoryAdapter(MemoryPort):
    """
    エピソード記憶（出来事単位）をChromaに保存するアダプタ。
    """

    def __init__(
        self,
        *,
        persist_dir: str,
        collection_name: str,
        embeder: Embedder,
        llm: LLMPort,
        small_llm: LLMPort,
        **kwargs: Any,
    ) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=_ChromaEF(embeder.embed_query),
            metadata={"hnsw:space": "cosine"},
        )
        self._llm = llm
        self._small_llm = small_llm

    async def recall(
        self, query: str, *, k: int = 5, tw: str = "all"
    ) -> list[dict[str, Any]]:
        """
        query: 検索クエリ文字列
        tw: time window: "all" | "7d" | "30d" | "180d"
        """
        t: Optional[int] = None
        if tw == "7d":
            t = weeks_ago(1)
        elif tw == "30d":
            t = months_ago(1)
        elif tw == "180d":
            t = months_ago(6)

        where = None
        if t is not None:
            # 要件に合わせて created_ts を使う
            where = {"created_ts": {"$gte": t}}

        res = self._col.query(query_texts=[query], n_results=k, where=where)

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        out: list[dict[str, Any]] = []
        for d, m in zip(docs, metas, strict=False):
            if not d:
                continue
            out.append({"text": d, "metadata": m or {}})
        return out

    async def store(
        self, items: list[dict[str, Any]], metadata: dict | None = None
    ) -> None:
        """
        items: [{"text": "...", "metadata": {...}}] を想定。
        - item.metadata が無ければ、引数metadataを適用してよい（後方互換のため）
        """
        base_meta = metadata or {}
        for item in items:
            text = item["text"]
            item_meta = dict(base_meta)
            item_meta.update(item.get("metadata") or {})
            norm_meta = _normalize_episode_metadata(item_meta)

            # idはChroma側でユニークなら何でもよいのでuuidを推奨
            _id = item.get("id") or norm_meta.get("id") or str(uuid.uuid4())

            self._col.add(
                ids=[str(_id)],
                documents=[text],
                metadatas=[norm_meta],
            )

    async def _build_episode_from_history(
        self,
        wm_messages: list[dict[str, str]],
        *,
        default_topic: str = "general",
        default_importance: float = 0.5,
        max_summary_chars: int = 1200,
        tags_max: int = 8,
    ) -> dict[str, Any]:
        """
        historyから、エピソード本文（要約）とmetadataを生成する。

        戻り値:
          {
            "text": "<episode summary>",
            "metadata": {type, topic, created_ts, importance, tags}
          }
        """

        async def extract():
            system_prompt = (
                "You are a classifier for episodic memory metadata.\n"
                "Return ONLY valid JSON.\n"
                "Rules:\n"
                "- topic: short, snake_case preferred.\n"
                "- importance: higher if decisions, TODOs, constraints, or repeated important preferences appear.\n"
                f"- tags: up to {tags_max} short tokens.\n"
                "Schema:\n"
                '{ "topic": "string", "importance": 0.0-1.0, "tags": ["tag1", ...] }\n'
            )

            prompt = f"Extract metadata\nhistory:\n{format_wm_messages(wm_messages, limit=8)}\n"

            meta_raw = await self._small_llm.ainvoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )

            meta_text = (
                getattr(meta_raw, "content", None) if meta_raw is not None else None
            )
            if meta_text is None and isinstance(meta_raw, str):
                meta_text = meta_raw

            meta_obj = _extract_json_object(meta_text or "") or {}
            topic = str(meta_obj.get("topic") or default_topic)
            importance = meta_obj.get("importance", default_importance)
            tags = meta_obj.get("tags", [])

            # topic整形（軽く）
            topic = topic.strip()
            topic = re.sub(r"\s+", "_", topic)
            topic = re.sub(r"[^a-zA-Z0-9_\-]", "", topic) or default_topic
            return tags, topic, importance

        async def summary():
            system_prompt = (
                "You write a concise episodic memory summary for later retrieval.\n"
                "Requirements:\n"
                "- Include decisions, constraints, TODOs, and key context.\n"
                "- Be faithful to the conversation (no inventions).\n"
                f"- Max {max_summary_chars} characters.\n"
                "- Write in Japanese.\n"
            )

            prompt = f"Summarize this conversation as one episodic memory entry:\n{format_wm_messages(wm_messages, limit=8)}"
            sum_raw = await self._small_llm.ainvoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            summary = getattr(sum_raw, "content", None) if sum_raw is not None else None
            if summary is None and isinstance(sum_raw, str):
                summary = sum_raw
            summary = (summary or "").strip()
            if len(summary) > max_summary_chars:
                summary = summary[:max_summary_chars].rstrip()
            return summary

        extracted, summary_text = await asyncio.gather(extract(), summary())
        tags, topic, importance = extracted

        created_ts = unix_now()
        metadata = _normalize_episode_metadata(
            {
                "type": "episodic",
                "topic": topic,
                "created_ts": created_ts,
                "importance": importance,
                "tags": tags,
            }
        )

        return {"text": summary_text, "metadata": metadata}

    async def store_episode_from_history(
        self,
        history: list[dict[str, str]],
        *,
        default_topic: str = "general",
        default_importance: float = 0.5,
    ) -> dict[str, Any]:
        episode = await self._build_episode_from_history(
            history,
            default_topic=default_topic,
            default_importance=default_importance,
        )
        await self.store([episode], metadata=None)
        return episode
