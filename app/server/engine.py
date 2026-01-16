from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, List, Optional
from zoneinfo import ZoneInfo
from app.server.schema import ChatMessage, ToolCall, ToolDefinition, ThinkType


from app.config.settings import Settings
from app.core.deps import Deps
from app.adapters.ollama_llm import OllamaChatAdapter, OllamaEmbedder
from app.adapters.firecrawl_search import FirecrawlSearchAdapter
from app.adapters.chroma_memory import ChromaMemoryAdapter
from app.adapters.trace_json import JsonTraceAdapter
from app.adapters.clock import SystemClock
from app.models.state import initial_state
from app.graph.build_graph import build_graph


@dataclass
class EngineChunk:
    """
    /api/chat のストリーミング 1 チャンク分に相当する出力。
    content/thinking/tool_calls を部分的に埋められるようにする。
    """

    content_delta: str = ""
    thinking_delta: str = ""
    tool_calls_delta: Optional[List[ToolCall]] = None


@dataclass
class EngineResult:
    content: str
    thinking: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class BaseChatEngine:
    async def chat_once(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]],
        think: Optional[ThinkType],
    ) -> EngineResult:
        raise NotImplementedError

    async def chat_stream(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]],
        think: Optional[ThinkType],
    ) -> AsyncIterator[EngineChunk]:
        raise NotImplementedError


class MockChatEngine(BaseChatEngine):
    async def chat_once(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]],
        think: Optional[ThinkType],
    ) -> EngineResult:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        thinking = None
        if think:
            thinking = "thinking... (mock)\n"

        # tools を受け取ったら、デモとして tool_calls を返すことも可能（ここでは返さない）
        content = f"[mock:{model}] {last_user}"
        return EngineResult(content=content, thinking=thinking)


TOKYO = ZoneInfo("Asia/Tokyo")


class ChatEngine(BaseChatEngine):
    def __init__(self):
        s = Settings()
        emb = OllamaEmbedder(model=s.embed_model, base_url=s.ollama_base_url)
        trace = JsonTraceAdapter(trace_dir=s.trace_dir)
        llm = OllamaChatAdapter(base_url=s.ollama_base_url, model=s.llm_model)
        small_llm = OllamaChatAdapter(
            base_url=s.ollama_base_url, model=s.small_llm_model
        )
        deps = Deps(
            llm=llm,
            small_llm=small_llm,
            memory=ChromaMemoryAdapter(
                persist_dir=s.chroma_persist_dir,
                collection_name=s.chroma_collection,
                embeder=emb,
                llm=llm,
                small_llm=small_llm,
            ),
            web=FirecrawlSearchAdapter(
                api_key=s.firecrawl_api_key, base_url=s.firecrawl_base_url
            ),
            trace=trace,
            clock=SystemClock(),
        )
        d = trace.load()
        self.graph = build_graph(deps)
        self.state = initial_state()
        self.state["wm_messages"] = d.get("wm_messages", [])
        _u = d.get("user_model")
        if _u is not None:
            self.state["user_model"] = _u

    async def chat_once(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]],
        think: Optional[ThinkType],
    ) -> EngineResult:
        user_in = next((m.content for m in reversed(messages) if m.role == "user"), "")
        thinking = None

        self.state["user_input"] = user_in
        result = await self.graph.ainvoke(self.state)
        ans = result["response"]["final_text"]
        return EngineResult(content=ans, thinking=thinking)

    async def chat_stream(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]],
        think: Optional[ThinkType],
    ) -> AsyncIterator[EngineChunk]:
        user_in = next((m.content for m in reversed(messages) if m.role == "user"), "")
        self.state["user_input"] = user_in
        try:
            async for mode, chunk in self.graph.astream(
                self.state, stream_mode=["values", "custom"]
            ):
                if mode == "custom":
                    if chunk["type"] == "status":
                        pass
                    if chunk["type"] == "thinking":
                        yield EngineChunk(thinking_delta="[" + chunk["text"] + "] ")
                    if chunk["type"] == "token":
                        yield EngineChunk(content_delta=chunk["text"])
        except:
            return


def build_engine() -> BaseChatEngine:
    # mock
    # return MockChatEngine()
    return ChatEngine()
