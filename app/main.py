# app/main.py
from __future__ import annotations
import asyncio

from app.config.settings import Settings
from app.core.deps import Deps
from app.adapters.ollama_llm import OllamaChatAdapter
from app.adapters.firecrawl_search import FirecrawlSearchAdapter
from app.adapters.chroma_memory import ChromaMemoryAdapter
from app.adapters.trace_json import JsonTraceAdapter
from app.models.state import initial_state
from app.graph.build_graph import build_graph


async def main():
    s = Settings()

    deps = Deps(
        llm=OllamaChatAdapter(base_url=s.ollama_base_url, model=s.llm_model),
        small_llm=OllamaChatAdapter(
            base_url=s.ollama_base_url, model=s.small_llm_model
        ),
        memory=ChromaMemoryAdapter(
            persist_dir=s.chroma_persist_dir, collection_name=s.chroma_collection
        ),
        web=FirecrawlSearchAdapter(
            api_key=s.firecrawl_api_key, base_url=s.firecrawl_base_url
        ),
        trace=JsonTraceAdapter(trace_dir=s.trace_dir),
    )

    graph = build_graph(deps)
    state = initial_state()
    state["user_input"] = "hello"
    # NOTE: 実処理は stub のため、ここでは動作確認用の枠だけ
    _ = await graph.ainvoke(state)


if __name__ == "__main__":
    asyncio.run(main())
