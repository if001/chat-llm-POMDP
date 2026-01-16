from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.server.engine import build_engine
from app.server.schema import (
    ChatRequest,
    ChatResponse,
    ChatResponseMessage,
    ChatStreamEvent,
)

app = FastAPI(title="Ollama-Compatible /api/chat Server", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE = build_engine()


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ndjson(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    created_at = now_rfc3339()

    # 超ざっくりの timing（必要ならあなたの実装で正確に）
    t0 = time.perf_counter_ns()

    if not req.stream:
        result = await ENGINE.chat_once(
            model=req.model,
            messages=req.messages,
            tools=req.tools,
            think=req.think,
        )
        t1 = time.perf_counter_ns()

        body = ChatResponse(
            model=req.model,
            created_at=created_at,
            message=ChatResponseMessage(
                role="assistant",
                content=result.content,
                thinking=result.thinking,
                tool_calls=result.tool_calls,
            ),
            done=True,
            done_reason="stop",
            total_duration=(t1 - t0),
        ).model_dump(exclude_none=True)

        return JSONResponse(content=body)

    async def stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in ENGINE.chat_stream(
                model=req.model,
                messages=req.messages,
                tools=req.tools,
                think=req.think,
            ):
                event = ChatStreamEvent(
                    model=req.model,
                    created_at=created_at,
                    message=ChatResponseMessage(
                        role="assistant",
                        content=chunk.content_delta,
                        thinking=chunk.thinking_delta or None,
                        tool_calls=chunk.tool_calls_delta,
                    ),
                    done=False,
                ).model_dump(exclude_none=True)
                yield ndjson(event)

            final = ChatStreamEvent(
                model=req.model,
                created_at=created_at,
                message=ChatResponseMessage(role="assistant", content=""),
                done=True,
                done_reason="stop",
            ).model_dump(exclude_none=True)
            yield ndjson(final)

        except Exception as e:
            # Ollama 互換のため、ストリーム途中の失敗は NDJSON の最終イベントとして返す
            err = {
                "model": req.model,
                "created_at": created_at,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "error",
                "error": str(e),
            }
            yield ndjson(err)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def run() -> None:
    import uvicorn

    print("start...")
    uvicorn.run(
        "ollama_compat_chat_server.main:app", host="0.0.0.0", port=11434, reload=False
    )
