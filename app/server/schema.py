from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ToolCallFunction(BaseModel):
    name: str
    description: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


class ToolCall(BaseModel):
    function: ToolCallFunction


class ToolDefinitionFunction(BaseModel):
    name: str
    parameters: Dict[str, Any]
    description: Optional[str] = None


class ToolDefinition(BaseModel):
    type: Literal["function"]
    function: ToolDefinitionFunction


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    images: Optional[List[str]] = None
    tool_calls: Optional[List[ToolCall]] = None


class ModelOptions(BaseModel):
    # OpenAPI上は additionalProperties: true なので柔軟に受ける
    # 代表例のみ型を置き、未知キーは許容
    seed: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    num_ctx: Optional[int] = None
    num_predict: Optional[int] = None

    model_config = {"extra": "allow"}


ThinkType = Union[bool, Literal["high", "medium", "low"]]


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[ToolDefinition]] = None
    format: Optional[Union[Literal["json"], Dict[str, Any]]] = None
    options: Optional[ModelOptions] = None
    stream: bool = True
    think: Optional[ThinkType] = None
    keep_alive: Optional[Union[str, float, int]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None


class ChatResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str = ""
    thinking: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    images: Optional[List[str]] = None


class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: ChatResponseMessage
    done: bool
    done_reason: Optional[str] = None

    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

    # logprobs はここでは省略実装（必要なら追加）
    logprobs: Optional[list[Any]] = None


class ChatStreamEvent(BaseModel):
    model: str
    created_at: str
    message: ChatResponseMessage
    done: bool
    done_reason: Optional[str] = None
