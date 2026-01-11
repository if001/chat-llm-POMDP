# app/core/deps.py
from __future__ import annotations
from dataclasses import dataclass
from app.ports.llm import LLMPort
from app.ports.search import WebSearchPort
from app.ports.memory import MemoryPort
from app.ports.trace import TracePort


@dataclass(frozen=True)
class Deps:
    llm: LLMPort
    small_llm: LLMPort
    memory: MemoryPort
    web: WebSearchPort
    trace: TracePort
