# app/ports/trace.py
from __future__ import annotations
from typing import Any, Protocol


class TracePort(Protocol):
    async def write_turn(self, turn_id: int, payload: dict[str, Any]) -> None: ...
