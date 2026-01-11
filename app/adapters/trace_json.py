# app/adapters/trace_json.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from app.ports.trace import TracePort


class JsonTraceAdapter(TracePort):
    def __init__(self, *, trace_dir: str) -> None:
        self._dir = Path(trace_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    async def write_turn(self, turn_id: int, payload: dict[str, Any]) -> None:
        path = self._dir / f"turn_{turn_id:06d}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
