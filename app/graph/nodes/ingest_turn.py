# app/graph/nodes/ingest_turn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.state import AgentState


@dataclass(frozen=True)
class IngestTurnIn:
    turn_id: int
    user_input: str
    wm_messages: list[dict[str, Any]]


@dataclass(frozen=True)
class IngestTurnOut:
    status: str
    turn_id: int
    wm_messages: list[dict[str, Any]]


def make_ingest_turn_node():
    def inner(inp: IngestTurnIn) -> IngestTurnOut:
        """
        何をするか:
        - user_input を wm_messages に追加（role='user'）
        - turn_id を進める（または「このターンのID」を確定させる）
        - working memory window のトリム（必要なら）
        """
        return IngestTurnOut(
            status="ingest_turn:stub",
            turn_id=inp.turn_id,
            wm_messages=inp.wm_messages,
        )

    def node(state: AgentState) -> dict:
        out = inner(
            IngestTurnIn(
                turn_id=state["turn_id"],
                user_input=state["user_input"],
                wm_messages=state["wm_messages"],
            )
        )
        return {
            "turn_id": out.turn_id,
            "wm_messages": out.wm_messages,
        }

    return node
