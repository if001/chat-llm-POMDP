# app/graph/nodes/persist_trace.py
from __future__ import annotations

from dataclasses import dataclass

from app.core.deps import Deps
from app.models.state import AgentState


@dataclass(frozen=True)
class PersistTraceIn:
    turn_id: int
    payload: dict


@dataclass(frozen=True)
class PersistTraceOut:
    status: str


def make_persist_trace_node(deps: Deps):
    def inner(inp: PersistTraceIn) -> PersistTraceOut:
        """
        何をするか:
        - turn_id ごとの trace を永続化（JSON or Postgres）
          - stateのサマリ
          - action / observation / metrics / deep_decision
          - 重要なら wm_messages の一部
        """
        # TODO: await deps.trace.write_turn(inp.turn_id, inp.payload)
        return PersistTraceOut(status="persist_trace:stub")

    def node(state: AgentState) -> dict:
        _ = inner(PersistTraceIn(turn_id=state["turn_id"], payload={"state": state}))
        return {}

    return node
