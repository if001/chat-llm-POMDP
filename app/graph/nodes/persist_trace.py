# app/graph/nodes/persist_trace.py
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

from app.core.deps import Deps
from app.models.state import AgentState


@dataclass(frozen=True)
class PersistTraceIn:
    turn_id: int
    payload: dict


@dataclass(frozen=True)
class PersistTraceOut:
    status: str


def _build_learning_state(state: AgentState) -> dict:
    return {
        "joint_context": deepcopy(state["joint_context"]),
        "common_ground": deepcopy(state["common_ground"]),
        "unresolved_points": deepcopy(state["unresolved_points"]),
        "affective_state": deepcopy(state["affective_state"]),
        "epistemic_state": deepcopy(state["epistemic_state"]),
        "user_model": deepcopy(state["user_model"]),
        "policy": deepcopy(state["policy"]),
    }


def _build_trace_payload(state: AgentState) -> dict:
    return {
        "turn": {
            "id": state["turn_id"],
            "user_input": state["user_input"],
        },
        "learning_state": _build_learning_state(state),
        "signals": {
            "predictions": deepcopy(state["predictions"]),
            "observation": deepcopy(state["observation"]),
            "metrics": deepcopy(state["metrics"]),
            "deep_decision": deepcopy(state["deep_decision"]),
        },
        "action": deepcopy(state["action"]),
        "response": deepcopy(state["response"]),
    }


def make_persist_trace_node(deps: Deps):
    async def inner(inp: PersistTraceIn) -> PersistTraceOut:
        """
        何をするか:
        - turn_id ごとのstateをJSONファイルに保存
          - ターン毎に永続化すべきstateのみを保存
        """
        # await deps.trace.write_turn(inp.turn_id, inp.payload)
        await deps.trace.write_state(inp.payload)
        return PersistTraceOut(status="persist_trace:ok")

    async def node(state: AgentState) -> dict:
        payload = _build_trace_payload(state)
        _ = await inner(PersistTraceIn(turn_id=state["turn_id"], payload=payload))
        return {}

    return node
