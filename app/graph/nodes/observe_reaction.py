# app/graph/nodes/observe_reaction.py
from __future__ import annotations

from dataclasses import dataclass

from app.models.state import AgentState, Observation


@dataclass(frozen=True)
class ObserveReactionIn:
    turn_id: int
    user_input: str
    prev_assistant_text: str
    prev_action: dict


@dataclass(frozen=True)
class ObserveReactionOut:
    status: str
    observation: Observation


def make_observe_reaction_node():
    def inner(inp: ObserveReactionIn) -> ObserveReactionOut:
        """
        何をするか:
        - 前ターンのassistant出力(prev_assistant_text)に対する、今回user_inputの反応を分類
          - reaction_type / ack_type
          - events(E_correct/E_refuse/E_clarify/E_miss/E_frame_break/E_overstep)
        - small LLM vote + 構造特徴（質問未回答など）で confidence を算出
        """
        obs: Observation = {
            "reaction_type": "mixed",
            "ack_type": "none",
            "events": {
                "E_correct": 0,
                "E_refuse": 0,
                "E_clarify": 0,
                "E_miss": 0,
                "E_frame_break": 0,
                "E_overstep": 0,
            },
            "confidence": 0.0,
        }
        return ObserveReactionOut(status="observe_reaction:stub", observation=obs)

    def node(state: AgentState) -> dict:
        out = inner(
            ObserveReactionIn(
                turn_id=state["turn_id"],
                user_input=state["user_input"],
                prev_assistant_text=state["last_turn"]["prev_assistant_text"],
                prev_action=state["last_turn"]["prev_action"],
            )
        )
        return {"observation": out.observation}

    return node
