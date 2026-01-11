# app/graph/nodes/observe_reaction.py
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from app.core.deps import Deps
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


def _get_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    return getattr(result, "content", "")


def _run_async(coro):
    return asyncio.run(coro)


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def make_observe_reaction_node(deps: Deps):
    def inner(inp: ObserveReactionIn) -> ObserveReactionOut:
        """
        何をするか:
        - 前ターンのassistant出力(prev_assistant_text)に対する、今回user_inputの反応を分類
          - reaction_type / ack_type
          - events(E_correct/E_refuse/E_clarify/E_miss/E_frame_break/E_overstep)
        - small LLM vote + 構造特徴（質問未回答など）で confidence を算出
        """
        miss = 0
        if inp.prev_action.get("questions_asked", 0) > 0 and not inp.user_input.strip():
            miss = 1
        prompt = (
            "Classify user reaction to the previous assistant output. "
            "Return JSON with keys reaction_type and ack_type."
        )
        response = _run_async(
            deps.small_llm.ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"prev:{inp.prev_assistant_text}\nuser:{inp.user_input}",
                    },
                ]
            )
        )
        parsed = _parse_json(_get_content(response))
        obs: Observation = {
            "reaction_type": parsed.get("reaction_type", "mixed"),
            "ack_type": parsed.get("ack_type", "none"),
            "events": {
                "E_correct": 0,
                "E_refuse": 0,
                "E_clarify": 0,
                "E_miss": miss,
                "E_frame_break": 0,
                "E_overstep": 0,
            },
            "confidence": 0.0,
        }
        return ObserveReactionOut(status="observe_reaction:ok", observation=obs)

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
