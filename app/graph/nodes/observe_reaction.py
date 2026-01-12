# app/graph/nodes/observe_reaction.py
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState, Observation
from app.ports.llm import LLMPort


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
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


def _parse_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


async def _classify_reaction(
    small_llm: LLMPort,
    user_input: str,
    prev_assistant_text: str,
    prev_action: dict,
    fallback: Observation,
) -> Observation:
    prompt = (
        "Return JSON with keys: reaction_type, ack_type, events "
        "(E_correct, E_refuse, E_clarify, E_miss, E_frame_break, E_overstep), "
        "and confidence (0-1)."
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"user_input: {user_input}\n"
                        f"prev_assistant_text: {prev_assistant_text}\n"
                        f"prev_action: {prev_action}"
                    ),
                },
            ]
        )
    except Exception:
        return fallback

    payload = _parse_json(_get_content(result))
    if not payload:
        return fallback

    events = payload.get("events", {})
    fallback_events = fallback["events"]
    merged_events = {
        "E_correct": _coerce_int(events.get("E_correct", fallback_events["E_correct"]), 0),
        "E_refuse": _coerce_int(events.get("E_refuse", fallback_events["E_refuse"]), 0),
        "E_clarify": _coerce_int(events.get("E_clarify", fallback_events["E_clarify"]), 0),
        "E_miss": _coerce_int(events.get("E_miss", fallback_events["E_miss"]), 0),
        "E_frame_break": _coerce_int(
            events.get("E_frame_break", fallback_events["E_frame_break"]), 0
        ),
        "E_overstep": _coerce_int(
            events.get("E_overstep", fallback_events["E_overstep"]), 0
        ),
    }

    return {
        "reaction_type": payload.get("reaction_type", fallback["reaction_type"]),
        "ack_type": payload.get("ack_type", fallback["ack_type"]),
        "events": merged_events,
        "confidence": _coerce_float(payload.get("confidence", fallback["confidence"]), 0.0),
    }


def make_observe_reaction_node(deps: Deps):
    async def inner(inp: ObserveReactionIn) -> ObserveReactionOut:
        """
        何をするか:
        - 前ターンのassistant出力(prev_assistant_text)に対する、今回user_inputの反応を分類
          - reaction_type / ack_type
          - events(E_correct/E_refuse/E_clarify/E_miss/E_frame_break/E_overstep)
        - small LLM vote + 構造特徴（質問未回答など）で confidence を算出
        """
        fallback: Observation = {
            "reaction_type": "mixed",
            "ack_type": "none",
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
        obs = await _classify_reaction(
            deps.small_llm,
            inp.user_input,
            inp.prev_assistant_text,
            inp.prev_action,
            fallback,
        )
        return ObserveReactionOut(status="observe_reaction:ok", observation=obs)

    async def node(state: AgentState) -> dict:
        out = await inner(
            ObserveReactionIn(
                turn_id=state["turn_id"],
                user_input=state["user_input"],
                prev_assistant_text=state["last_turn"]["prev_assistant_text"],
                prev_action=state["last_turn"]["prev_action"],
            )
        )
        return {"observation": out.observation}

    return node
